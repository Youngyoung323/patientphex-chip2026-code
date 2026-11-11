"""Dependency-free PatientPheX baseline and local reference evaluation.

This is an independent lexical baseline, not the organizers' PhenoTagger system.
All offsets are Unicode character offsets within the original BioC passages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HPO_VERSION = "2026-06-23"
HPO_ROOT = "HP:0000118"
HPO_ID = re.compile(r"HP:\d{7}\Z")
TOKEN = re.compile(r"\w+", re.UNICODE)
BOUNDARY = re.compile(r"[;!?\n]|\.(?=\s|$)|\b(?:but|however|yet|although)\b", re.I)
NEGATION = re.compile(r"\b(?:no|without|denies|denied|neither|absence of|negative for|free of|not have)\b", re.I)
NEGATED_AFTER = re.compile(r"^\s*(?:(?:was|were|is|are)\s+)?(?:absent|not (?:present|seen|observed|detected))\b", re.I)
GENERAL = re.compile(
    r"\b(?:usually|typically|may|literature|patients with|previously reported|characterized by)\b", re.I
)
NON_BODY_SECTIONS = {
    "REF",
    "REFS",
    "REFERENCE",
    "REFERENCES",
    "REF-LIST",
    "ACK",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "FUNDING",
    "AUTH_CONT",
    "COMP_INT",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"{path}:{number}: blank JSONL line")
            record = json.loads(line)
            require(isinstance(record, dict), f"{path}:{number}: expected an object")
            records.append(record)
    require(bool(records), f"{path}: empty file")
    return records


def write_json(path, value, jsonl=False):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Completed predictions and reports must never be overwritten implicitly.
    with destination.open("x", encoding="utf-8") as handle:
        if jsonl:
            for row in value:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(text):
    return tuple(match.group().casefold() for match in TOKEN.finditer(text))


def identifier_parts(value):
    if value == "-1" or (type(value) is int and value == -1):
        return ("-1",)
    require(isinstance(value, str), "entity identifier must be an HPO string or -1")
    parts = tuple(part.strip() for part in value.split(";"))
    require(
        bool(parts) and all(part == "-1" or HPO_ID.fullmatch(part) for part in parts),
        f"invalid HPO identifier: {value!r}",
    )
    require(len(parts) == len(set(parts)), f"duplicate compound HPO identifiers: {value!r}")
    return tuple(sorted(parts))


def gold_identifier_parts(value):
    """Read released labels literally; never guess corrections to gold IDs."""
    if type(value) is int and value == -1:
        return ("-1",)
    require(isinstance(value, str) and bool(value.strip()), "gold identifier must be a nonempty string or -1")
    parts = tuple(part.strip() for part in value.split(";"))
    require(all(parts), f"empty component in gold identifier: {value!r}")
    return parts


def negative(entity):
    return str(entity.get("note") or "").upper() == "NO"


def scored_passage(passage):
    # Exact metadata values only; do not guess from clinical text or section titles.
    return not any(
        str(passage.get(field, "")).strip().upper() in NON_BODY_SECTIONS for field in ["section_type", "type"]
    )


def scored_entity(entity, source):
    return any(
        scored_passage(passage)
        and passage["offset"] <= entity["offset"]
        and entity["offset"] + entity["length"] <= passage["offset"] + len(passage["text"])
        for passage in source["full_text"]
    )


def span_text(document, offset, length):
    require(type(offset) is int and offset >= 0, "offset must be a nonnegative integer")
    require(type(length) is int and length > 0, "length must be a positive integer")
    for passage in document["full_text"]:
        start = passage["offset"]
        if start <= offset and offset + length <= start + len(passage["text"]):
            return passage["text"][offset - start : offset - start + length]
    raise ValueError(f"{document['pmc_id']}: span ({offset}, {length}) falls outside a passage")


def index_documents(documents, source=False):
    result = {}
    for document in documents:
        require(isinstance(document, dict), "document must be an object")
        key = document.get("pmc_id")
        require(isinstance(key, str) and bool(key.strip()), "pmc_id must be a nonempty string")
        require(key not in result, f"duplicate pmc_id: {key}")
        require("pmid" in document, f"{key}: missing pmid")
        require(document["pmid"] is None or isinstance(document["pmid"], str), f"{key}: pmid must be a string or null")
        result[key] = document
        if not source:
            continue
        passages = document.get("full_text")
        patients = document.get("patient")
        require(isinstance(passages, list) and bool(passages), f"{key}: missing full_text passages")
        require(isinstance(patients, list), f"{key}: missing patient list")
        ends = []
        for passage in passages:
            require(isinstance(passage, dict), f"{key}: passage must be an object")
            require(type(passage.get("offset")) is int and passage["offset"] >= 0, f"{key}: invalid passage offset")
            require(isinstance(passage.get("text"), str), f"{key}: passage text must be a string")
            if passage["text"]:
                ends.append((passage["offset"], passage["offset"] + len(passage["text"])))
        ends.sort()
        require(all(a[1] <= b[0] for a, b in zip(ends, ends[1:], strict=False)), f"{key}: overlapping passages")
        patient_ids = set()
        for patient in patients:
            require(isinstance(patient, dict), f"{key}: patient must be an object")
            patient_id = patient.get("patient_id")
            require(isinstance(patient_id, str) and bool(patient_id.strip()), f"{key}: invalid patient_id")
            require(patient_id not in patient_ids, f"{key}: duplicate patient_id: {patient_id}")
            patient_ids.add(patient_id)
            require(isinstance(patient.get("mention"), list), f"{key}/{patient_id}: missing mention list")
            for mention in patient["mention"]:
                require(isinstance(mention, dict), f"{key}: mention must be an object")
                require(
                    span_text(document, mention.get("offset"), mention.get("length")) == mention.get("text"),
                    f"{key}/{patient_id}: patient mention text disagrees with offset/length",
                )
    require(bool(result), "no documents")
    return result


def validate_predictions(sources, predictions, ontology=None):
    expected = index_documents(sources, source=True)
    actual = index_documents(predictions)
    require(
        set(expected) == set(actual),
        f"pmc_id mismatch: missing={sorted(set(expected) - set(actual))}, extra={sorted(set(actual) - set(expected))}",
    )
    totals = Counter(documents=len(expected))
    for key, source in expected.items():
        prediction = actual[key]
        require(source["pmid"] == prediction["pmid"], f"{key}: pmid mismatch")
        require(isinstance(prediction.get("entities"), list), f"{key}: missing entities list")
        require(isinstance(prediction.get("association"), list), f"{key}: missing association list")
        units = set()
        for entity in prediction["entities"]:
            require(isinstance(entity, dict), f"{key}: entity must be an object")
            require(entity.get("type") == "Phenotype", f"{key}: entity type must be Phenotype")
            require(entity.get("note") is None or str(entity["note"]).upper() == "NO", f"{key}: invalid note")
            parts = identifier_parts(entity.get("identifier"))
            require(
                span_text(source, entity.get("offset"), entity.get("length")) == entity.get("text"),
                f"{key}: entity text disagrees with offset/length",
            )
            for part in parts:
                if part != "-1" and ontology is not None:
                    require(part in ontology["terms"], f"{key}: HPO ID outside frozen phenotype branch: {part}")
                unit = (entity["offset"], entity["length"], part)
                require(unit not in units, f"{key}: duplicate or contradictory entity unit: {unit}")
                units.add(unit)
            totals["entities"] += 1
            totals["negative_entities"] += negative(entity)
        wanted = {patient["patient_id"] for patient in source["patient"]}
        found = set()
        for association in prediction["association"]:
            require(isinstance(association, dict), f"{key}: association must be an object")
            patient_id = association.get("patient_id")
            require(isinstance(patient_id, str), f"{key}: association patient_id must be a string")
            require(patient_id in wanted and patient_id not in found, f"{key}: unknown/duplicate patient: {patient_id}")
            found.add(patient_id)
            phenotypes = association.get("phenotype")
            require(isinstance(phenotypes, list), f"{key}/{patient_id}: phenotype must be a list")
            require(
                all(isinstance(item, str) and bool(item.strip()) for item in phenotypes),
                f"{key}/{patient_id}: phenotype entries must be nonempty strings",
            )
            require(len(phenotypes) == len(set(phenotypes)), f"{key}/{patient_id}: duplicate association")
            for item in phenotypes:
                require(item != "-1", f"{key}/{patient_id}: unmapped associations need their original text")
                if item.startswith("HP:"):
                    require(bool(HPO_ID.fullmatch(item)), f"{key}/{patient_id}: split compound association IDs")
                    if ontology is not None:
                        require(item in ontology["terms"], f"{key}: invalid association HPO ID: {item}")
            totals["patient_phenotype_links"] += len(phenotypes)
        require(found == wanted, f"{key}: missing patient predictions: {sorted(wanted - found)}")
        totals["patients"] += len(wanted)
    return dict(totals)


def gold_association_set(values, compound_policy="split_hpo"):
    require(compound_policy in {"split_hpo", "literal"}, "unknown gold association compound policy")
    result = set()
    for value in values:
        parts = tuple(part.strip() for part in value.split(";"))
        if compound_policy == "split_hpo" and len(parts) > 1 and all(part.startswith("HP:") for part in parts):
            # Retain malformed/out-of-branch units as gold rather than deleting them.
            result.update(parts)
        else:
            result.add(value)
    return result


def validate_gold(documents, ontology=None):
    """Validate source structure while reporting, not rewriting, label anomalies."""
    index_documents(documents, source=True)
    counts = Counter(documents=len(documents))
    issues = []

    def issue(kind, document, **details):
        counts[kind] += 1
        issues.append({"kind": kind, "pmc_id": document["pmc_id"], **details})

    for document in documents:
        key = document["pmc_id"]
        require(isinstance(document.get("entities"), list), f"{key}: missing gold entities")
        require(isinstance(document.get("association"), list), f"{key}: missing gold association")
        units = set()
        for index, entity in enumerate(document["entities"]):
            require(isinstance(entity, dict), f"{key}: gold entity must be an object")
            require(entity.get("type") == "Phenotype", f"{key}: invalid gold entity type")
            require(
                span_text(document, entity.get("offset"), entity.get("length")) == entity.get("text"),
                f"{key}: gold entity text disagrees with offset/length",
            )
            require(entity.get("note") is None or isinstance(entity["note"], str), f"{key}: invalid gold note")
            if entity.get("note") is not None and not negative(entity):
                issue("nonstandard_note", document, entity_index=index, note=entity["note"])
            parts = gold_identifier_parts(entity.get("identifier"))
            for part in parts:
                unit = (entity["offset"], entity["length"], part)
                if unit in units:
                    issue("duplicate_entity_unit", document, entity_index=index, identifier=part)
                units.add(unit)
                if part != "-1" and not HPO_ID.fullmatch(part):
                    issue("nonstandard_identifier_unit", document, entity_index=index, identifier=part)
                if ontology is not None and part != "-1" and part not in ontology["terms"]:
                    target = ontology.get("canonical_ids", {}).get(part)
                    issue(
                        "identifier_outside_active_branch",
                        document,
                        entity_index=index,
                        identifier=part,
                        canonical_training_target=target,
                    )
            counts["entities"] += 1
            counts["negative_entities"] += negative(entity)
        wanted = {patient["patient_id"] for patient in document["patient"]}
        found = set()
        for association in document["association"]:
            require(isinstance(association, dict), f"{key}: gold association must be an object")
            patient_id = association.get("patient_id")
            require(isinstance(patient_id, str), f"{key}: invalid gold patient_id")
            require(patient_id in wanted and patient_id not in found, f"{key}: unknown/duplicate gold patient")
            found.add(patient_id)
            values = association.get("phenotype")
            require(isinstance(values, list), f"{key}: gold phenotype must be a list")
            require(
                all(isinstance(value, str) and bool(value.strip()) for value in values),
                f"{key}: gold phenotype entries must be nonempty strings",
            )
            for value, count in Counter(values).items():
                if count > 1:
                    issue(
                        "duplicate_association_value", document, patient_id=patient_id, value=value, repeats=count - 1
                    )
                    counts["duplicate_association_excess"] += count - 1
                if value.startswith("HP:") and ";" in value:
                    issue("compound_association_value", document, patient_id=patient_id, value=value)
            counts["patient_links_raw"] += len(values)
            counts["patient_links_literal_unique"] += len(set(values))
            counts["patient_links_split_unique"] += len(gold_association_set(values))
        require(found == wanted, f"{key}: missing gold patients: {sorted(wanted - found)}")
        counts["patients"] += len(wanted)
    return {"counts": dict(counts), "issues": issues, "gold_labels_rewritten": False}


def load_ontology(path):
    text = Path(path).read_text(encoding="utf-8")
    require(f"data-version: hp/releases/{HPO_VERSION}\n" in text[:500], "incorrect HPO release; need 2026-06-23")
    all_terms = {}
    for block in text.split("\n[Term]\n")[1:]:
        block = block.split("\n[", 1)[0]
        term = {"parents": [], "synonyms": [], "alt_ids": [], "replaced_by": []}
        for line in block.splitlines():
            if line.startswith("id: "):
                term["id"] = line[4:]
            elif line.startswith("name: "):
                term["name"] = line[6:]
            elif line.startswith("is_a: "):
                term["parents"].append(line[6:].split()[0])
            elif line.startswith("alt_id: "):
                term["alt_ids"].append(line[8:].split()[0])
            elif line.startswith("replaced_by: "):
                term["replaced_by"].append(line[13:].split()[0])
            elif line.startswith("synonym: "):
                match = re.match(r'synonym: ("(?:[^"\\]|\\.)*") EXACT\b', line)
                if match:
                    # OBO uses quoted strings; decode only escaped quote/backslash.
                    value = match[1][1:-1].replace(r"\"", '"').replace(r"\\", "\\")
                    term["synonyms"].append(value)
            elif line == "is_obsolete: true":
                term["obsolete"] = True
        if term.get("id"):
            all_terms[term["id"]] = term
    require(HPO_ROOT in all_terms, "HPO phenotype root missing")
    children = defaultdict(set)
    for key, term in all_terms.items():
        if term.get("obsolete"):
            continue
        for parent in term["parents"]:
            children[parent].add(key)
    branch, frontier = set(), [HPO_ROOT]
    while frontier:
        key = frontier.pop()
        if key not in branch:
            branch.add(key)
            frontier.extend(children[key])
    replacements = defaultdict(set)
    for key, term in all_terms.items():
        if term.get("obsolete"):
            replacements[key].update(term["replaced_by"])
        else:
            for alias in term["alt_ids"]:
                replacements[alias].add(key)

    def resolve(key, visited):
        if key in branch:
            return key
        if key in visited or len(replacements[key]) != 1:
            return None
        return resolve(next(iter(replacements[key])), visited | {key})

    canonical = {}
    for key in sorted(set(all_terms) | set(replacements)):
        target = resolve(key, set())
        if target:
            canonical[key] = target
    return {
        "version": HPO_VERSION,
        "sha256": digest(path),
        "terms": {key: all_terms[key] for key in sorted(branch)},
        "canonical_ids": canonical,
    }


def source_fingerprint(document):
    passages = [(passage["offset"], passage["text"]) for passage in document["full_text"]]
    return hashlib.sha256(json.dumps(passages, ensure_ascii=False).encode()).hexdigest()


def build_model(ontology, training):
    audit = validate_gold(training, ontology) if training else {"counts": {}, "issues": []}
    candidates = defaultdict(dict)
    for key, term in ontology["terms"].items():
        for phrase, priority in [(term["name"], 2)] + [(phrase, 1) for phrase in term["synonyms"]]:
            normalized = tokens(phrase)
            if not normalized or (len(normalized) == 1 and len(normalized[0]) < 4):
                continue
            candidates[normalized][key] = max(priority, candidates[normalized].get(key, 0))
    learned = defaultdict(Counter)
    training_stats = Counter()
    for document in training:
        seen = set()
        for entity in document["entities"]:
            if not scored_entity(entity, document):
                continue
            normalized = tokens(entity["text"])
            parts = gold_identifier_parts(entity["identifier"])
            canonical = tuple(
                "-1" if part == "-1" else ontology.get("canonical_ids", {}).get(part, part) for part in parts
            )
            if any(part != "-1" and part not in ontology["terms"] for part in canonical):
                training_stats["entities_excluded_from_alias_learning"] += 1
                continue
            if canonical != parts:
                training_stats["entities_with_ontology_declared_remapping"] += 1
            unit = (entity["offset"], entity["length"], tuple(sorted(set(canonical))))
            if normalized and unit not in seen:
                learned[normalized][";".join(sorted(set(canonical)))] += 1
                training_stats["entities_used_for_alias_learning"] += 1
            seen.add(unit)
    aliases = {}
    for normalized in sorted(set(candidates) | set(learned)):
        if normalized in learned:
            scores = learned[normalized]
        else:
            scores = candidates[normalized]
        best = max(scores.values())
        winners = [key for key, score in scores.items() if score == best]
        if len(winners) == 1:
            aliases[" ".join(normalized)] = winners[0]
    return {
        "schema_version": 1,
        "method": "lexical_cpu_v2_gold_compatible",
        "hpo_version": ontology["version"],
        "hpo_sha256": ontology["sha256"],
        "training_documents": [
            {"pmc_id": d["pmc_id"], "pmid": d["pmid"], "text_sha256": source_fingerprint(d)} for d in training
        ],
        "aliases": aliases,
        "training_audit_counts": audit["counts"],
        "training_alias_stats": dict(training_stats),
        "training_label_policy": (
            "Use only active phenotype IDs, ontology-declared unambiguous alt/replaced IDs and explicit -1. "
            "Skip a whole training entity if any component is unresolved; do not rewrite evaluation gold."
        ),
    }


class Matcher:
    def __init__(self, model):
        require(model.get("schema_version") == 1, "unsupported baseline model schema")
        require(model.get("hpo_version") == HPO_VERSION, "model has incorrect HPO version")
        self.trie = {}
        for alias, identifier in model["aliases"].items():
            node = self.trie
            for token in alias.split():
                node = node.setdefault(token, {})
            node[None] = identifier

    def entities(self, passage):
        text = passage["text"]
        words = list(TOKEN.finditer(text))
        result, index = [], 0
        while index < len(words):
            node, position, best = self.trie, index, None
            while position < len(words) and words[position].group().casefold() in node:
                if position > index and re.search(
                    r"[;.!?\n]", text[words[position - 1].end() : words[position].start()]
                ):
                    break
                node = node[words[position].group().casefold()]
                position += 1
                if None in node:
                    best = (position, node[None])
            if best is None:
                index += 1
                continue
            stop, identifier = best
            start, end = words[index].start(), words[stop - 1].end()
            boundaries = list(BOUNDARY.finditer(text[:start]))
            preceding = text[boundaries[-1].end() if boundaries else 0 : start][-160:]
            preceding = re.sub(r"\bnot only\b", "", preceding, flags=re.I)
            negated = bool(NEGATION.search(preceding) or NEGATED_AFTER.search(text[end : end + 80]))
            result.append(
                {
                    "identifier": identifier,
                    "type": "Phenotype",
                    "offset": passage["offset"] + start,
                    "length": end - start,
                    "text": text[start:end],
                    "note": "NO" if negated else None,
                }
            )
            index = stop
        return result


def patient_anchors(document, passage):
    aliases = defaultdict(set)
    for patient in document["patient"]:
        aliases[patient["patient_id"]].add(patient["patient_id"])
        for mention in patient["mention"]:
            aliases[mention["text"]].add(patient["patient_id"])
    anchors = set()
    for phrase, patient_ids in aliases.items():
        if len(patient_ids) != 1 or not phrase.strip():
            continue
        for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", passage["text"], re.I):
            anchors.add((match.start(), match.end(), next(iter(patient_ids))))
    # Preserve supplied offsets even when an alias is shared by several people.
    for patient in document["patient"]:
        for mention in patient["mention"]:
            local = mention["offset"] - passage["offset"]
            if 0 <= local and local + mention["length"] <= len(passage["text"]):
                anchors.add((local, local + mention["length"], patient["patient_id"]))
    return sorted(anchors)


def assign_patient(entity, passage, anchors):
    if negative(entity):
        return [], "negated"
    text = passage["text"]
    start = entity["offset"] - passage["offset"]
    end = start + entity["length"]
    boundaries = list(BOUNDARY.finditer(text[:start]))
    left = boundaries[-1].end() if boundaries else 0
    following = BOUNDARY.search(text, end)
    right = following.start() if following else len(text)
    before = [anchor for anchor in anchors if left <= anchor[0] and anchor[1] <= start]
    if before:
        nearest = max(before, key=lambda anchor: (anchor[1], anchor[0]))
        assigned = {anchor[2] for anchor in before if anchor[:2] == nearest[:2]}
        others = [anchor for anchor in before if anchor[1] <= nearest[0] and anchor[2] not in assigned]
        if others:
            previous = max(others, key=lambda anchor: anchor[1])
            connector = text[previous[1] : nearest[0]].strip()
            if re.fullmatch(r"(?:and|&|,|,\s*and)", connector, re.I):
                assigned.add(previous[2])
        return sorted(assigned), "explicit_same_clause"
    after = [anchor for anchor in anchors if end <= anchor[0] < right]
    if after:
        nearest = min(after)
        if re.search(r"\b(?:in|of)\s*$", text[end : nearest[0]]) and nearest[0] - end < 80:
            return [nearest[2]], "explicit_following"
    previous = [anchor for anchor in anchors if anchor[1] <= left and start - anchor[1] < 800]
    clinical = re.search(r"case|clinical|presentation|patient|history", passage.get("section_type", ""), re.I)
    coreference = re.search(r"\b(?:he|she|his|her|the patient|the proband)\b", text[left:right], re.I)
    if previous and (clinical or coreference) and not GENERAL.search(text[left:right]):
        nearest = max(previous, key=lambda anchor: anchor[1])
        tied = {anchor[2] for anchor in previous if anchor[:2] == nearest[:2]}
        if len(tied) == 1:
            return [nearest[2]], "within_passage_coreference_heuristic"
    return [], "unresolved"


def predict(model, documents, allow_training_overlap=False):
    index_documents(documents, source=True)
    seen_ids = {row["pmc_id"] for row in model["training_documents"]}
    seen_pmids = {row["pmid"] for row in model["training_documents"] if row["pmid"]}
    seen_text = {row["text_sha256"] for row in model["training_documents"]}
    overlap = [
        d["pmc_id"]
        for d in documents
        if d["pmc_id"] in seen_ids or d["pmid"] in seen_pmids or source_fingerprint(d) in seen_text
    ]
    require(allow_training_overlap or not overlap, f"training/inference overlap: {overlap}; use held-out papers")
    matcher = Matcher(model)
    predictions, evidence = [], []
    for document in documents:
        entities, decisions = [], []
        association = {patient["patient_id"]: set() for patient in document["patient"]}
        for passage in document["full_text"]:
            if not scored_passage(passage):
                continue
            anchors = patient_anchors(document, passage)
            for entity in matcher.entities(passage):
                entities.append(entity)
                patient_ids, method = assign_patient(entity, passage, anchors)
                for patient_id in patient_ids:
                    parts = identifier_parts(entity["identifier"])
                    association[patient_id].update(entity["text"] if part == "-1" else part for part in parts)
                decisions.append(
                    {
                        "offset": entity["offset"],
                        "length": entity["length"],
                        "identifier": entity["identifier"],
                        "patient_ids": patient_ids,
                        "method": method,
                    }
                )
        predictions.append(
            {
                "pmc_id": document["pmc_id"],
                "pmid": document["pmid"],
                "entities": sorted(entities, key=lambda entity: entity["offset"]),
                "association": [
                    {"patient_id": key, "phenotype": sorted(values)} for key, values in association.items()
                ],
            }
        )
        evidence.append({"pmc_id": document["pmc_id"], "decisions": decisions})
    validate_predictions(documents, predictions)
    return predictions, evidence


def metric(tp, fp, fn, empty_perfect=False):
    if empty_perfect and tp + fp + fn == 0:
        precision = recall = f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compare(gold, prediction):
    return len(gold & prediction), len(prediction - gold), len(gold - prediction)


def compare_mentions(gold, prediction):
    # A gold -1 means there is no normalization target: match its boundaries.
    exact = {unit for unit in gold & prediction if unit[2] != "-1"}
    remaining_gold, remaining_prediction = gold - exact, prediction - exact
    matched = len(exact)
    for unit in sorted(remaining_gold):
        if unit[2] != "-1":
            continue
        same_span = sorted(candidate for candidate in remaining_prediction if candidate[:2] == unit[:2])
        if same_span:
            remaining_prediction.remove(same_span[0])
            matched += 1
    return matched, len(prediction) - matched, len(gold) - matched


def entity_units(document, source=None, gold=False):
    source = source if source is not None else document
    mentions, concepts = set(), set()
    for entity in document["entities"]:
        if negative(entity) or not scored_entity(entity, source):
            continue
        for part in (gold_identifier_parts if gold else identifier_parts)(entity["identifier"]):
            mentions.add((entity["offset"], entity["length"], part))
            if part != "-1":
                concepts.add(part)
    return mentions, concepts


def evaluate(gold, predictions, association_compounds="split_hpo"):
    audit = validate_gold(gold)
    validate_predictions(gold, predictions)
    predicted = index_documents(predictions)
    counts = {name: [0, 0, 0] for name in ["mention", "document", "patient_micro"]}
    patient_metrics = []
    for document in gold:
        output = predicted[document["pmc_id"]]
        for name, expected, actual in zip(
            ["mention", "document"],
            entity_units(document, gold=True),
            entity_units(output, source=document),
            strict=True,
        ):
            values = compare_mentions(expected, actual) if name == "mention" else compare(expected, actual)
            counts[name] = [a + b for a, b in zip(counts[name], values, strict=True)]
        actual = {item["patient_id"]: set(item["phenotype"]) for item in output["association"]}
        for item in document["association"]:
            values = compare(gold_association_set(item["phenotype"], association_compounds), actual[item["patient_id"]])
            counts["patient_micro"] = [a + b for a, b in zip(counts["patient_micro"], values, strict=True)]
            patient_metrics.append(metric(*values, empty_perfect=True))
    result = {name: metric(*values) for name, values in counts.items()}
    result["patient_macro"] = {
        name: sum(row[name] for row in patient_metrics) / len(patient_metrics) if patient_metrics else 0.0
        for name in ["precision", "recall", "f1"]
    }
    result["score"] = sum(result[name]["f1"] for name in [*counts, "patient_macro"]) / 4
    result["score_percent"] = result["score"] * 100
    result["patients"] = len(patient_metrics)
    result["gold_audit_counts"] = audit["counts"]
    result["gold_association_compound_policy"] = association_compounds
    result["evaluation_status"] = "local_reference_only_not_official_scorer"
    result["assumptions"] = [
        "Gold unmapped (-1) mentions match one prediction by offset/length regardless of predicted ID; "
        "excluded from document HPO sets.",
        "Negated predictions are excluded; positive predictions on gold-negated spans are false positives.",
        "Empty global mention/document/micro denominators yield zero; both-empty patient sets yield one.",
        "Known reference/acknowledgment/funding metadata are excluded. Other non-body categories remain unspecified; "
        "title/abstract are retained pending official scorer verification.",
        "Duplicate gold entries are sets. Only NO is negative; other notes remain positive. "
        "Malformed, obsolete and out-of-branch gold IDs are retained literally, not dropped or repaired.",
        f"Gold patient compounds use {association_compounds!r}; report the alternative as scoring sensitivity.",
        "Public rules do not fully specify these edge cases; verify against the organizers' scorer when available.",
    ]
    return result


def split_training(documents, fraction=0.2, seed=42):
    index_documents(documents, source=True)
    require(0 < fraction < 1, "validation fraction must lie strictly between zero and one")
    # Group shared PMID and identical input text together, including transitive links.
    groups = []
    for document in documents:
        keys = {"text:" + source_fingerprint(document)}
        if document["pmid"]:
            keys.add("pmid:" + document["pmid"])
        linked = [group for group in groups if keys & group[0]]
        rows = [document]
        for group in linked:
            keys |= group[0]
            rows += group[1]
            groups.remove(group)
        groups.append((keys, rows))
    require(len(groups) >= 2, "at least two independent papers are needed for a held-out split")
    groups.sort(key=lambda group: hashlib.sha256(f"{seed}:{min(group[0])}".encode()).hexdigest())
    count = min(len(groups) - 1, max(1, math.ceil(len(groups) * fraction)))
    dev = [row for _, rows in groups[:count] for row in rows]
    train = [row for _, rows in groups[count:] for row in rows]
    return train, dev


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="check source passages and supplied patient mentions")
    audit.add_argument("--input", required=True)
    build = commands.add_parser(
        "build", help="build HPO lexical model; optionally learn only from official training data"
    )
    build.add_argument("--hpo", required=True)
    build.add_argument("--train")
    build.add_argument("--output", required=True)
    infer = commands.add_parser("predict", help="generate both subtasks without reading input labels")
    infer.add_argument("--model", required=True)
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--evidence")
    infer.add_argument("--allow-training-overlap", action="store_true", help="diagnostics only; invalid for validation")
    check = commands.add_parser("validate", help="validate complete submission against the corresponding input file")
    check.add_argument("--input", required=True)
    check.add_argument("--predictions", required=True)
    check.add_argument("--hpo", required=True)
    score = commands.add_parser("evaluate", help="local reference metrics; requires gold labels")
    score.add_argument("--gold", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output")
    score.add_argument("--association-compounds", choices=["split_hpo", "literal"], default="split_hpo")
    holdout = commands.add_parser("holdout", help="fit/evaluate on a deterministic paper-grouped training split")
    holdout.add_argument("--train", required=True)
    holdout.add_argument("--hpo", required=True)
    holdout.add_argument("--output-dir", required=True)
    holdout.add_argument("--dev-fraction", type=float, default=0.2)
    holdout.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "audit":
        documents = read_jsonl(args.input)
        index_documents(documents, source=True)
        result = {
            "documents": len(documents),
            "patients": sum(len(d["patient"]) for d in documents),
            "passages": sum(len(d["full_text"]) for d in documents),
            "known_non_body_passages": sum(not scored_passage(p) for d in documents for p in d["full_text"]),
            "input_sha256": digest(args.input),
            "labeled_documents": sum("entities" in d and "association" in d for d in documents),
        }
    elif args.command == "build":
        ontology = load_ontology(args.hpo)
        model = build_model(ontology, read_jsonl(args.train) if args.train else [])
        write_json(args.output, model)
        result = {
            "aliases": len(model["aliases"]),
            "hpo_terms": len(ontology["terms"]),
            "training_papers": len(model["training_documents"]),
            "model_sha256": digest(args.output),
        }
    elif args.command == "predict":
        model = json.loads(Path(args.model).read_text(encoding="utf-8"))
        predictions, evidence = predict(model, read_jsonl(args.input), args.allow_training_overlap)
        for path in [args.output, args.evidence]:
            if path:
                require(not Path(path).exists(), f"refusing to overwrite: {path}")
        require(
            not args.evidence or Path(args.evidence).resolve() != Path(args.output).resolve(),
            "predictions and evidence need different output paths",
        )
        write_json(args.output, predictions, jsonl=True)
        if args.evidence:
            write_json(args.evidence, evidence, jsonl=True)
        result = {
            "documents": len(predictions),
            "prediction_sha256": digest(args.output),
            "input_sha256": digest(args.input),
            "model_sha256": digest(args.model),
            "training_overlap_allowed": args.allow_training_overlap,
            "attribution_decisions": dict(Counter(d["method"] for row in evidence for d in row["decisions"])),
        }
    elif args.command == "validate":
        result = validate_predictions(read_jsonl(args.input), read_jsonl(args.predictions), load_ontology(args.hpo))
        result["prediction_sha256"] = digest(args.predictions)
    elif args.command == "evaluate":
        result = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions), args.association_compounds)
        result.update({"gold_sha256": digest(args.gold), "prediction_sha256": digest(args.predictions)})
        if args.output:
            write_json(args.output, result)
    else:
        destination = Path(args.output_dir)
        require(not destination.exists(), f"refusing to reuse run directory: {destination}")
        documents = read_jsonl(args.train)
        ontology = load_ontology(args.hpo)
        audit = validate_gold(documents, ontology)
        training, dev = split_training(documents, args.dev_fraction, args.seed)
        model = build_model(ontology, training)
        predictions, evidence = predict(model, dev)
        validate_predictions(dev, predictions, ontology)
        result = evaluate(dev, predictions)
        result.update(
            {
                "seed": args.seed,
                "training_input_sha256": digest(args.train),
                "hpo_sha256": ontology["sha256"],
                "train_pmc_ids": [d["pmc_id"] for d in training],
                "dev_pmc_ids": [d["pmc_id"] for d in dev],
            }
        )
        for name, value, jsonl in [
            ("train.jsonl", training, True),
            ("dev.gold.jsonl", dev, True),
            ("dev.predictions.jsonl", predictions, True),
            ("dev.evidence.jsonl", evidence, True),
            ("model.json", model, False),
            ("metrics.json", result, False),
            ("gold.audit.json", audit, False),
            ("metrics.literal_associations.json", evaluate(dev, predictions, "literal"), False),
        ]:
            write_json(destination / name, value, jsonl=jsonl)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
