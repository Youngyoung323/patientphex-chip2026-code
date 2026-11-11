"""Conservative additional surface candidates from the fixed ontology and fitting papers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import patientphex as pp

FUNCTION_WORDS = frozenset("of the a an in is are was were".split())
SAFE_GAP = re.compile(r"(?:[^\S\r\n]|[\-–—])*\Z")


def configurations():
    return {
        "baseline": {},
        "synonyms": {"synonyms": True},
        "surface": {"surface": True},
        "bag": {"bag": True},
        "surface_bag": {"surface": True, "bag": True},
        "all": {"synonyms": True, "surface": True, "bag": True},
        "all_longest": {"synonyms": True, "surface": True, "bag": True, "longest": True},
        "bag_longest": {"bag": True, "longest": True},
    }


def spelling(word):
    for before, after in (
        ("haem", "hem"),
        ("anaem", "anem"),
        ("oedem", "edem"),
        ("tumour", "tumor"),
        ("behaviour", "behavior"),
        ("colour", "color"),
    ):
        word = word.replace(before, after)
    return word


def additional_synonyms(path, ontology, existing):
    choices = defaultdict(set)
    current = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("["):
                current = None
            elif line.startswith("id: "):
                current = line[4:].strip()
            elif current in ontology["terms"] and line.startswith("synonym: "):
                match = re.match(r'synonym: ("(?:[^"\\]|\\.)*") (?:BROAD|NARROW|RELATED)\b', line)
                if match:
                    phrase = match[1][1:-1].replace(r"\"", '"').replace(r"\\", "\\")
                    words = pp.tokens(phrase)
                    if words and not (len(words) == 1 and len(words[0]) < 4):
                        choices[" ".join(words)].add(current)
    return {key: next(iter(values)) for key, values in choices.items() if len(values) == 1 and key not in existing}


def unambiguous(choices):
    return {key: next(iter(ids)) for key, ids in choices.items() if len(ids) == 1}


def overlap(lhs, rhs):
    return lhs["offset"] < rhs["offset"] + rhs["length"] and rhs["offset"] < lhs["offset"] + lhs["length"]


class RecallMatcher:
    def __init__(self, base, model, ignore_rejections=False):
        self.base, self.config = base, model["recall_config"]
        self.original = base.EnhancedMatcher(model)
        extra_model = {**model, "aliases": model.get("recall_extra_aliases", {})}
        self.extra = base.EnhancedMatcher(extra_model)
        self.rejected = set() if ignore_rejections else set(model.get("recall_rejected_keys", []))
        surface, bags = defaultdict(set), defaultdict(set)
        aliases = {**model["aliases"], **model.get("recall_extra_aliases", {})}
        for alias, identifier in aliases.items():
            words = [base.singular(word) for word in alias.split()]
            if self.config.get("surface") and len(words) <= 12:
                key = "".join(spelling(word) for word in words)
                if len(key) >= 6:
                    surface[key].add(identifier)
            if self.config.get("bag"):
                words = [word for word in words if word not in FUNCTION_WORDS]
                if 2 <= len(words) <= 8:
                    bags[tuple(sorted(words))].add(identifier)
        self.surface, self.bags = unambiguous(surface), unambiguous(bags)

    def additional(self, passage):
        text = passage["text"]
        rows = []
        if self.config.get("synonyms"):
            for entity in self.extra.entities(passage, {}):
                tokens = list(pp.TOKEN.finditer(entity["text"]))
                if any(
                    not SAFE_GAP.fullmatch(entity["text"][lhs.end() : rhs.start()])
                    for lhs, rhs in zip(tokens, tokens[1:], strict=False)
                ):
                    continue
                key = "synonym|" + " ".join(pp.tokens(entity["text"])) + "|" + entity["identifier"]
                rows.append((entity, key))
        if self.surface or self.bags:
            words = list(pp.TOKEN.finditer(text))
            normalized = [self.base.singular(word.group().casefold()) for word in words]
            for start in range(len(words)):
                compact, content = "", []
                for end in range(start, min(start + 12, len(words))):
                    if end > start and not SAFE_GAP.fullmatch(text[words[end - 1].end() : words[end].start()]):
                        break
                    word = normalized[end]
                    compact += spelling(word)
                    if word not in FUNCTION_WORDS:
                        content.append(word)
                    matches = []
                    if compact in self.surface:
                        matches.append((self.surface[compact], "surface|" + compact))
                    if normalized[start] not in FUNCTION_WORDS and word not in FUNCTION_WORDS:
                        bag = tuple(sorted(content))
                        if bag in self.bags:
                            matches.append((self.bags[bag], "bag|" + " ".join(bag)))
                    for identifier, key in matches:
                        entity = self.base.make_entity(passage, words[start].start(), words[end].end(), identifier)
                        rows.append((entity, key + "|" + identifier))
        return [(e, key) for e, key in rows if key not in self.rejected]

    def entities(self, passage, abbreviations):
        original = self.original.entities(passage, abbreviations)
        extra = self.additional(passage)
        if not self.config.get("longest"):
            extra = [(e, key) for e, key in extra if not any(overlap(e, old) for old in original)]
        candidates = [(e, "original", 2) for e in original] + [(e, key, 1) for e, key in extra]
        selected, previous_end = [], -1
        for entity, key, _ in sorted(
            candidates, key=lambda row: (row[0]["offset"], -row[0]["length"], -row[2], row[1])
        ):
            if entity["offset"] >= previous_end:
                selected.append((entity, key))
                previous_end = entity["offset"] + entity["length"]
        return selected


def fit_lexical(base, ontology, training, original_config, recall_config, hpo_path):
    pp.require(pp.digest(hpo_path) == ontology["sha256"], "recall ontology hash mismatch")
    model = base.fit_lexical(ontology, training, original_config)
    model["recall_config"] = dict(recall_config)
    model["recall_extra_aliases"] = (
        additional_synonyms(hpo_path, ontology, model["aliases"]) if recall_config.get("synonyms") else {}
    )
    if not recall_config:
        return model
    matcher = RecallMatcher(base, model, ignore_rejections=True)
    correct, total = Counter(), Counter()
    for document in training:
        gold = base.canonical_gold_units(document, ontology)
        for passage in document["full_text"]:
            if pp.scored_passage(passage):
                seen = set()
                for entity, key in matcher.additional(passage):
                    unit = entity["offset"], entity["length"], entity["identifier"]
                    if (unit, key) not in seen:
                        total[key] += 1
                        correct[key] += unit in gold
                        seen.add((unit, key))
    model["recall_rejected_keys"] = sorted(key for key, count in total.items() if correct[key] / count < 0.5)
    model["recall_filter_counts"] = {key: {"matched": total[key], "correct": correct[key]} for key in sorted(total)}
    return model


def predict_lexical(base, model, documents, allow_training_overlap=False):
    if not model["recall_config"]:
        return base.predict_lexical(model, documents, allow_training_overlap)
    base.assert_disjoint(model, documents, allow_training_overlap)
    matcher = RecallMatcher(base, model)
    predictions, evidence = [], []
    for document in documents:
        abbreviations, definitions = matcher.original.document_abbreviations(document)
        associations = {p["patient_id"]: set() for p in document["patient"]}
        entities, decisions = [], []
        for passage in document["full_text"]:
            if not pp.scored_passage(passage):
                continue
            anchors = pp.patient_anchors(document, passage)
            for entity, key in matcher.entities(passage, abbreviations):
                entities.append(entity)
                patients, method = pp.assign_patient(entity, passage, anchors)
                for patient in patients:
                    associations[patient].update(
                        entity["text"] if p == "-1" else p for p in pp.identifier_parts(entity["identifier"])
                    )
                decisions.append(
                    {
                        "offset": entity["offset"],
                        "length": entity["length"],
                        "identifier": entity["identifier"],
                        "patient_ids": patients,
                        "method": method,
                        "recall_key": key,
                    }
                )
        predictions.append(
            {
                "pmc_id": document["pmc_id"],
                "pmid": document["pmid"],
                "entities": sorted(entities, key=lambda e: e["offset"]),
                "association": [
                    {"patient_id": key, "phenotype": sorted(values)} for key, values in associations.items()
                ],
            }
        )
        evidence.append({"pmc_id": document["pmc_id"], "decisions": decisions, "abbreviations": definitions})
    pp.validate_predictions(documents, predictions)
    return predictions, evidence
