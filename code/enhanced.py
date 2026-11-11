"""Training-only alias filtering, document abbreviations and patient relation models."""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict

import patientphex as pp

SECTIONS = ("TITLE", "ABSTRACT", "INTRO", "CASE", "RESULTS", "METHODS", "DISCUSS", "FIG", "TABLE", "CONCL")
CLINICAL = re.compile(
    r"\b(?:he|she|his|her|proband|presented|examination|admitted|diagnosed|our patient|our case|we report)\b", re.I
)
FAMILY = re.compile(r"\b(?:mother|father|brother|sister|parents|siblings|family)\b", re.I)
ABBREVIATION = re.compile(r"\(([A-Za-z][A-Za-z0-9-]{1,9})\)")
PATIENT_LABEL = re.compile(
    r"\b(patient|case|individual|subject|proband)\s+(?:no\.?\s*)?"
    r"([IVX\d]+(?:[.-][IVX\d]+)*|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.I,
)
NUMBER_WORDS = {word: str(n) for n, word in enumerate("one two three four five six seven eight nine ten".split(), 1)}


def singular(word):
    """Conservative plural normalization, without touching original character offsets."""
    irregular = {"feet": "foot", "teeth": "tooth"}
    if word in irregular:
        return irregular[word]
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def canonical_gold_units(document, ontology):
    units = set()
    for entity in document["entities"]:
        parts = pp.gold_identifier_parts(entity["identifier"])
        ids = tuple("-1" if p == "-1" else ontology.get("canonical_ids", {}).get(p, p) for p in parts)
        if all(p == "-1" or p in ontology["terms"] for p in ids):
            units.add((entity["offset"], entity["length"], ";".join(sorted(set(ids)))))
    return units


def fit_lexical(ontology, training, config):
    fitted = pp.build_model(ontology, training)
    fitted["enhanced_config"] = dict(config)
    if config.get("legacy", False):
        return fitted
    ontology_aliases = pp.build_model(ontology, [])["aliases"]
    matcher = pp.Matcher(fitted)
    correct, total = Counter(), Counter()
    for document in training:
        gold = canonical_gold_units(document, ontology)
        for passage in document["full_text"]:
            if not pp.scored_passage(passage):
                continue
            for entity in matcher.entities(passage):
                alias = " ".join(pp.tokens(entity["text"]))
                total[alias] += 1
                unit = (entity["offset"], entity["length"], entity["identifier"])
                correct[alias] += unit in gold
    rejected, stats = [], {}
    for alias, identifier in list(fitted["aliases"].items()):
        # The original ontology already excludes global one-token aliases shorter than four characters.
        short = len(alias.split()) == 1 and len(alias) < 4
        learned_change = ontology_aliases.get(alias) != identifier
        precision = correct[alias] / total[alias] if total[alias] else 1.0
        if learned_change or short:
            stats[alias] = {"matched": total[alias], "correct": correct[alias], "precision": precision}
        if short or (learned_change and precision < config.get("min_precision", 0.0)):
            if alias in ontology_aliases and not short:
                fitted["aliases"][alias] = ontology_aliases[alias]
            else:
                del fitted["aliases"][alias]
            rejected.append(alias)
    fitted["method"] = "filtered_lexical_v3"
    fitted["alias_filter"] = {"rejected_or_reverted": rejected, "counts": stats}
    return fitted


def negated(text, start, end):
    boundaries = list(pp.BOUNDARY.finditer(text[:start]))
    preceding = text[boundaries[-1].end() if boundaries else 0 : start][-160:]
    preceding = re.sub(r"\bnot only\b", "", preceding, flags=re.I)
    return bool(pp.NEGATION.search(preceding) or pp.NEGATED_AFTER.search(text[end : end + 80]))


def make_entity(passage, start, end, identifier):
    return {
        "identifier": identifier,
        "type": "Phenotype",
        "offset": passage["offset"] + start,
        "length": end - start,
        "text": passage["text"][start:end],
        "note": "NO" if negated(passage["text"], start, end) else None,
    }


def acronym_aligns(short, phrase):
    """Require letters in order, with the first acronym letter at a word boundary."""
    letters = "".join(c.casefold() for c in short if c.isalpha())
    text = phrase.casefold()
    position = len(text) - 1
    for index in range(len(letters) - 1, -1, -1):
        while position >= 0:
            if text[position] == letters[index] and (index or position == 0 or not text[position - 1].isalnum()):
                break
            position -= 1
        if position < 0:
            return False
        position -= 1
    return bool(letters)


class EnhancedMatcher:
    def __init__(self, model):
        self.aliases = model["aliases"]
        self.exact = pp.Matcher(model).trie
        self.stemmed = {}
        choices = defaultdict(set)
        if model["enhanced_config"].get("morphology"):
            for alias, identifier in self.aliases.items():
                choices[tuple(singular(word) for word in alias.split())].add(identifier)
        self.stem_aliases = {key: next(iter(values)) for key, values in choices.items() if len(values) == 1}
        for words, identifier in self.stem_aliases.items():
            node = self.stemmed
            for word in words:
                node = node.setdefault(word, {})
            node[None] = identifier

    def document_abbreviations(self, document):
        choices, definitions = defaultdict(set), defaultdict(list)
        for passage in document["full_text"]:
            if not pp.scored_passage(passage):
                continue
            for match in ABBREVIATION.finditer(passage["text"]):
                short = match[1]
                if sum(c.isupper() for c in short) < 2:
                    continue
                before = passage["text"][: match.start()].rstrip()
                words = list(pp.TOKEN.finditer(before))
                # Only a known full phrase immediately preceding "(SHORT)" can define an alias.
                for size in range(min(len(words), 18), 0, -1):
                    start = words[-size].start()
                    phrase = before[start:]
                    if re.search(r"[;.!?\n]", phrase) or not acronym_aligns(short, phrase):
                        continue
                    key = pp.tokens(phrase)
                    identifier = self.aliases.get(" ".join(key)) or self.stem_aliases.get(
                        tuple(singular(w) for w in key)
                    )
                    if identifier:
                        choices[short].add(identifier)
                        definitions[short].append(
                            {
                                "short": short,
                                "identifier": identifier,
                                "full_text": phrase,
                                "offset": passage["offset"] + start,
                            }
                        )
                        break
        aliases = {short: next(iter(ids)) for short, ids in choices.items() if len(ids) == 1}
        evidence = [row for short, rows in definitions.items() if short in aliases for row in rows]
        return aliases, evidence

    def entities(self, passage, abbreviations):
        text = passage["text"]
        words = list(pp.TOKEN.finditer(text))
        candidates = []
        for index in range(len(words)):
            best = None
            for trie, transform, priority in [
                (self.exact, str.casefold, 2),
                (self.stemmed, lambda x: singular(x.casefold()), 1),
            ]:
                node, position = trie, index
                while position < len(words) and transform(words[position].group()) in node:
                    if position > index and re.search(
                        r"[;.!?\n]", text[words[position - 1].end() : words[position].start()]
                    ):
                        break
                    node = node[transform(words[position].group())]
                    position += 1
                    if None in node and (best is None or (position, priority) > best[:2]):
                        best = (position, priority, node[None])
            if best:
                stop, priority, identifier = best
                candidates.append((words[index].start(), words[stop - 1].end(), priority, identifier))
        for short, identifier in abbreviations.items():
            # Case-sensitive: a document definition for HE must never match the pronoun "He".
            for match in re.finditer(r"(?<!\w)" + re.escape(short) + r"(?!\w)", text):
                candidates.append((match.start(), match.end(), 3, identifier))
        result, previous_end = [], -1
        for start, end, _, identifier in sorted(candidates, key=lambda row: (row[0], -row[1], -row[2])):
            if start >= previous_end:
                result.append(make_entity(passage, start, end, identifier))
                previous_end = end
        return result


def assert_disjoint(model, documents, allow_training_overlap=False):
    pp.index_documents(documents, source=True)
    trained = model["training_documents"]
    ids = {row["pmc_id"] for row in trained}
    pmids = {row["pmid"] for row in trained if row["pmid"]}
    hashes = {row["text_sha256"] for row in trained}
    overlapping = [
        d["pmc_id"] for d in documents if d["pmc_id"] in ids or d["pmid"] in pmids or pp.source_fingerprint(d) in hashes
    ]
    pp.require(allow_training_overlap or not overlapping, f"training/inference overlap: {overlapping}")


def predict_lexical(model, documents, allow_training_overlap=False):
    if model["enhanced_config"].get("legacy", False):
        return pp.predict(model, documents, allow_training_overlap)
    assert_disjoint(model, documents, allow_training_overlap)
    matcher = EnhancedMatcher(model)
    predictions, evidence = [], []
    for document in documents:
        if model["enhanced_config"].get("abbreviations"):
            abbreviations, definitions = matcher.document_abbreviations(document)
        else:
            abbreviations, definitions = {}, []
        entities, decisions = [], []
        associations = {patient["patient_id"]: set() for patient in document["patient"]}
        for passage in document["full_text"]:
            if not pp.scored_passage(passage):
                continue
            anchors = pp.patient_anchors(document, passage)
            for entity in matcher.entities(passage, abbreviations):
                entities.append(entity)
                patients, method = pp.assign_patient(entity, passage, anchors)
                for patient_id in patients:
                    associations[patient_id].update(
                        entity["text"] if value == "-1" else value
                        for value in pp.identifier_parts(entity["identifier"])
                    )
                decisions.append(
                    {
                        "offset": entity["offset"],
                        "length": entity["length"],
                        "identifier": entity["identifier"],
                        "patient_ids": patients,
                        "method": method,
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


def extended_anchors(document, passage):
    anchors = set(pp.patient_anchors(document, passage))
    aliases = defaultdict(set)
    for patient in document["patient"]:
        patient_id = patient["patient_id"]
        suffix = patient_id[1:] if patient_id.startswith("O") else ""
        if suffix and re.fullmatch(r"[IVX\d]+(?:[.-][IVX\d]+)*", suffix, re.I):
            if "." in suffix or "-" in suffix:
                aliases[suffix].add(patient_id)
            for noun in ("patient", "case", "individual", "subject"):
                aliases[f"{noun} {suffix}"].add(patient_id)
        for mention in patient["mention"]:
            for match in PATIENT_LABEL.finditer(mention["text"]):
                noun, value = match[1].lower(), match[2]
                aliases[f"{noun} {value}"].add(patient_id)
                if value.lower() in NUMBER_WORDS:
                    aliases[f"{noun} {NUMBER_WORDS[value.lower()]}"].add(patient_id)
    for alias, owners in aliases.items():
        if len(owners) == 1:
            for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", passage["text"], re.I):
                anchors.add((match.start(), match.end(), next(iter(owners))))
    return sorted(anchors)


def span_distance(start, end, anchor):
    return max(0, anchor[0] - end, start - anchor[1])


def relation_rows(documents, predictions):
    """Feature extraction uses source text/patient mentions and predicted entities only."""
    outputs = pp.index_documents(predictions)
    rows = []
    for document in documents:
        passages = sorted(document["full_text"], key=lambda p: p["offset"])
        anchors = {p["offset"]: extended_anchors(document, p) for p in passages}
        global_anchors = sorted(
            (start + p["offset"], end + p["offset"], owner)
            for p in passages
            for start, end, owner in anchors[p["offset"]]
        )
        headings, owner = {}, set()
        for passage in passages:
            if passage.get("type", "").startswith("title"):
                owner = {a[2] for a in anchors[passage["offset"]]}
            headings[passage["offset"]] = set(owner)
        mentions = defaultdict(list)
        for entity in outputs[document["pmc_id"]]["entities"]:
            if pp.negative(entity) or not pp.scored_entity(entity, document):
                continue
            passage = next(p for p in passages if p["offset"] <= entity["offset"] < p["offset"] + len(p["text"]))
            start = entity["offset"] - passage["offset"]
            end = start + entity["length"]
            boundaries = list(pp.BOUNDARY.finditer(passage["text"][:start]))
            left = boundaries[-1].end() if boundaries else 0
            following = pp.BOUNDARY.search(passage["text"], end)
            right = following.start() if following else len(passage["text"])
            local = anchors[passage["offset"]]
            explicit, _ = pp.assign_patient(entity, passage, local)
            before = [a for a in global_anchors if a[1] <= entity["offset"]]
            last_owners = {a[2] for a in before if a[:2] == max(before, key=lambda x: x[1])[:2]} if before else set()
            context = passage["text"][left:right]
            for value in pp.identifier_parts(entity["identifier"]):
                concept = entity["text"] if value == "-1" else value
                mentions[concept].append(
                    (entity, passage, start, end, left, right, local, explicit, last_owners, context)
                )
        document_length = max((p["offset"] + len(p["text"]) for p in passages), default=1)
        for concept, items in sorted(mentions.items()):
            for patient in document["patient"]:
                patient_id = patient["patient_id"]
                own = [a for a in global_anchors if a[2] == patient_id]
                others = [a for a in global_anchors if a[2] != patient_id]
                values = defaultdict(list)
                evidence = []
                for entity, passage, start, end, left, right, local, explicit, last_owners, context in items:
                    absolute_start, absolute_end = entity["offset"], entity["offset"] + entity["length"]
                    own_distance = min((span_distance(absolute_start, absolute_end, a) for a in own), default=100000)
                    other_distance = min(
                        (span_distance(absolute_start, absolute_end, a) for a in others), default=100000
                    )
                    same = {a[2] for a in local if a[0] < right and a[1] > left}
                    nearby = {a[2] for a in local}
                    per_mention = {
                        "distance_log": math.log1p(own_distance),
                        "near_own_200": math.exp(-own_distance / 200),
                        "near_own_1000": math.exp(-own_distance / 1000),
                        "nearest_target": float(own_distance < other_distance),
                        "distance_margin": math.exp(-own_distance / 500) - math.exp(-other_distance / 500),
                        "same_clause": float(patient_id in same),
                        "same_clause_exclusive": float(same == {patient_id}),
                        "clause_other": float(bool(same - {patient_id})),
                        "same_passage": float(patient_id in nearby),
                        "same_passage_exclusive": float(nearby == {patient_id}),
                        "explicit_rule": float(patient_id in explicit),
                        "most_recent_patient": float(last_owners == {patient_id}),
                        "heading_owner": float(headings[passage["offset"]] == {patient_id}),
                        "general": float(bool(pp.GENERAL.search(context))),
                        "clinical": float(bool(CLINICAL.search(context))),
                        "family": float(bool(FAMILY.search(context))),
                        "position": entity["offset"] / max(document_length, 1),
                    }
                    for section in SECTIONS:
                        per_mention["section_" + section] = float(passage.get("section_type", "").upper() == section)
                    for name, value in per_mention.items():
                        values[name].append(value)
                    evidence.append(
                        {
                            "offset": entity["offset"],
                            "length": entity["length"],
                            "section": passage.get("section_type"),
                            "distance_to_patient": own_distance,
                            "explicit_rule": patient_id in explicit,
                            "same_clause": patient_id in same,
                        }
                    )
                features = {
                    "patients_log": math.log1p(len(document["patient"])),
                    "single_patient": float(len(document["patient"]) == 1),
                    "mentions_log": math.log1p(len(items)),
                    "patient_anchors_log": math.log1p(len(own)),
                    "concepts_log": math.log1p(len(mentions)),
                    "unmapped": float(not concept.startswith("HP:")),
                }
                for name, numbers in values.items():
                    features[name + "_max"] = max(numbers)
                    features[name + "_min"] = min(numbers)
                    features[name + "_mean"] = sum(numbers) / len(numbers)
                rows.append(
                    {
                        "pmc_id": document["pmc_id"],
                        "patient_id": patient_id,
                        "concept": concept,
                        "features": features,
                        "evidence": evidence,
                    }
                )
    return rows


def fit_relations(documents, predictions, ontology):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    from threadpoolctl import threadpool_limits

    rows = relation_rows(documents, predictions)
    pp.require(bool(rows), "no patient-concept training candidates")
    expected = {}
    for document in documents:
        for association in document["association"]:
            expected[document["pmc_id"], association["patient_id"]] = {
                ontology.get("canonical_ids", {}).get(value, value)
                for value in pp.gold_association_set(association["phenotype"])
            }
    labels = [int(row["concept"] in expected[row["pmc_id"], row["patient_id"]]) for row in rows]
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform([row["features"] for row in rows])
    if len(set(labels)) == 1:
        classifier = DummyClassifier(strategy="prior")
    else:
        classifier = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
    with threadpool_limits(limits=2):
        classifier.fit(matrix, labels)
    return {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "training_rows": len(rows),
        "training_positive_rows": sum(labels),
        "training_documents": [
            {"pmc_id": d["pmc_id"], "pmid": d["pmid"], "text_sha256": pp.source_fingerprint(d)} for d in documents
        ],
        "feature_names": list(vectorizer.get_feature_names_out()),
        "method": "patient_concept_hist_gradient_boosting_v1",
    }


def score_relations(model, documents, predictions, allow_training_overlap=False):
    from threadpoolctl import threadpool_limits

    assert_disjoint(model, documents, allow_training_overlap)
    rows = relation_rows(documents, predictions)
    if not rows:
        return []
    matrix = model["vectorizer"].transform([row["features"] for row in rows])
    classes = list(model["classifier"].classes_)
    with threadpool_limits(limits=2):
        probabilities = model["classifier"].predict_proba(matrix)
    for index, row in enumerate(rows):
        row["probability"] = float(probabilities[index, classes.index(1)]) if 1 in classes else 0.0
    return rows


def apply_relations(documents, predictions, scored, threshold):
    pp.require(0 <= threshold <= 1, "relation threshold must be between zero and one")
    result = copy.deepcopy(predictions)
    outputs = pp.index_documents(result)
    associations = {d["pmc_id"]: {p["patient_id"]: set() for p in d["patient"]} for d in documents}
    seen = set()
    for row in scored:
        key = (row["pmc_id"], row["patient_id"], row["concept"])
        pp.require(key not in seen, "duplicate scored relation")
        seen.add(key)
        if row["probability"] >= threshold:
            associations[row["pmc_id"]][row["patient_id"]].add(row["concept"])
    for key, patients in associations.items():
        outputs[key]["association"] = [
            {"patient_id": patient, "phenotype": sorted(values)} for patient, values in patients.items()
        ]
    pp.validate_predictions(documents, result)
    return result
