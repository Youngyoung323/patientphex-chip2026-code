"""Document-isolated neural spans, ontology retrieval, and patient evidence classification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import patientphex as pp

NER_SETTINGS = {"epochs": 4, "lr": 3e-5, "length": 384, "stride": 64, "batch": 8, "seed": 42}
RELATION_SETTINGS = {"epochs": 3, "lr": 2e-5, "length": 512, "batch": 8, "seed": 43}


def emit(**kwargs):
    print(json.dumps(kwargs, ensure_ascii=False), flush=True)


def fingerprints(documents):
    return [{"pmc_id": d["pmc_id"], "pmid": d["pmid"], "text_sha256": pp.source_fingerprint(d)} for d in documents]


def check_disjoint(metadata, documents, allow_training_overlap=False):
    trained = metadata["training_documents"]
    ids = {r["pmc_id"] for r in trained}
    pmids = {r["pmid"] for r in trained if r["pmid"]}
    hashes = {r["text_sha256"] for r in trained}
    overlap = [
        d["pmc_id"] for d in documents if d["pmc_id"] in ids or d["pmid"] in pmids or pp.source_fingerprint(d) in hashes
    ]
    pp.require(allow_training_overlap or not overlap, f"neural training/inference overlap: {overlap}")


def setup_torch(seed, device):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if device.startswith("cuda"):
        pp.require(torch.cuda.is_available(), "requested CUDA is unavailable")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch


def collate(items, tokenizer, device, token_labels=False):
    import torch

    inputs = [
        {key: row[key] for key in ("input_ids", "attention_mask", "token_type_ids") if key in row} for row in items
    ]
    batch = tokenizer.pad(inputs, padding=True, return_tensors="pt")
    if "labels" in items[0]:
        if token_labels:
            width = batch["input_ids"].shape[1]
            labels = [r["labels"] + [-100] * (width - len(r["labels"])) for r in items]
        else:
            labels = [r["labels"] for r in items]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return {k: v.to(device) for k, v in batch.items()}


def bio_labels(offsets, spans):
    """Label only complete, token-aligned entities; ignore partially visible spans."""
    labels = [0 if end > start else -100 for start, end in offsets]
    statistics = Counter()
    for start, end in sorted(set(spans), key=lambda s: (s[0], -(s[1] - s[0]))):
        covered = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
        if not covered:
            continue
        if offsets[covered[0]][0] != start or offsets[covered[-1]][1] != end:
            for i in covered:
                if labels[i] == 0:
                    labels[i] = -100
            statistics["unaligned_or_truncated"] += 1
            continue
        if any(labels[i] != 0 for i in covered):
            statistics["overlapping_annotation"] += 1
            continue
        for n, i in enumerate(covered):
            labels[i] = 1 if n == 0 else 2
        statistics["complete_entity_instances"] += 1
    return labels, statistics


def ner_examples(documents, tokenizer, settings, training=False):
    examples, stats = [], Counter()
    for document in documents:
        for passage in document["full_text"]:
            if not pp.scored_passage(passage) or not passage["text"].strip():
                continue
            encoded = tokenizer(
                passage["text"],
                max_length=settings["length"],
                stride=settings["stride"],
                truncation=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            spans = []
            if training:
                for entity in document["entities"]:
                    start = entity["offset"] - passage["offset"]
                    end = start + entity["length"]
                    if 0 <= start < end <= len(passage["text"]):
                        spans.append((start, end))
            for index, offsets in enumerate(encoded["offset_mapping"]):
                row = {k: encoded[k][index] for k in ("input_ids", "attention_mask", "token_type_ids") if k in encoded}
                row.update({"pmc_id": document["pmc_id"], "passage": passage, "offsets": offsets})
                if training:
                    row["labels"], counts = bio_labels(offsets, spans)
                    stats.update(counts)
                    if all(v == -100 for v in row["labels"]):
                        continue
                examples.append(row)
    stats["examples"] = len(examples)
    stats["documents"] = len(documents)
    return examples, dict(stats)


def train_transformer(model, tokenizer, examples, settings, destination, device, metadata, token_labels=False):
    """Fixed-length training; no development labels, early stopping, or online tracking."""
    from transformers import get_linear_schedule_with_warmup

    torch = setup_torch(settings["seed"], device)
    destination = Path(destination)
    pp.require(not destination.exists(), f"refusing to overwrite checkpoint: {destination}")
    destination.mkdir(parents=True)
    pp.require(bool(examples), "no neural training examples")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.01)
    steps = math.ceil(len(examples) / settings["batch"]) * settings["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * 0.1)), steps)
    rng = random.Random(settings["seed"])
    history, step, started = [], 0, time.monotonic()
    with (destination / "training.jsonl").open("x") as log:
        for epoch in range(settings["epochs"]):
            order = list(range(len(examples)))
            rng.shuffle(order)
            losses = []
            for offset in range(0, len(order), settings["batch"]):
                rows = [examples[i] for i in order[offset : offset + settings["batch"]]]
                batch = collate(rows, tokenizer, device, token_labels)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    loss = model(**batch).loss
                pp.require(bool(torch.isfinite(loss)), "nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                losses.append(float(loss.detach().cpu()))
                if step % 25 == 0:
                    row = {
                        "step": step,
                        "epoch": epoch + 1,
                        "mean_loss": float(np.mean(losses)),
                        "seconds": time.monotonic() - started,
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    emit(stage="training", checkpoint=str(destination), **row)
            history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "steps": step})
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    complete = {
        **metadata,
        "settings": settings,
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer_steps": step,
        "history": history,
        "seconds": time.monotonic() - started,
        "device": device,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "model_sha256": pp.digest(destination / "model.safetensors"),
        "state": "complete",
    }
    pp.write_json(destination / "training_metadata.json", complete)
    del optimizer, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return complete


def train_ner(documents, pretrained, destination, device, settings=None):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    settings = settings or NER_SETTINGS
    setup_torch(settings["seed"], device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained, num_labels=3, local_files_only=True, attn_implementation="eager"
    )
    examples, stats = ner_examples(documents, tokenizer, settings, training=True)
    return train_transformer(
        model,
        tokenizer,
        examples,
        settings,
        destination,
        device,
        {"task": "phenotype_BIO", "training_documents": fingerprints(documents), "data_audit": stats},
        True,
    )


def decode_bio(offsets, probabilities, text, min_probability=0.5):
    """Reject orphan I labels and chunks that end inside an entity."""
    chosen = probabilities.argmax(axis=-1)
    result, current, confidence = [], [], []

    def finish(truncated=False):
        if current and not truncated:
            start, end = offsets[current[0]][0], offsets[current[-1]][1]
            score = math.exp(sum(math.log(max(p, 1e-12)) for p in confidence) / len(confidence))
            left_ok = start == 0 or not (text[start - 1].isalnum() and text[start].isalnum())
            right_ok = end == len(text) or not (text[end - 1].isalnum() and text[end].isalnum())
            if left_ok and right_ok and score >= min_probability and text[start:end].strip():
                result.append({"start": start, "end": end, "ner_probability": score})
        current.clear()
        confidence.clear()

    real = [i for i, (a, b) in enumerate(offsets) if b > a]
    for position, i in enumerate(real):
        label = int(chosen[i])
        if label == 1:
            finish()
            current.append(i)
            confidence.append(float(probabilities[i, 1]))
        elif label == 2 and current:
            current.append(i)
            confidence.append(float(probabilities[i, 2]))
        else:
            finish()
        if position == len(real) - 1:
            end = offsets[i][1]
            # A chunk boundary can cut an entity even when it falls on a word boundary.
            finish(truncated=bool(current) and end < len(text.rstrip()))
    return result


def predict_ner(documents, checkpoint, device, allow_training_overlap=False):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    metadata = json.loads((Path(checkpoint) / "training_metadata.json").read_text())
    check_disjoint(metadata, documents, allow_training_overlap)
    torch = setup_torch(42, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    examples, stats = ner_examples(documents, tokenizer, metadata["settings"])
    candidates = {}
    with torch.inference_mode():
        for offset in range(0, len(examples), 16):
            batch_rows = examples[offset : offset + 16]
            batch = collate(batch_rows, tokenizer, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                outputs = model(**batch).logits.float().softmax(-1).cpu().numpy()
            for row, probs in zip(batch_rows, outputs, strict=True):
                passage = row["passage"]
                for span in decode_bio(row["offsets"], probs[: len(row["offsets"])], passage["text"]):
                    absolute = passage["offset"] + span["start"]
                    key = row["pmc_id"], absolute, span["end"] - span["start"]
                    candidate = {
                        "pmc_id": row["pmc_id"],
                        "offset": absolute,
                        "length": key[2],
                        "text": passage["text"][span["start"] : span["end"]],
                        "ner_probability": span["ner_probability"],
                    }
                    if key not in candidates or candidate["ner_probability"] > candidates[key]["ner_probability"]:
                        candidates[key] = candidate
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return [candidates[k] for k in sorted(candidates)], stats


class OntologyRetriever:
    """SapBERT CLS pooling, normalized vectors, and distinct-concept retrieval."""

    def __init__(self, pretrained, ontology, cache, device):
        from transformers import AutoModel, AutoTokenizer

        self.torch = setup_torch(42, device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
        self.model = AutoModel.from_pretrained(pretrained, local_files_only=True).to(device).eval()
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        pairs = sorted(
            {(text, key) for key, term in ontology["terms"].items() for text in [term["name"], *term["synonyms"]]}
        )
        self.pairs = pairs
        expected = {
            "hpo_sha256": ontology["sha256"],
            "model_sha256": pp.digest(Path(pretrained) / "model.safetensors"),
            "pairs_sha256": hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).hexdigest(),
            "pooling": "normalized_cls",
            "max_length": 64,
        }
        if (self.cache / "ontology.npy").exists():
            pp.require(json.loads((self.cache / "index.json").read_text()) == expected, "ontology cache mismatch")
            self.matrix = np.load(self.cache / "ontology.npy", allow_pickle=False)
        else:
            self.matrix = self.encode([p[0] for p in pairs])
            np.save(self.cache / "ontology.npy", self.matrix, allow_pickle=False)
            pp.write_json(self.cache / "index.json", expected)
        self.phrases = {}

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        result = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), 128):
                inputs = self.tokenizer(
                    texts[start : start + 128], padding=True, truncation=True, max_length=64, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # FP32 retrieval supports stable thresholds and identical cache replay.
                vectors = self.model(**inputs).last_hidden_state[:, 0, :].float()
                result.append(self.torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy())
        return np.concatenate(result)

    def retrieve(self, texts):
        missing = sorted(set(texts) - self.phrases.keys())
        for offset in range(0, len(missing), 128):
            current = missing[offset : offset + 128]
            vectors = self.encode(current)
            similarities = vectors @ self.matrix.T
            for text, row in zip(current, similarities, strict=True):
                # Adaptive retrieval: synonyms of one concept must not crowd out the runner-up.
                width = min(32, len(row))
                while True:
                    indices = np.argpartition(-row, width - 1)[:width]
                    indices = sorted(indices, key=lambda i: (-float(row[i]), self.pairs[i]))
                    seen, hits = set(), []
                    for index in indices:
                        label, identifier = self.pairs[index]
                        if identifier not in seen:
                            hits.append({"identifier": identifier, "score": float(row[index]), "alias": label})
                            seen.add(identifier)
                            if len(hits) == 3:
                                break
                    if len(hits) >= 3 or width == len(row):
                        break
                    width = min(width * 2, len(row))
                self.phrases[text] = hits
        return {text: self.phrases[text] for text in texts}

    def close(self):
        del self.model
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


def link_spans(spans, lexical, retriever):
    matches = retriever.retrieve([row["text"] for row in spans])
    result = []
    for row in spans:
        exact = lexical["aliases"].get(" ".join(pp.tokens(row["text"])))
        hits = matches[row["text"]]
        result.append({**row, "exact_identifier": exact, "hits": hits})
    return result


def entity_grid():
    return [{"mode": "baseline", "similarity": 1.0, "margin": 1.0}] + [
        {"mode": mode, "similarity": similarity, "margin": margin}
        for mode in ("augment", "replace")
        for similarity in (0.8, 0.85, 0.9)
        for margin in (0.0, 0.03)
    ]


def merge_entities(base, documents, lexical_predictions, linked, settings):
    outputs = copy.deepcopy(lexical_predictions)
    indexed = pp.index_documents(outputs)
    sources = pp.index_documents(documents, source=True)
    if settings["mode"] == "baseline":
        return outputs
    for row in sorted(linked, key=lambda r: (r["pmc_id"], -r["length"], r["offset"])):
        hits = row["hits"]
        identifier = row["exact_identifier"]
        if not identifier:
            if not hits or hits[0]["score"] < settings["similarity"]:
                continue
            if len(hits) > 1 and hits[0]["score"] - hits[1]["score"] < settings["margin"]:
                continue
            identifier = hits[0]["identifier"]
        source = sources[row["pmc_id"]]
        pp.require(pp.span_text(source, row["offset"], row["length"]) == row["text"], "neural span text mismatch")
        entities = indexed[row["pmc_id"]]["entities"]
        start, end = row["offset"], row["offset"] + row["length"]
        overlapping = [e for e in entities if e["offset"] < end and start < e["offset"] + e["length"]]
        if any(e["offset"] == start and e["length"] == row["length"] for e in overlapping):
            continue
        if overlapping:
            contains = all(start <= e["offset"] and e["offset"] + e["length"] <= end for e in overlapping)
            if settings["mode"] != "replace" or not contains:
                continue
            entities[:] = [e for e in entities if e not in overlapping]
        passage = next(p for p in source["full_text"] if p["offset"] <= start < p["offset"] + len(p["text"]))
        entities.append(base.make_entity(passage, start - passage["offset"], end - passage["offset"], identifier))
    for output in outputs:
        output["entities"].sort(key=lambda e: (e["offset"], e["length"], e["identifier"]))
        output["association"] = [{"patient_id": r["patient_id"], "phenotype": []} for r in output["association"]]
    pp.validate_predictions(documents, outputs)
    return outputs


def mark_excerpt(passage, left, right, anchors, patient_id, phenotype):
    marks = []
    for start, end, owner in anchors:
        if left <= start < end <= right:
            marks.append((start, end, "TARGET_PATIENT" if owner == patient_id else "OTHER_PATIENT"))
    start = phenotype["offset"] - passage["offset"]
    end = start + phenotype["length"]
    if left <= start < end <= right:
        marks.append((start, end, "PHENOTYPE"))
    # Right-to-left insertions preserve original text offsets. Nested/colliding marks are skipped.
    result, previous = passage["text"][left:right], right
    for start, end, tag in sorted(set(marks), reverse=True):
        if end <= previous:
            result = (
                result[: start - left]
                + f" [{tag}] "
                + result[start - left : end - left]
                + f" [/{tag}] "
                + result[end - left :]
            )
            previous = start
    return result


def relation_texts(base, documents, predictions, ontology):
    """Candidate evidence construction never reads source entities or associations."""
    outputs = pp.index_documents(predictions)
    rows = []
    for document in documents:
        passages = document["full_text"]
        anchors = {p["offset"]: base.extended_anchors(document, p) for p in passages}
        global_anchors = [
            (a + p["offset"], b + p["offset"], owner) for p in passages for a, b, owner in anchors[p["offset"]]
        ]
        concepts = defaultdict(list)
        for entity in outputs[document["pmc_id"]]["entities"]:
            if pp.negative(entity) or not pp.scored_entity(entity, document):
                continue
            for identifier in pp.identifier_parts(entity["identifier"]):
                concepts[entity["text"] if identifier == "-1" else identifier].append(entity)
        for concept, mentions in sorted(concepts.items()):
            name = ontology["terms"].get(concept, {}).get("name", concept)
            for patient in document["patient"]:
                pid = patient["patient_id"]
                own = [a for a in global_anchors if a[2] == pid]
                ranked = sorted(
                    mentions,
                    key=lambda e: (
                        min(
                            (base.span_distance(e["offset"], e["offset"] + e["length"], a) for a in own), default=100000
                        ),
                        e["offset"],
                    ),
                )
                evidence, excerpts, seen = [], [], set()
                for entity in ranked:
                    passage = next(
                        p for p in passages if p["offset"] <= entity["offset"] < p["offset"] + len(p["text"])
                    )
                    start = entity["offset"] - passage["offset"]
                    left, right = max(0, start - 240), min(len(passage["text"]), start + entity["length"] + 240)
                    key = (passage["offset"], left, right)
                    if key in seen:
                        continue
                    seen.add(key)
                    marked = mark_excerpt(passage, left, right, anchors[passage["offset"]], pid, entity)
                    excerpts.append(f"Section {passage.get('section_type', '')}: {marked}")
                    evidence.append(
                        {
                            "offset": entity["offset"],
                            "length": entity["length"],
                            "context_offset": passage["offset"] + left,
                            "context_length": right - left,
                        }
                    )
                    if len(excerpts) == 3:
                        break
                aliases = [s[:80] for s in dict.fromkeys(m["text"] for m in patient.get("mention", []))][:4]
                query = (
                    f"Patient {pid}: {'; '.join(aliases)}. Phenotype: {name[:200]}. "
                    "Does this patient have this phenotype?"
                )
                if own and ranked:
                    nearest = min(
                        own,
                        key=lambda a: base.span_distance(
                            ranked[0]["offset"], ranked[0]["offset"] + ranked[0]["length"], a
                        ),
                    )
                    reference = next(p for p in passages if p["offset"] <= nearest[0] < p["offset"] + len(p["text"]))
                    start = nearest[0] - reference["offset"]
                    left, right = (
                        max(0, start - 80),
                        min(len(reference["text"]), nearest[1] - reference["offset"] + 120),
                    )
                    marker = mark_excerpt(reference, left, right, anchors[reference["offset"]], pid, ranked[0])
                    excerpts.insert(0, "Patient reference: " + marker)
                rows.append(
                    {
                        "pmc_id": document["pmc_id"],
                        "patient_id": pid,
                        "concept": concept,
                        "query": query,
                        "context": "\n".join(excerpts),
                        "evidence": evidence,
                    }
                )
    return rows


def relation_examples(rows, tokenizer, settings, gold=None):
    encodings = tokenizer(
        [r["query"] for r in rows],
        [r["context"] for r in rows],
        truncation="only_second",
        max_length=settings["length"],
    )
    result = []
    for i, row in enumerate(rows):
        example = {k: encodings[k][i] for k in ("input_ids", "attention_mask", "token_type_ids") if k in encodings}
        if gold is not None:
            example["labels"] = int(row["concept"] in gold[row["pmc_id"], row["patient_id"]])
        result.append(example)
    return result


def train_relation(base, documents, predictions, ontology, pretrained, destination, device, settings=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = settings or RELATION_SETTINGS
    setup_torch(settings["seed"], device)
    rows = relation_texts(base, documents, predictions, ontology)
    gold = {
        (d["pmc_id"], a["patient_id"]): {
            ontology.get("canonical_ids", {}).get(c, c) for c in pp.gold_association_set(a["phenotype"])
        }
        for d in documents
        for a in d["association"]
    }
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, num_labels=2, local_files_only=True, attn_implementation="eager"
    )
    examples = relation_examples(rows, tokenizer, settings, gold)
    return train_transformer(
        model,
        tokenizer,
        examples,
        settings,
        destination,
        device,
        {
            "task": "patient_concept_evidence",
            "training_documents": fingerprints(documents),
            "data_audit": {
                "candidates": len(rows),
                "positive": sum(r["labels"] for r in examples),
                "examples_at_token_limit": sum(len(r["input_ids"]) == settings["length"] for r in examples),
            },
        },
    )


def predict_relation(base, documents, predictions, ontology, checkpoint, device, allow_training_overlap=False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    metadata = json.loads((Path(checkpoint) / "training_metadata.json").read_text())
    check_disjoint(metadata, documents, allow_training_overlap)
    torch = setup_torch(43, device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device).eval()
    rows = relation_texts(base, documents, predictions, ontology)
    if rows:
        examples = relation_examples(rows, tokenizer, metadata["settings"])
        with torch.inference_mode():
            for offset in range(0, len(rows), 16):
                batch = collate(examples[offset : offset + 16], tokenizer, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    scores = model(**batch).logits.float().softmax(-1)[:, 1].cpu().tolist()
                for row, score in zip(rows[offset : offset + 16], scores, strict=True):
                    row["probability"] = score
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return [{k: v for k, v in r.items() if k not in {"query", "context"}} for r in rows]


def blend_scores(classical, neural, weight):
    pp.require(0 <= weight <= 1, "invalid blend weight")
    key = lambda r: (r["pmc_id"], r["patient_id"], r["concept"])
    neural_index = {key(r): r for r in neural}
    pp.require(len(neural_index) == len(neural), "duplicate neural relation")
    pp.require(set(neural_index) == {key(r) for r in classical}, "neural/classical candidate mismatch")
    result = []
    for row in classical:
        other = neural_index[key(row)]
        result.append(
            {
                **{k: v for k, v in row.items() if k != "features"},
                "classical_probability": row["probability"],
                "neural_probability": other["probability"],
                "probability": (1 - weight) * row["probability"] + weight * other["probability"],
                "neural_evidence": other["evidence"],
            }
        )
    return result
