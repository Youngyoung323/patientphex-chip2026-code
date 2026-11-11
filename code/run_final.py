"""Train the fixed PatientPheX pipeline and generate A/B predictions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import enhanced as base
import lexical_recall as recall
import neural_frontier as nf
import patientphex as pp


LEXICAL_SETTINGS = {"min_precision": 0.50, "morphology": True, "abbreviations": True}
RECALL_SETTINGS = {"synonyms": True, "surface": True, "bag": True, "longest": True}
ENTITY_SETTINGS = {"mode": "replace", "similarity": 0.90, "margin": 0.03}
RELATION_WEIGHT = 0.25
RELATION_THRESHOLD = 0.35


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            value.update(block)
    return value.hexdigest()


def write_pickle(path: Path, value: object) -> None:
    with path.open("xb") as handle:
        pickle.dump(value, handle, protocol=5)


def fit_entities(
    documents: list[dict],
    targets: dict[str, list[dict]],
    ontology: dict,
    hpo_path: Path,
    model_root: Path,
    output_dir: Path,
    device: str,
) -> tuple[dict, dict[str, list[dict]]]:
    lexical = recall.fit_lexical(
        base,
        ontology,
        documents,
        LEXICAL_SETTINGS,
        RECALL_SETTINGS,
        hpo_path,
    )
    pp.write_json(output_dir / "lexical.json", lexical)

    ner_checkpoint = output_dir / "models" / "ner"
    nf.train_ner(documents, model_root / "biolinkbert_base", ner_checkpoint, device)

    all_documents = {"train": documents, **targets}
    spans = {}
    for name, rows in all_documents.items():
        spans[name], audit = nf.predict_ner(
            rows,
            ner_checkpoint,
            device,
            allow_training_overlap=name == "train",
        )
        pp.write_json(output_dir / f"{name}.span_audit.json", audit)

    retriever = nf.OntologyRetriever(
        model_root / "sapbert",
        ontology,
        output_dir / "artifacts" / "ontology_index",
        device,
    )
    entities = {}
    try:
        for name, rows in all_documents.items():
            linked = nf.link_spans(spans[name], lexical, retriever)
            previous, _ = recall.predict_lexical(
                base,
                lexical,
                rows,
                allow_training_overlap=name == "train",
            )
            entities[name] = nf.merge_entities(
                base,
                rows,
                previous,
                linked,
                ENTITY_SETTINGS,
            )
            pp.validate_predictions(rows, entities[name], ontology)
            pp.write_json(output_dir / f"{name}.entities.jsonl", entities[name], jsonl=True)
    finally:
        retriever.close()
    return lexical, entities


def fit_and_predict(
    train_path: Path,
    hpo_path: Path,
    target_paths: dict[str, Path],
    model_root: Path,
    output_dir: Path,
    device: str,
) -> None:
    pp.require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    ontology = pp.load_ontology(hpo_path)
    train = pp.read_jsonl(train_path)
    targets = {name: pp.read_jsonl(path) for name, path in target_paths.items()}
    pp.validate_gold(train, ontology)

    lexical, entities = fit_entities(
        train,
        targets,
        ontology,
        hpo_path,
        model_root,
        output_dir,
        device,
    )

    relation_checkpoint = output_dir / "models" / "relation"
    nf.train_relation(
        base,
        train,
        entities["train"],
        ontology,
        model_root / "biolinkbert_base",
        relation_checkpoint,
        device,
    )
    classical_model = base.fit_relations(train, entities["train"], ontology)
    write_pickle(output_dir / "relation.pkl", classical_model)

    for name, rows in targets.items():
        classical = base.score_relations(
            classical_model,
            rows,
            entities[name],
        )
        neural = nf.predict_relation(
            base,
            rows,
            entities[name],
            ontology,
            relation_checkpoint,
            device,
        )
        scores = nf.blend_scores(classical, neural, RELATION_WEIGHT)
        predictions = base.apply_relations(rows, entities[name], scores, RELATION_THRESHOLD)
        validation = pp.validate_predictions(rows, predictions, ontology)
        pp.write_json(output_dir / f"{name}.evidence.jsonl", scores, jsonl=True)
        pp.write_json(output_dir / f"{name}.predictions.jsonl", predictions, jsonl=True)
        pp.write_json(output_dir / f"{name}.validation.json", validation)

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "train": {"path": str(train_path.resolve()), "sha256": digest(train_path)},
            "hpo": {"path": str(hpo_path.resolve()), "sha256": digest(hpo_path)},
            **{
                name: {"path": str(path.resolve()), "sha256": digest(path)}
                for name, path in target_paths.items()
            },
        },
        "models": {
            "root": str(model_root.resolve()),
            "biolinkbert_base": str((model_root / "biolinkbert_base").resolve()),
            "sapbert": str((model_root / "sapbert").resolve()),
        },
        "entity_settings": ENTITY_SETTINGS,
        "relation_weight": RELATION_WEIGHT,
        "relation_threshold": RELATION_THRESHOLD,
        "ner_settings": nf.NER_SETTINGS,
        "relation_settings": nf.RELATION_SETTINGS,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "torch", "transformers", "tokenizers", "safetensors")
        },
        "source_files": sorted(path.name for path in Path(__file__).parent.glob("*.py")),
        "artifacts": {
            name: digest(output_dir / f"{name}.predictions.jsonl")
            for name in target_paths
        },
        "lexical_model_sha256": digest(output_dir / "lexical.json"),
        "relation_model_sha256": digest(output_dir / "relation.pkl"),
    }
    pp.write_json(output_dir / "run_config.json", config)
    pp.write_json(output_dir / "run_summary.json", {"state": "complete", "targets": sorted(target_paths)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--hpo", required=True, type=Path)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    fit_and_predict(
        args.train,
        args.hpo,
        {"a": args.a, "b": args.b},
        args.model_root,
        args.output_dir,
        args.device,
    )


if __name__ == "__main__":
    main()
