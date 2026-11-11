"""Reload a completed run and predict a new PatientPheX JSONL file."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import enhanced as base
import lexical_recall as recall
import neural_frontier as nf
import patientphex as pp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--hpo", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pp.require(not args.output.exists(), f"output exists: {args.output}")
    pp.require(not args.evidence.exists(), f"evidence exists: {args.evidence}")
    config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    lexical = json.loads((args.run_dir / "lexical.json").read_text(encoding="utf-8"))
    documents = pp.read_jsonl(args.input)
    ontology = pp.load_ontology(args.hpo)

    lexical_predictions, _ = recall.predict_lexical(base, lexical, documents)
    spans, _ = nf.predict_ner(
        documents,
        args.run_dir / "models" / "ner",
        args.device,
    )
    retriever = nf.OntologyRetriever(
        args.model_root / "sapbert",
        ontology,
        args.run_dir / "artifacts" / "ontology_index",
        args.device,
    )
    try:
        linked = nf.link_spans(spans, lexical, retriever)
    finally:
        retriever.close()
    entities = nf.merge_entities(
        base,
        documents,
        lexical_predictions,
        linked,
        config["entity_settings"],
    )

    with (args.run_dir / "relation.pkl").open("rb") as handle:
        classical_model = pickle.load(handle)
    classical = base.score_relations(classical_model, documents, entities)
    neural = nf.predict_relation(
        base,
        documents,
        entities,
        ontology,
        args.run_dir / "models" / "relation",
        args.device,
    )
    scores = nf.blend_scores(classical, neural, config["relation_weight"])
    predictions = base.apply_relations(
        documents,
        entities,
        scores,
        config["relation_threshold"],
    )
    validation = pp.validate_predictions(documents, predictions, ontology)
    pp.write_json(args.output, predictions, jsonl=True)
    pp.write_json(args.evidence, scores, jsonl=True)
    print(json.dumps({"validation": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
