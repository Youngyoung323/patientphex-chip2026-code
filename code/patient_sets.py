"""Deterministic patient-set decisions over frozen relation scores and source evidence."""

from __future__ import annotations

import copy
import math

import patientphex as pp


def policy_grid():
    policies = [{"kind": "global", "threshold": t} for t in (0.2, 0.3, 0.4, 0.5, 0.6)]
    for kind, name in (("evidence", "weak_threshold"), ("patient_gate", "gate")):
        policies.extend({"kind": kind, "threshold": 0.2, name: t} for t in (0.4, 0.6, 0.8))
    policies.extend({"kind": "relative", "threshold": 0.2, "fraction": t} for t in (0.5, 0.75))
    policies.append({"kind": "rules", "threshold": 0.2})
    return policies


def policy_name(policy):
    return "_".join(str(value) for value in policy.values())


def strong_evidence(row):
    features = row["features"]
    return any(
        features.get(name, 0) >= 1 for name in ("explicit_rule_max", "same_clause_exclusive_max", "heading_owner_max")
    )


def validate_policy(policy):
    kind = policy.get("kind")
    pp.require(kind in {"global", "evidence", "patient_gate", "relative", "rules"}, "unknown set policy")
    pp.require(0 <= policy["threshold"] <= 1, "invalid set threshold")
    if kind == "evidence":
        pp.require(policy["threshold"] <= policy["weak_threshold"] <= 1, "invalid weak threshold")
    if kind == "patient_gate":
        pp.require(policy["threshold"] <= policy["gate"] <= 1, "invalid patient gate")
    if kind == "relative":
        pp.require(0 <= policy["fraction"] <= 1, "invalid relative threshold")


def apply_policy(documents, predictions, scored, policy):
    """Do not read gold entities/associations or infer labels from patient identifiers."""
    validate_policy(policy)
    pp.validate_predictions(documents, predictions)
    result = copy.deepcopy(predictions)
    outputs = pp.index_documents(result)
    groups = {(d["pmc_id"], p["patient_id"]): [] for d in documents for p in d["patient"]}
    rules = {(doc["pmc_id"], a["patient_id"]): set(a["phenotype"]) for doc in predictions for a in doc["association"]}
    seen = set()
    for row in scored:
        key = row["pmc_id"], row["patient_id"]
        pp.require(key in groups, "scored relation names an unknown patient/document")
        unique = (*key, row["concept"])
        pp.require(unique not in seen, "duplicate scored relation")
        seen.add(unique)
        probability = row["probability"]
        pp.require(math.isfinite(probability) and 0 <= probability <= 1, "invalid relation probability")
        groups[key].append(row)
    ledger = []
    selected = {}
    threshold = policy["threshold"]
    for key, rows in groups.items():
        maximum = max((r["probability"] for r in rows), default=0)
        supported = any(r["probability"] >= threshold and strong_evidence(r) for r in rows)
        closed = policy["kind"] == "patient_gate" and not supported and maximum < policy["gate"]
        kept = set()
        for row in rows:
            cutoff = threshold
            if policy["kind"] == "evidence" and not strong_evidence(row):
                cutoff = policy["weak_threshold"]
            elif policy["kind"] == "relative":
                cutoff = max(threshold, maximum * policy["fraction"])
            accept = not closed and row["probability"] >= cutoff
            if policy["kind"] == "rules":
                accept = row["concept"] in rules[key]
            if accept:
                kept.add(row["concept"])
            ledger.append(
                {
                    **{k: v for k, v in row.items() if k != "features"},
                    "strong_evidence": strong_evidence(row),
                    "accepted": accept,
                    "threshold": cutoff,
                    "patient_gate_closed": closed,
                }
            )
        selected[key] = rules[key] if policy["kind"] == "rules" else kept
    for document in documents:
        outputs[document["pmc_id"]]["association"] = [
            {"patient_id": p["patient_id"], "phenotype": sorted(selected[document["pmc_id"], p["patient_id"]])}
            for p in document["patient"]
        ]
    pp.validate_predictions(documents, result)
    return result, ledger


def grouped_patient_metrics(diagnosis):
    result = {}
    for name, predicate in (
        ("empty", lambda n: n == 0),
        ("nonempty", lambda n: n > 0),
        ("one_to_three", lambda n: 1 <= n <= 3),
        ("four_plus", lambda n: n >= 4),
    ):
        rows = [r for r in diagnosis["patients"] if predicate(r["tp"] + r["fn"])]
        result[name] = {
            "patients": len(rows),
            "macro_f1": sum(r["f1"] for r in rows) / len(rows) if rows else None,
            **{key: sum(r[key] for r in rows) for key in ("tp", "fp", "fn")},
        }
    return result
