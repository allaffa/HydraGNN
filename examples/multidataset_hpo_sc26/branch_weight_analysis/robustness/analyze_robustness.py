#!/usr/bin/env python3
"""Second-round composition-routing robustness diagnostics.

Runs only the trained BranchWeightMLP. The HydraGNN checkpoint is inspected
only for provenance and is never executed.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ase.data import atomic_numbers, chemical_symbols
from scipy.stats import kendalltau, spearmanr

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
ROBUSTNESS_DIR = Path(__file__).resolve().parent
SC26_DIR = ANALYSIS_DIR.parent
REPO_ROOT = SC26_DIR.parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))
import analyze as first_round  # noqa: E402


SEED = 260508
N_PERMUTATIONS = 499
N_BOOTSTRAPS = 500
BRANCH_NAMES = first_round.BRANCH_NAMES
N_BRANCHES = len(BRANCH_NAMES)
BRANCH_COLORS = first_round.BRANCH_COLORS
BRANCH_LABELS = first_round.BRANCH_LABELS
MLP_PATH = SC26_DIR / "mlp_branch_weights.pt"
BACKBONE_PATH = (
    SC26_DIR
    / "multidataset_hpo-BEST6-fp64"
    / "multidataset_hpo-BEST6-fp64_epoch_97.pk"
)


def _composition(counts: dict[str, int]) -> np.ndarray:
    vector = np.zeros(118, dtype=np.float64)
    for symbol, count in counts.items():
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown element symbol: {symbol}")
        if count < 0 or int(count) != count:
            raise ValueError(f"Counts must be nonnegative integers: {symbol}={count}")
        vector[atomic_numbers[symbol] - 1] = int(count)
    if vector.sum() <= 0:
        raise ValueError("Composition must contain at least one atom")
    return vector


def _formula(vector: np.ndarray) -> str:
    return first_round.Atoms(
        numbers=[
            atomic_numbers[chemical_symbols[index + 1]]
            for index, count in enumerate(vector)
            for _ in range(int(count))
        ],
        positions=np.zeros((int(vector.sum()), 3)),
    ).get_chemical_formula(mode="hill")


def _vector_json(vector: np.ndarray, fractional: bool = False) -> str:
    values = vector / vector.sum() if fractional else vector
    return json.dumps(
        {
            chemical_symbols[index + 1]: float(value)
            for index, value in enumerate(values)
            if value > 0
        },
        sort_keys=True,
    )


def _load_first_round() -> tuple[list[dict[str, Any]], np.ndarray]:
    suite = first_round._make_suite()
    manifest = pd.read_csv(ANALYSIS_DIR / "composition_manifest.csv")
    weights = pd.read_csv(ANALYSIS_DIR / "branch_weights.csv")
    if len(suite) != len(manifest) or len(suite) != len(weights):
        raise ValueError("First-round manifest row count does not match deterministic suite")
    if not manifest["sample_id"].equals(weights["sample_id"]):
        raise ValueError("First-round composition and weight records are not aligned")
    for index, sample in enumerate(suite):
        if sample["sample_id"] != manifest.iloc[index]["sample_id"]:
            raise ValueError("First-round synthetic suite IDs changed")
        if sample["formula"] != manifest.iloc[index]["chemical_formula"]:
            raise ValueError("First-round formula differs from saved manifest")
    branch_columns = [
        f"branch_{index:02d}_{name}" for index, name in enumerate(BRANCH_NAMES)
    ]
    matrix = weights[branch_columns].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    for sample in suite:
        records.append(
            {
                "sample_id": sample["sample_id"],
                "origin": "first_round_fixed_100_atom_synthetic",
                "family": sample["family"],
                "formula": sample["formula"],
                "vector": sample["vector"],
                "atom_count": int(sample["vector"].sum()),
            }
        )
    return records, matrix


def _size_series() -> list[dict[str, Any]]:
    specs: list[tuple[str, list[dict[str, int]]]] = [
        (
            "Nb-Ta binary alloy",
            [
                {"Nb": 5 * n, "Ta": 5 * n}
                for n in (1, 2, 5, 10, 20)
            ],
        ),
        (
            "TiO2",
            [
                {"Ti": ti, "O": oxygen}
                for ti, oxygen in ((2, 4), (10, 20), (20, 40), (40, 80))
            ],
        ),
        (
            "CH4",
            [
                {"C": carbon, "H": hydrogen}
                for carbon, hydrogen in ((1, 4), (4, 16), (10, 40), (20, 80))
            ],
        ),
        (
            "LiFePO4",
            [
                {"Li": n, "Fe": n, "P": n, "O": 4 * n}
                for n in (1, 2, 5, 10)
            ],
        ),
        (
            "Nb-Ta-V-Hf-Zr multicomponent alloy",
            [
                {element: 2 * n for element in ("Nb", "Ta", "V", "Hf", "Zr")}
                for n in (1, 2, 5, 10, 20)
            ],
        ),
    ]
    result: list[dict[str, Any]] = []
    for family, compositions in specs:
        for step, counts in enumerate(compositions):
            vector = _composition(counts)
            result.append(
                {
                    "sample_id": f"SIZE-{len(result) + 1:03d}",
                    "origin": "variable_size_synthetic",
                    "family": family,
                    "step": step,
                    "counts": counts,
                    "vector": vector,
                    "formula": _formula(vector),
                    "atom_count": int(vector.sum()),
                }
            )
    return result


def _path_compositions() -> list[dict[str, Any]]:
    paths: list[tuple[str, list[dict[str, int]], str]] = []
    paths.append(
        (
            "Nb-Ta alloy (fixed 100 atoms)",
            [{"Nb": 100 - 10 * i, "Ta": 10 * i} for i in range(11)],
            "binary alloy composition path",
        )
    )
    paths.append(
        (
            "Mo-W alloy (fixed 100 atoms)",
            [{"Mo": 100 - 10 * i, "W": 10 * i} for i in range(11)],
            "binary alloy composition path",
        )
    )
    paths.append(
        (
            "Ti-Zr dioxide substitution (99 atoms)",
            [{"Ti": 33 - 3 * i, "Zr": 3 * i, "O": 66} for i in range(12)],
            "33 cation sites exchanged; fixed O66; composition-space alloyed oxide probe",
        )
    )
    paths.append(
        (
            "Fe-Co monoxide substitution (100 atoms)",
            [{"Fe": 50 - 5 * i, "Co": 5 * i, "O": 50} for i in range(11)],
            "fixed-size composition-space oxide probe; mixed Fe/Co phase not asserted",
        )
    )
    paths.append(
        (
            "Organic C-H substitution probe (100 atoms)",
            [{"C": 20 + 3 * i, "H": 80 - 3 * i} for i in range(11)],
            "abstract fixed-size C/H composition path; not a molecular series",
        )
    )
    paths.append(
        (
            "Organic C-H-N substitution probe (100 atoms)",
            [{"C": 10 + 3 * i, "H": 50, "N": 40 - 3 * i} for i in range(11)],
            "abstract fixed-size C/H/N composition path; not a molecular series",
        )
    )
    paths.append(
        (
            "Organic C-H-O substitution probe (100 atoms)",
            [{"C": 10 + 3 * i, "H": 50, "O": 40 - 3 * i} for i in range(11)],
            "abstract fixed-size C/H/O composition path; not a molecular series",
        )
    )
    result: list[dict[str, Any]] = []
    for path_name, compositions, note in paths:
        for step, counts in enumerate(compositions):
            vector = _composition(counts)
            result.append(
                {
                    "path": path_name,
                    "step": step,
                    "t": step / (len(compositions) - 1),
                    "note": note,
                    "counts": counts,
                    "vector": vector,
                    "atom_count": int(vector.sum()),
                }
            )
    return result


def _evaluate(
    model: torch.nn.Module, matrix: np.ndarray, dtype: torch.dtype
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inputs = torch.as_tensor(matrix, dtype=dtype, device="cpu")
    with torch.inference_mode():
        logits = model(inputs)
        log_probabilities = F.log_softmax(logits, dim=-1)
        stable_weights = torch.exp(log_probabilities)
        raw_softmax = torch.softmax(logits, dim=-1)
        entropy = -(stable_weights * log_probabilities).sum(dim=-1)
    return tuple(
        value.detach().cpu().to(torch.float64).numpy()
        for value in (logits, log_probabilities, stable_weights, raw_softmax, entropy)
    )  # type: ignore[return-value]


def _metric(weight: np.ndarray, entropy_value: float | None = None) -> dict[str, Any]:
    order = np.argsort(-weight, kind="stable")
    entropy = entropy_value
    if entropy is None:
        entropy = float(-np.sum(weight[weight > 0] * np.log(weight[weight > 0])))
    return {
        "dominant_branch": f"branch-{order[0]}:{BRANCH_NAMES[order[0]]}",
        "max_weight": float(weight[order[0]]),
        "effective_branches": float(np.exp(entropy)),
        "ranking": order,
        "top3_weight": float(weight[order[:3]].sum()),
        "support90": int(np.searchsorted(np.cumsum(weight[order]), 0.90) + 1),
        "support95": int(np.searchsorted(np.cumsum(weight[order]), 0.95) + 1),
    }


def _xlog2x(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    positive = values > 0
    result[positive] = values[positive] * np.log2(values[positive])
    return result


def _js(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = (first + second) / 2.0
    divergence = (
        _xlog2x(first) + _xlog2x(second) - 2.0 * _xlog2x(midpoint)
    ).sum() / 2.0
    return max(0.0, float(divergence))


def _tv(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.abs(first - second).sum())


def _pairwise_matrices(
    vectors: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = vectors / vectors.sum(axis=1, keepdims=True)
    composition_l1 = np.abs(fractions[:, None, :] - fractions[None, :, :]).sum(axis=2)
    midpoint = (weights[:, None, :] + weights[None, :, :]) / 2.0
    js = np.maximum(
        0.0,
        (
            _xlog2x(weights[:, None, :])
            + _xlog2x(weights[None, :, :])
            - 2.0 * _xlog2x(midpoint)
        ).sum(axis=2)
        / 2.0,
    )
    tv = 0.5 * np.abs(weights[:, None, :] - weights[None, :, :]).sum(axis=2)
    return composition_l1, js, tv


def _pair_records(
    records: list[dict[str, Any]],
    weights: np.ndarray,
    size_records: list[dict[str, Any]],
    size_weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vectors = np.vstack([record["vector"] for record in records])
    comp_dist, js_dist, tv_dist = _pairwise_matrices(vectors, weights)
    pair_rows: list[dict[str, Any]] = []
    upper = np.triu_indices(len(records), 1)
    for left, right in zip(*upper):
        first, second = records[left], records[right]
        same_family = first["family"] == second["family"]
        distance = float(comp_dist[left, right])
        group: list[str] = []
        if first["atom_count"] == second["atom_count"] == 100:
            if distance <= 0.5:
                group.append("same_total_small_composition_distance")
            if distance >= 1.5:
                group.append("same_total_large_composition_distance")
            group.append("within_family" if same_family else "between_family")
        pair_rows.append(
            {
                "pair_id": f"{first['sample_id']}__{second['sample_id']}",
                "pair_group": ";".join(group) if group else "unmatched",
                "sample_a": first["sample_id"],
                "sample_b": second["sample_id"],
                "family_a": first["family"],
                "family_b": second["family"],
                "atom_count_a": first["atom_count"],
                "atom_count_b": second["atom_count"],
                "composition_l1_distance": distance,
                "jensen_shannon_divergence_bits": float(js_dist[left, right]),
                "total_variation_distance": float(tv_dist[left, right]),
                "dominant_branch_agreement": bool(
                    np.argmax(weights[left]) == np.argmax(weights[right])
                ),
                "distance_bin": int(np.floor(distance / 0.1)),
                "matched_set_id": "",
            }
        )

    # Match within-family and between-family pairs on 0.1-wide composition-L1 bins.
    within_indices: dict[int, list[int]] = {}
    between_indices: dict[int, list[int]] = {}
    for row_index, row in enumerate(pair_rows):
        if row["atom_count_a"] != 100 or row["atom_count_b"] != 100:
            continue
        bin_id = row["distance_bin"]
        target = (
            within_indices
            if row["family_a"] == row["family_b"]
            else between_indices
        )
        target.setdefault(bin_id, []).append(row_index)
    rng = np.random.default_rng(SEED)
    matched_groups: list[tuple[int, int, int]] = []
    for bin_id in sorted(set(within_indices) & set(between_indices)):
        count = min(len(within_indices[bin_id]), len(between_indices[bin_id]))
        within_choice = rng.choice(within_indices[bin_id], size=count, replace=False)
        between_choice = rng.choice(between_indices[bin_id], size=count, replace=False)
        for within_index, between_index in zip(within_choice, between_choice):
            matched_set_id = f"MATCH-{len(matched_groups) + 1:05d}"
            pair_rows[within_index]["matched_set_id"] = matched_set_id
            pair_rows[between_index]["matched_set_id"] = matched_set_id
            pair_rows[within_index]["pair_group"] += ";distance_matched_within_family"
            pair_rows[between_index]["pair_group"] += ";distance_matched_between_family"
            matched_groups.append((within_index, between_index, bin_id))

    small = [
        row for row in pair_rows
        if "same_total_small_composition_distance" in row["pair_group"]
    ]
    large = [
        row for row in pair_rows
        if "same_total_large_composition_distance" in row["pair_group"]
    ]

    def summarize_distance_group(
        rows: list[dict[str, Any]], same_family: bool
    ) -> dict[str, Any]:
        selected = [
            row
            for row in rows
            if (row["family_a"] == row["family_b"]) == same_family
        ]
        return {
            "pair_count": len(selected),
            "mean_js_bits": (
                float(np.mean([row["jensen_shannon_divergence_bits"] for row in selected]))
                if selected
                else None
            ),
            "mean_total_variation": (
                float(np.mean([row["total_variation_distance"] for row in selected]))
                if selected
                else None
            ),
            "dominant_branch_agreement_fraction": (
                float(np.mean([row["dominant_branch_agreement"] for row in selected]))
                if selected
                else None
            ),
        }

    matched_delta = np.array(
        [
            pair_rows[within_index]["jensen_shannon_divergence_bits"]
            - pair_rows[between_index]["jensen_shannon_divergence_bits"]
            for within_index, between_index, _ in matched_groups
        ],
        dtype=np.float64,
    )
    if matched_groups:
        endpoints = sorted({row["sample_id"] for row in records})
        index_by_id = {record["sample_id"]: i for i, record in enumerate(records)}
        bootstrap_values = []
        rng = np.random.default_rng(SEED + 1)
        for _ in range(N_BOOTSTRAPS):
            selected = rng.integers(0, len(endpoints), size=len(endpoints))
            multiplicity = np.bincount(selected, minlength=len(endpoints))
            within_values: list[float] = []
            between_values: list[float] = []
            within_weights: list[int] = []
            between_weights: list[int] = []
            for within_index, between_index, _ in matched_groups:
                within_row = pair_rows[within_index]
                between_row = pair_rows[between_index]
                wa = multiplicity[index_by_id[within_row["sample_a"]]] * multiplicity[
                    index_by_id[within_row["sample_b"]]
                ]
                wb = multiplicity[index_by_id[between_row["sample_a"]]] * multiplicity[
                    index_by_id[between_row["sample_b"]]
                ]
                if wa:
                    within_values.append(within_row["jensen_shannon_divergence_bits"])
                    within_weights.append(int(wa))
                if wb:
                    between_values.append(between_row["jensen_shannon_divergence_bits"])
                    between_weights.append(int(wb))
            if within_weights and between_weights:
                bootstrap_values.append(
                    np.average(within_values, weights=within_weights)
                    - np.average(between_values, weights=between_weights)
                )
        matched_ci = (
            np.quantile(bootstrap_values, [0.025, 0.975]).tolist()
            if bootstrap_values
            else [None, None]
        )
    else:
        matched_ci = [None, None]

    scale_rows = []
    for left in range(len(size_records)):
        for right in range(left + 1, len(size_records)):
            first, second = size_records[left], size_records[right]
            fraction_a = first["vector"] / first["vector"].sum()
            fraction_b = second["vector"] / second["vector"].sum()
            if np.allclose(fraction_a, fraction_b, atol=1e-12, rtol=0):
                scale_rows.append(
                    {
                        "pair_id": f"{first['sample_id']}__{second['sample_id']}",
                        "pair_group": "same_stoichiometry_different_atom_count",
                        "sample_a": first["sample_id"],
                        "sample_b": second["sample_id"],
                        "family_a": first["family"],
                        "family_b": second["family"],
                        "atom_count_a": first["atom_count"],
                        "atom_count_b": second["atom_count"],
                        "composition_l1_distance": 0.0,
                        "jensen_shannon_divergence_bits": _js(size_weights[left], size_weights[right]),
                        "total_variation_distance": _tv(size_weights[left], size_weights[right]),
                        "dominant_branch_agreement": bool(
                            np.argmax(size_weights[left]) == np.argmax(size_weights[right])
                        ),
                        "distance_bin": 0,
                        "matched_set_id": "",
                    }
                )
    pair_rows.extend(scale_rows)
    return pair_rows, {
        "within_small_distance_pair_count": len(
            [row for row in small if row["family_a"] == row["family_b"]]
        ),
        "between_small_distance_pair_count": len(
            [row for row in small if row["family_a"] != row["family_b"]]
        ),
        "within_large_distance_pair_count": len(
            [row for row in large if row["family_a"] == row["family_b"]]
        ),
        "between_large_distance_pair_count": len(
            [row for row in large if row["family_a"] != row["family_b"]]
        ),
        "matched_pair_count": len(matched_groups),
        "matched_common_distance_bins": sorted(
            {bin_id for _, _, bin_id in matched_groups}
        ),
        "matched_mean_within_minus_between_js_bits": (
            float(np.mean(matched_delta)) if matched_delta.size else None
        ),
        "same_total_small_distance_mean_js_bits": (
            float(np.mean([row["jensen_shannon_divergence_bits"] for row in small]))
            if small
            else None
        ),
        "same_total_large_distance_mean_js_bits": (
            float(np.mean([row["jensen_shannon_divergence_bits"] for row in large]))
            if large
            else None
        ),
        "small_distance": {
            "within_family": summarize_distance_group(small, True),
            "between_families": summarize_distance_group(small, False),
        },
        "large_distance": {
            "within_family": summarize_distance_group(large, True),
            "between_families": summarize_distance_group(large, False),
        },
        "matched_composition_bootstrap_ci95": matched_ci,
        "matched_bootstrap_replicates": len(bootstrap_values) if matched_groups else 0,
        "matched_bootstrap_unit": "composition IDs; endpoints weighted by bootstrap multiplicity",
        "same_stoichiometry_different_size_pair_count": len(scale_rows),
        "same_stoichiometry_different_size_mean_js": (
            float(np.mean([row["jensen_shannon_divergence_bits"] for row in scale_rows]))
            if scale_rows
            else None
        ),
    }

def _logit_diagnostics(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    additional: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined = records + additional
    matrix = np.vstack([record["vector"] for record in combined])
    logits64, logp64, weights64, raw64, entropy64 = _evaluate(
        model, matrix, torch.float64
    )
    model32 = copy.deepcopy(model).to(dtype=torch.float32)
    logits32, logp32, weights32, raw32, entropy32 = _evaluate(
        model32, matrix, torch.float32
    )
    if not np.isfinite(logits64).all() or not np.isfinite(logits32).all():
        raise ValueError("Nonfinite logits found")
    stable_error64 = float(np.max(np.abs(weights64 - raw64)))
    stable_error32 = float(np.max(np.abs(weights32 - raw32)))
    if not np.isfinite(entropy64).all() or not np.isfinite(entropy32).all():
        raise ValueError("Stable log-softmax entropy contains nonfinite values")
    thresholds = (1e-12, 1e-8, 1e-4, 1e-2)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(combined):
        order = np.argsort(logits64[index])
        metric64 = _metric(weights64[index], float(entropy64[index]))
        row: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "origin": record["origin"],
            "family": record["family"],
            "formula": record["formula"],
            "atom_count": record["atom_count"],
            "largest_logit_fp64": float(logits64[index, order[-1]]),
            "second_largest_logit_fp64": float(logits64[index, order[-2]]),
            "max_minus_second_logit_fp64": float(logits64[index, order[-1]] - logits64[index, order[-2]]),
            "minimum_logit_fp64": float(logits64[index].min()),
            "maximum_logit_fp64": float(logits64[index].max()),
            "exact_zero_weights_fp64": int(np.count_nonzero(weights64[index] == 0)),
            "nonzero_weights_fp64": int(np.count_nonzero(weights64[index])),
            "entropy_log_softmax_fp64": float(entropy64[index]),
            "entropy_from_weights_fp64": float(
                -np.sum(
                    weights64[index][weights64[index] > 0]
                    * np.log(weights64[index][weights64[index] > 0])
                )
            ),
            "effective_branches_fp64": float(np.exp(entropy64[index])),
            "max_weight_fp64": float(weights64[index].max()),
            "softmax_vs_exp_logsoftmax_max_abs_fp64": stable_error64,
            "softmax_vs_exp_logsoftmax_max_abs_fp32": stable_error32,
            "max_abs_logit_fp64_vs_fp32": float(
                np.max(np.abs(logits64[index] - logits32[index]))
            ),
            "max_abs_weight_fp64_vs_fp32": float(
                np.max(np.abs(weights64[index] - weights32[index]))
            ),
            "entropy_log_softmax_fp32": float(entropy32[index]),
            "effective_branches_fp32": float(np.exp(entropy32[index])),
            "max_weight_fp32": float(weights32[index].max()),
            "nonzero_weights_fp32": int(np.count_nonzero(weights32[index])),
            "dominant_branch_fp64": metric64["dominant_branch"],
            "dominant_branch_fp32": _metric(
                weights32[index], float(entropy32[index])
            )["dominant_branch"],
        }
        for threshold in thresholds:
            suffix = f"{threshold:g}".replace("-", "m")
            row[f"weights_below_{suffix}_fp64"] = int(
                np.count_nonzero(weights64[index] < threshold)
            )
            row[f"weights_below_{suffix}_fp32"] = int(
                np.count_nonzero(weights32[index] < threshold)
            )
        for branch_index, name in enumerate(BRANCH_NAMES):
            row[f"logit_fp64_{branch_index:02d}_{name}"] = float(
                logits64[index, branch_index]
            )
            row[f"logit_fp32_{branch_index:02d}_{name}"] = float(
                logits32[index, branch_index]
            )
            row[f"log_probability_fp64_{branch_index:02d}_{name}"] = float(
                logp64[index, branch_index]
            )
            row[f"weight_fp64_{branch_index:02d}_{name}"] = float(
                weights64[index, branch_index]
            )
        rows.append(row)
    stats = {
        "record_count": len(combined),
        "first_round_count": len(records),
        "size_sensitivity_count": len(additional),
        "all_logits_finite_fp64": bool(np.isfinite(logits64).all()),
        "all_logits_finite_fp32": bool(np.isfinite(logits32).all()),
        "max_softmax_exp_logsoftmax_abs_error_fp64": stable_error64,
        "max_softmax_exp_logsoftmax_abs_error_fp32": stable_error32,
        "max_abs_logit_fp64_vs_fp32": float(np.max(np.abs(logits64 - logits32))),
        "max_abs_weight_fp64_vs_fp32": float(np.max(np.abs(weights64 - weights32))),
        "fp64_exact_zero_count": int(np.count_nonzero(weights64 == 0)),
        "fp32_exact_zero_count": int(np.count_nonzero(weights32 == 0)),
        "fp64_threshold_counts": {
            str(threshold): int(np.count_nonzero(weights64 < threshold))
            for threshold in thresholds
        },
        "fp32_threshold_counts": {
            str(threshold): int(np.count_nonzero(weights32 < threshold))
            for threshold in thresholds
        },
        "entropy_method": "-sum(exp(log_softmax(logits)) * log_softmax(logits))",
        "temperature_scaling_used": False,
    }
    return rows, {
        **stats,
        "matrix": matrix,
        "logits64": logits64,
        "weights64": weights64,
        "weights32": weights32,
        "entropy64": entropy64,
    }


def _size_sensitivity(
    size_records: list[dict[str, Any]],
    weights: np.ndarray,
    entropies: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    start = 0
    by_family: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {}
    for record, weight, entropy in zip(size_records, weights, entropies):
        by_family.setdefault(record["family"], []).append(
            (record, weight, float(entropy))
        )
    for family, items in by_family.items():
        reference_record, reference_weight, _ = items[0]
        reference_order = np.argsort(-reference_weight, kind="stable")
        for record, weight, entropy in items:
            metric = _metric(weight, entropy)
            order = metric["ranking"]
            reference_rank = np.empty(N_BRANCHES, dtype=np.int64)
            current_rank = np.empty(N_BRANCHES, dtype=np.int64)
            reference_rank[reference_order] = np.arange(N_BRANCHES)
            current_rank[order] = np.arange(N_BRANCHES)
            output.append(
                {
                    "sample_id": record["sample_id"],
                    "family": family,
                    "scale_step": record["step"],
                    "atom_count": record["atom_count"],
                    "chemical_formula": _formula(record["vector"]),
                    "element_counts_json": _vector_json(record["vector"]),
                    "atomic_fractions_json": _vector_json(record["vector"], fractional=True),
                    "reference_sample_id": reference_record["sample_id"],
                    "reference_atom_count": reference_record["atom_count"],
                    "dominant_branch": metric["dominant_branch"],
                    "maximum_branch_weight": metric["max_weight"],
                    "effective_number_of_branches": metric["effective_branches"],
                    "js_divergence_vs_smallest_bits": _js(reference_weight, weight),
                    "total_variation_vs_smallest": _tv(reference_weight, weight),
                    "branch_rank_changes_vs_smallest": int(
                        np.count_nonzero(current_rank != reference_rank)
                    ),
                    "kendall_rank_correlation_vs_smallest": float(
                        kendalltau(reference_rank, current_rank).statistic
                    ),
                    "top3_branch_ids": ";".join(
                        f"branch-{i}:{BRANCH_NAMES[i]}" for i in order[:3]
                    ),
                    **{
                        f"branch_{i:02d}_{name}": float(value)
                        for i, (name, value) in enumerate(zip(BRANCH_NAMES, weight))
                    },
                }
            )
    return output


def _substitution_rows(
    path_records: list[dict[str, Any]], weights: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(path_records):
        groups.setdefault(record["path"], []).append(index)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for path, indices in groups.items():
        order = np.argsort(weights[indices].mean(axis=0))[::-1][:5]
        js_steps = []
        tv_steps = []
        dominant = []
        for local_index, index in enumerate(indices):
            record = path_records[index]
            metric = _metric(weights[index])
            if local_index:
                previous = weights[indices[local_index - 1]]
                js_value = _js(previous, weights[index])
                tv_value = _tv(previous, weights[index])
                js_steps.append(js_value)
                tv_steps.append(tv_value)
            else:
                js_value = float("nan")
                tv_value = float("nan")
            dominant.append(metric["dominant_branch"])
            rows.append(
                {
                    "path": path,
                    "step": record["step"],
                    "t": record["t"],
                    "chemical_formula": _formula(record["vector"]),
                    "element_counts_json": _vector_json(record["vector"]),
                    "atomic_fractions_json": _vector_json(record["vector"], fractional=True),
                    "atom_count": record["atom_count"],
                    "probe_note": record["note"],
                    "dominant_branch": metric["dominant_branch"],
                    "maximum_branch_weight": metric["max_weight"],
                    "effective_number_of_branches": metric["effective_branches"],
                    "adjacent_js_divergence_bits": js_value,
                    "adjacent_total_variation_distance": tv_value,
                    "top3_branch_ids": ";".join(
                        f"branch-{i}:{BRANCH_NAMES[i]}" for i in metric["ranking"][:3]
                    ),
                    **{
                        f"branch_{i:02d}_{name}": float(value)
                        for i, (name, value) in enumerate(zip(BRANCH_NAMES, weights[index]))
                    },
                }
            )
        summary[path] = {
            "note": path_records[indices[0]]["note"],
            "sample_count": len(indices),
            "atom_counts": sorted({path_records[index]["atom_count"] for index in indices}),
            "dominant_start": dominant[0],
            "dominant_end": dominant[-1],
            "dominant_branch_changes": int(sum(a != b for a, b in zip(dominant, dominant[1:]))),
            "largest_adjacent_js_bits": float(max(js_steps)) if js_steps else 0.0,
            "largest_adjacent_tv": float(max(tv_steps)) if tv_steps else 0.0,
            "top_five_branches": [
                f"branch-{i}:{BRANCH_NAMES[i]}" for i in order
            ],
            "largest_step_description": (
                f"step {int(np.argmax(js_steps)) + 1} of {len(indices) - 1}"
                if js_steps
                else "not applicable"
            ),
        }
    return rows, summary


def _permutation_association(
    vectors: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    comp, observed_js, _ = _pairwise_matrices(vectors, weights)
    upper = np.triu_indices(len(vectors), 1)
    x = comp[upper]
    y = observed_js[upper]
    observed_rho_value = spearmanr(x, y).statistic
    if not np.isfinite(observed_rho_value):
        raise ValueError("Composition-distance/routing association is not estimable")
    observed_rho = float(observed_rho_value)
    rng = np.random.default_rng(SEED + 2)
    null = np.empty(N_PERMUTATIONS, dtype=np.float64)
    for iteration in range(N_PERMUTATIONS):
        permutation = rng.permutation(len(weights))
        _, js, _ = _pairwise_matrices(vectors, weights[permutation])
        null[iteration] = spearmanr(x, js[upper]).statistic
    valid_null = null[np.isfinite(null)]
    if not valid_null.size:
        raise ValueError("No valid permutation correlations were estimable")
    p_value = (1 + np.count_nonzero(np.abs(valid_null) >= abs(observed_rho))) / (
        len(valid_null) + 1
    )
    # Bootstrap the composition-level units, never individual pairs.
    bootstrap = []
    rng = np.random.default_rng(SEED + 3)
    for _ in range(N_BOOTSTRAPS):
        indices = rng.integers(0, len(vectors), size=len(vectors))
        sampled_vectors = vectors[indices]
        sampled_weights = weights[indices]
        sampled_comp, sampled_js, _ = _pairwise_matrices(sampled_vectors, sampled_weights)
        sampled_upper = np.triu_indices(len(indices), 1)
        bootstrap.append(
            float(
                spearmanr(
                    sampled_comp[sampled_upper], sampled_js[sampled_upper]
                ).statistic
            )
        )
    valid_bootstrap = np.asarray(bootstrap, dtype=np.float64)
    valid_bootstrap = valid_bootstrap[np.isfinite(valid_bootstrap)]
    return {
        "observed_spearman_rho": observed_rho,
        "composition_level_bootstrap_ci95": (
            np.quantile(valid_bootstrap, [0.025, 0.975]).tolist()
            if valid_bootstrap.size
            else [None, None]
        ),
        "valid_bootstrap_replicates": int(valid_bootstrap.size),
        "permutation_null_mean": float(valid_null.mean()),
        "permutation_null_sd": float(valid_null.std(ddof=1)),
        "permutation_null_95_interval": np.quantile(
            valid_null, [0.025, 0.975]
        ).tolist(),
        "two_sided_permutation_p": float(p_value),
        "permutations": N_PERMUTATIONS,
        "valid_permutations": int(valid_null.size),
        "bootstrap_replicates": N_BOOTSTRAPS,
        "permutation_unit": "complete 16-branch weight vectors permuted among compositions within the 100-atom stratum",
        "bootstrap_unit": "composition records sampled with replacement; pair distances recomputed",
        "pair_count": int(len(x)),
        "null_values": null,
    }


def _figure_size_sensitivity(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    frame = pd.DataFrame(rows)
    families = frame["family"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, family in zip(axes.flat[:5], families):
        group = frame[frame["family"] == family].sort_values("atom_count")
        branch_columns = [
            f"branch_{i:02d}_{name}" for i, name in enumerate(BRANCH_NAMES)
        ]
        values = group[branch_columns].to_numpy()
        top = np.argsort(values.mean(axis=0))[::-1][:5]
        for branch_index in top:
            axis.plot(
                group["atom_count"],
                values[:, branch_index],
                marker="o",
                color=BRANCH_COLORS[branch_index],
                label=BRANCH_LABELS[branch_index],
            )
        axis.set_title(family)
        axis.set_xlabel("Total atom count")
        axis.set_ylabel("Branch weight")
        axis.grid(alpha=0.2)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.03), fontsize=8)
    fig.suptitle("Count-scale sensitivity at fixed elemental fractions\n(original raw-count input; no normalization)")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(output_dir / "fig_size_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_size_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6))
    for family in families:
        group = frame[frame["family"] == family].sort_values("atom_count")
        axis.plot(
            group["atom_count"],
            group["js_divergence_vs_smallest_bits"],
            marker="o",
            label=family,
        )
    axis.set_xlabel("Total atom count")
    axis.set_ylabel("JS divergence vs. smallest count vector (bits)")
    axis.set_title("Routing change under proportional count scaling")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_size_sensitivity_js.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_size_sensitivity_js.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _figure_logit_diagnostics(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    size_frame = frame[frame["origin"] == "variable_size_synthetic"]
    fixed_frame = frame[frame["origin"] == "first_round_fixed_100_atom_synthetic"]
    axes[0, 0].scatter(
        fixed_frame["atom_count"],
        fixed_frame["max_minus_second_logit_fp64"],
        s=14,
        alpha=0.35,
        label="First-round fixed-size probes",
    )
    axes[0, 0].scatter(
        size_frame["atom_count"],
        size_frame["max_minus_second_logit_fp64"],
        s=24,
        alpha=0.8,
        label="Variable-size series",
    )
    axes[0, 0].set_xlabel("Total atom count")
    axes[0, 0].set_ylabel("Largest − second-largest logit")
    axes[0, 0].set_title("Logit separation")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].scatter(
        fixed_frame["atom_count"],
        fixed_frame["effective_branches_fp64"],
        s=14,
        alpha=0.35,
        label="First-round fixed-size probes",
    )
    axes[0, 1].scatter(
        size_frame["atom_count"],
        size_frame["effective_branches_fp64"],
        s=24,
        alpha=0.8,
        label="Variable-size series",
    )
    axes[0, 1].set_xlabel("Total atom count")
    axes[0, 1].set_ylabel("Effective number of branches")
    axes[0, 1].set_title("Routing concentration")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].hist(frame["max_weight_fp64"], bins=25, color="#4c78a8", alpha=0.8)
    axes[1, 0].set_xlabel("Maximum branch weight")
    axes[1, 0].set_ylabel("Composition count")
    axes[1, 0].set_title("Maximum-weight distribution")
    axes[1, 1].hist(
        frame["nonzero_weights_fp64"],
        bins=np.arange(-0.5, N_BRANCHES + 1.5, 1),
        color="#f58518",
        alpha=0.8,
    )
    axes[1, 1].set_xlabel("Numerically nonzero FP64 weights")
    axes[1, 1].set_ylabel("Composition count")
    axes[1, 1].set_title("Nonzero-weight count (underflow-sensitive)")
    fig.suptitle("Softmax numerical diagnostics; exact zeros are underflow, not branch pruning")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_logit_saturation.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_logit_saturation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _figure_substitution(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    frame = pd.DataFrame(rows)
    paths = frame["path"].drop_duplicates().tolist()
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharey=True)
    for axis, path in zip(axes.flat, paths):
        group = frame[frame["path"] == path].sort_values("step")
        columns = [f"branch_{i:02d}_{name}" for i, name in enumerate(BRANCH_NAMES)]
        values = group[columns].to_numpy()
        top = np.argsort(values.mean(axis=0))[::-1][:5]
        for branch_index in top:
            axis.plot(
                group["t"],
                values[:, branch_index],
                color=BRANCH_COLORS[branch_index],
                marker=".",
                linewidth=1.6,
                label=BRANCH_LABELS[branch_index],
            )
        other = np.clip(1.0 - values[:, top].sum(axis=1), 0.0, 1.0)
        axis.plot(group["t"], other, color="0.35", linestyle="--", label="Other (remaining 11)")
        axis.set_title(path, fontsize=9)
        axis.set_xlabel("Substitution fraction t")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(paths):]:
        axis.axis("off")
    axes.flat[0].set_ylabel("Branch weight")
    axes.flat[3].set_ylabel("Branch weight")
    axes.flat[6].set_ylabel("Branch weight")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=8)
    fig.suptitle(
        "Learned preferences along controlled composition paths\n"
        "Oxide/organic paths are composition-space probes; no phase transition is implied"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output_dir / "fig_substitution_routing.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_substitution_routing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _figure_similarity(
    pair_rows: list[dict[str, Any]], permutation: dict[str, Any], output_dir: Path
) -> None:
    frame = pd.DataFrame(pair_rows)
    fixed = frame[frame["atom_count_a"].eq(100) & frame["atom_count_b"].eq(100)]
    comp = fixed["composition_l1_distance"].to_numpy()
    js_values = fixed["jensen_shannon_divergence_bits"].to_numpy()
    sample = np.arange(len(comp))
    if len(sample) > 4000:
        sample = np.random.default_rng(SEED).choice(sample, 4000, replace=False)

    matched = fixed[fixed["matched_set_id"].ne("")]
    matched_within = matched[matched["pair_group"].str.contains("distance_matched_within_family")]
    matched_between = matched[matched["pair_group"].str.contains("distance_matched_between_family")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(comp[sample], js_values[sample], s=9, alpha=0.25, color="#4c78a8")
    axes[0].set_xlabel("Atomic-fraction L1 distance")
    axes[0].set_ylabel("Routing JS divergence (bits)")
    axes[0].set_title("Same 100-atom stratum")
    axes[0].grid(alpha=0.2)
    axes[1].boxplot(
        [
            matched_within["jensen_shannon_divergence_bits"].to_numpy(),
            matched_between["jensen_shannon_divergence_bits"].to_numpy(),
        ],
        tick_labels=[f"Within family\n(n={len(matched_within)})", f"Between families\n(n={len(matched_between)})"],
        showfliers=False,
    )
    axes[1].set_ylabel("Routing JS divergence (bits)")
    axes[1].set_title("Matched by L1-distance bins (width 0.1)")
    axes[1].grid(axis="y", alpha=0.2)
    axes[2].hist(
        permutation["null_values"][np.isfinite(permutation["null_values"])],
        bins=25,
        color="#bab0ac",
        edgecolor="white",
    )
    axes[2].axvline(
        permutation["observed_spearman_rho"],
        color="#e45756",
        linewidth=2,
        label=f"Observed ρ={permutation['observed_spearman_rho']:.3f}",
    )
    axes[2].set_xlabel("Spearman ρ after vector permutation")
    axes[2].set_ylabel("Permutation count")
    axes[2].set_title(
        f"Permutation null; p={permutation['two_sided_permutation_p']:.3g}"
    )
    axes[2].legend(fontsize=8)
    fig.suptitle("Composition–routing relationship with count-size and distance controls")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_similarity_controls.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_similarity_controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _branch_mapping_provenance() -> dict[str, Any]:
    config_path = SC26_DIR / "multidataset_hpo-BEST6-fp64" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    heads = config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"]
    config_branch_ids = [head["type"] for head in heads]

    launch_path = SC26_DIR / "SC26-job-best-continue-aurora.sh"
    launch_text = launch_path.read_text(encoding="utf-8")
    aliases = {
        int(index): name
        for index, name in re.findall(r"^export datadir(\d+)=(\S+)$", launch_text, re.M)
    }
    alias_order = [aliases[index] for index in sorted(aliases)]

    log_path = SC26_DIR / "multidataset_hpo-BEST6-fp64" / "run-4284444.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"--multi_model_list=([^\s]+)", log_text)
    if not match:
        raise ValueError("Could not recover --multi_model_list from run log")
    logged_order = match.group(1).split(",")

    training_source = (SC26_DIR / "gfm_mlip_all_mpnn.py").read_text(encoding="utf-8")
    source_has_enumerated_mapping = (
        "for i, name in enumerate(modellist):" in training_source
        and "torch.tensor([[i]])" in training_source
    )
    utils_source = (SC26_DIR / "utils.py").read_text(encoding="utf-8")
    source_validates_zero_based_ids = (
        "dataset_ids < 0" in utils_source and "dataset_ids >= num_branches" in utils_source
    )

    checkpoint = torch.load(BACKBONE_PATH, map_location="cpu", weights_only=True)
    model_state = checkpoint.get("model_state_dict", {})
    checkpoint_branch_ids = sorted(
        {
            f"branch-{match.group(1)}"
            for key in model_state
            if (match := re.search(r"heads_NN\.\d+\.branch-(\d+)\.", key))
        },
        key=lambda value: int(value.split("-")[1]),
    )
    expected_ids = [f"branch-{index}" for index in range(N_BRANCHES)]
    verified = (
        len(config_branch_ids) == N_BRANCHES
        and config_branch_ids == expected_ids
        and alias_order == list(BRANCH_NAMES)
        and logged_order == list(BRANCH_NAMES)
        and source_has_enumerated_mapping
        and source_validates_zero_based_ids
        and checkpoint_branch_ids == expected_ids
    )
    if not verified:
        raise ValueError(
            "Branch mapping discrepancy: "
            f"config={config_branch_ids}, aliases={alias_order}, "
            f"log={logged_order}, checkpoint={checkpoint_branch_ids}"
        )
    return {
        "status": "verified",
        "dataset_branch_order": [
            {"branch_id_zero_based": index, "branch_name": name}
            for index, name in enumerate(BRANCH_NAMES)
        ],
        "evidence": {
            "training_config": str(config_path.relative_to(SC26_DIR)),
            "training_config_graph_head_order": config_branch_ids,
            "launch_script": str(launch_path.relative_to(SC26_DIR)),
            "launch_alias_order": alias_order,
            "run_log": str(log_path.relative_to(SC26_DIR)),
            "run_log_multi_model_list_order": logged_order,
            "training_code_enumerates_model_order_zero_based": source_has_enumerated_mapping,
            "dataset_ids_validated_in_zero_based_range": source_validates_zero_based_ids,
            "hydragNN_checkpoint_branch_ids": checkpoint_branch_ids,
        },
        "interpretation": "branch-k is zero-based and associated with the kth name in multi_model_list",
        "mlp_checkpoint_branch_order_metadata": "not stored inside the MLP checkpoint; pairing/mapping comes from matched run metadata",
    }


def _real_composition_audit() -> dict[str, Any]:
    log_path = SC26_DIR / "multidataset_hpo-BEST6-fp64" / "run-4284444.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    dataset_root_matches = re.findall(r"Adios reading: ([^\s]+)", log_text)
    observed_roots = sorted(
        {
            str(Path(value.replace("/./", "/")).parent)
            for value in dataset_root_matches
        }
    )
    root = Path(
        "/lustre/orion/world-shared/lrn070/jyc/frontier/HydraGNN/examples/multidataset_hpo_sc26/dataset"
    )
    dataset_files = [
        root / f"{name}-v2.bp" for name in BRANCH_NAMES
    ]
    paths_checked = [
        {
            "path": str(path),
            "exists": path.exists(),
            "readable": path.is_file() and os.access(path, os.R_OK),
            "source": "Adios dataset location inferred from run-4284444.log",
        }
        for path in dataset_files
    ]
    paths_checked.extend(
        [
            {
                "path": str(SC26_DIR / "dataset"),
                "exists": (SC26_DIR / "dataset").exists(),
                "readable": (SC26_DIR / "dataset").is_dir()
                and bool(list((SC26_DIR / "dataset").glob("*"))),
                "source": "local analysis checkout",
            },
            {
                "path": "/Users/7ml/Documents/Codes/HydraGNN/examples/multidataset_hpo_sc26/dataset",
                "exists": Path("/Users/7ml/Documents/Codes/HydraGNN/examples/multidataset_hpo_sc26/dataset").exists(),
                "readable": Path("/Users/7ml/Documents/Codes/HydraGNN/examples/multidataset_hpo_sc26/dataset").is_dir()
                and bool(list(Path("/Users/7ml/Documents/Codes/HydraGNN/examples/multidataset_hpo_sc26/dataset").glob("*"))),
                "source": "nearby local checkout",
            },
        ]
    )
    return {
        "status": "not available locally",
        "required_archived_structures_for_encoding_check": 3,
        "archived_structures_audited": 0,
        "paths_checked": paths_checked,
        "run_log_dataset_directories": observed_roots,
        "relevant_data_reconstructable_from_local_metadata": False,
        "available_metadata": [
            "run logs mention ADIOS reads and trainset/valset/testset chemical_composition arrays, but contain no per-record vectors",
            "dataset histogram script requires source datasets and reports missing paths",
        ],
        "documented_real_formulas_obtained": False,
        "possible_without_calculations": "Yes, externally curated composition records can be obtained from public crystallographic/materials databases without running calculations, but none were downloaded or audited in this run. They would be real-composition probes, not evidence of training membership.",
        "no_training_domain_labels_assigned": True,
        "reason": "No checked ADIOS archives or other local structure files were present or readable; cluster storage roots are not mounted.",
    }


def _size_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    for family, group in frame.groupby("family", sort=False):
        ordered = group.sort_values("atom_count")
        summaries[family] = {
            "atom_counts": ordered["atom_count"].astype(int).tolist(),
            "dominant_branches": ordered["dominant_branch"].tolist(),
            "maximum_js_vs_smallest_bits": float(
                ordered["js_divergence_vs_smallest_bits"].max()
            ),
            "maximum_tv_vs_smallest": float(
                ordered["total_variation_vs_smallest"].max()
            ),
            "maximum_weight_range": [
                float(ordered["maximum_branch_weight"].min()),
                float(ordered["maximum_branch_weight"].max()),
            ],
            "effective_branch_range": [
                float(ordered["effective_number_of_branches"].min()),
                float(ordered["effective_number_of_branches"].max()),
            ],
            "rank_change_range": [
                int(ordered["branch_rank_changes_vs_smallest"].min()),
                int(ordered["branch_rank_changes_vs_smallest"].max()),
            ],
        }
    return summaries


def _report(
    output_dir: Path,
    summary: dict[str, Any],
    size_summary: dict[str, Any],
    path_summary: dict[str, Any],
) -> None:
    size_lines = []
    for family, stats in size_summary.items():
        size_lines.append(
            f"- **{family}:** N={stats['atom_counts']}; max JS vs smallest "
            f"{stats['maximum_js_vs_smallest_bits']:.4f} bits, max TV "
            f"{stats['maximum_tv_vs_smallest']:.4f}; dominant branches "
            f"{', '.join(stats['dominant_branches'])}; effective-support range "
            f"{stats['effective_branch_range'][0]:.3f}–{stats['effective_branch_range'][1]:.3f}."
        )
    path_lines = []
    for name, stats in path_summary.items():
        path_lines.append(
            f"- **{name}:** {stats['sample_count']} points, dominant branch "
            f"{stats['dominant_start']} → {stats['dominant_end']}, "
            f"{stats['dominant_branch_changes']} dominant changes; max adjacent JS "
            f"{stats['largest_adjacent_js_bits']:.4f} bits, max adjacent TV "
            f"{stats['largest_adjacent_tv']:.4f}. {stats['note']}"
        )
    association = summary["similarity_controls"]["permutation"]
    matched = summary["similarity_controls"]["matched"]
    small_within = matched["small_distance"]["within_family"]
    small_between = matched["small_distance"]["between_families"]
    large_within = matched["large_distance"]["within_family"]
    large_between = matched["large_distance"]["between_families"]

    def format_metric(value: Any, format_spec: str) -> str:
        return format(value, format_spec) if value is not None else "not estimable"
    report_text = f"""# BranchWeightMLP robustness analysis (second round)

## Executive scientific assessment

1. **Atom-count sensitivity:** proportional rescaling of raw count vectors
   changes routing in the specified families; the maximum per-family JS and TV
   changes are reported below. Therefore the input representation is not
   scale-invariant. This is a characterization, not a claim that scale
   invariance was intended.
2. **Softmax saturation:** concentration and underflow are reproducible for the
   supplied checkpoint. Across first-round and size probes, {summary['logit_diagnostics']['fp64_exact_zero_count']}
   of {summary['logit_diagnostics']['record_count'] * N_BRANCHES} FP64 values
   were exactly zero after softmax. This is numerical underflow, not explicit
   branch pruning. Size trends are summarized in the logit diagnostics and
   figure; training atom-count distributions were unavailable, so “realistic”
   training sizes cannot be verified.
3. **Chemistry–routing after count control:** all first-round compositions have
   exactly 100 atoms. Their matched-pair and complete-vector permutation
   results therefore isolate composition better than the first-round
   unadjusted comparison. Observed Spearman rho was
   {association['observed_spearman_rho']:.4f}, composition-level bootstrap 95%
   interval [{association['composition_level_bootstrap_ci95'][0]:.4f},
   {association['composition_level_bootstrap_ci95'][1]:.4f}], versus the
   within-size-stratum permutation null (mean
   {association['permutation_null_mean']:.4f}, 95% interval
   [{association['permutation_null_95_interval'][0]:.4f},
   {association['permutation_null_95_interval'][1]:.4f}], two-sided p
   {association['two_sided_permutation_p']:.4g}). This remains synthetic-only,
   with overlapping pairwise comparisons and constructed motif labels.
4. **Element substitutions:** measured path changes are listed below. A head
   change or sharp weight movement describes the learned preference response;
   it is not a phase transition or proof of specialization.
5. **Branch–chemistry associations:** the mapping is independently confirmed
   as zero-based using the config head list, launcher and training log
   `multi_model_list`, dataset-name enumeration in training code, and branch
   IDs in the backbone checkpoint. Even so, synthetic motif-to-head association
   is not empirical dataset specialization.
6. **Manuscript evidence:** these robustness results do not yet justify a
   manuscript figure about real chemical specialization. A synthetic
   supplementary/diagnostic figure may be useful if clearly labeled. Archived
   compositions and training-distribution statistics remain necessary for
   scientific claims about the target evaluation domain.

## Scope, checkpoint, and model

The trained MLP weights were not modified; no training, HydraGNN backbone
inference, DFT, or HPC jobs were run. The first-round results remain unchanged.

- MLP SHA-256: `{summary['checkpoint_manifest']['mlp_sha256']}`
- HydraGNN checkpoint SHA-256: `{summary['checkpoint_manifest']['backbone_sha256']}`
- Base MLP: 118 -> 128 -> 64 -> 16, float64; Git commit
  `{summary['checkpoint_manifest']['git_commit']}`.
- Pairing is user-confirmed. The MLP checkpoint does not itself record the
  backbone checkpoint hash.
- Full branch mapping is in `branch_mapping_provenance.json`; all branch IDs are
  zero-based.
- No discrepancy was found in the checkpoint hashes or model dimensions.

## Test 1 — Proportional atom-count scaling

**Hypothesis:** with raw-count inputs, identical elemental fractions at
different total counts may yield different full weight distributions.

**Construction:** exact integer series for Nb-Ta, TiO2, CH4, LiFePO4, and a
Nb-Ta-V-Hf-Zr alloy. Inputs remain unnormalized. Each row, all 16 weights,
dominant branch, top-three, effective support, rank changes, JS, and TV are in
`size_sensitivity.csv`.

{chr(10).join(size_lines)}

**Conclusion:** this model should not be assumed invariant to system size:
proportional count scaling is an actual change in input magnitude. Whether
these sizes occur in training cannot be established because atom-count records
are unavailable.

## Test 2 — Logit spread and numerical saturation

`logit_diagnostics.csv` records FP64 and FP32 logits, FP64 log probabilities,
full weights, logit ranges, largest-minus-second-largest logit gap, exact zeros,
counts below 1e-12/1e-8/1e-4/1e-2, entropy from `log_softmax`, and effective
branch count. Entropy is computed as
`-sum(exp(log_softmax(logits)) * log_softmax(logits))`.

- Logits finite in FP64: {summary['logit_diagnostics']['all_logits_finite_fp64']};
  FP32: {summary['logit_diagnostics']['all_logits_finite_fp32']}.
- Maximum absolute difference between FP64 softmax and `exp(log_softmax)`:
  {summary['logit_diagnostics']['max_softmax_exp_logsoftmax_abs_error_fp64']:.3e}.
- Maximum FP64-vs-FP32 logit difference:
  {summary['logit_diagnostics']['max_abs_logit_fp64_vs_fp32']:.6g}; weight
  difference: {summary['logit_diagnostics']['max_abs_weight_fp64_vs_fp32']:.6g}.
- Exact zero FP64 weights: {summary['logit_diagnostics']['fp64_exact_zero_count']};
  FP32: {summary['logit_diagnostics']['fp32_exact_zero_count']}.
- Temperature scaling was not used. No logits were clamped.

**Conclusion:** the zero weights are finite-precision softmax underflow, not a
runtime branch-activation decision. The plot contrasts 100-atom records with
the specified variable-size probes. Training structure sizes are missing, so
we cannot classify the observed saturation as typical or atypical for training.

## Test 3 — Archived data and composition encoding

The local analysis checkout and nearby HydraGNN checkout contain no dataset
directory. The ADIOS dataset paths referenced by the Frontier run log were
checked for all 16 `*-v2.bp` files; the `/lustre` root is not mounted here and
the paths are unavailable. The logs show that ADIOS records contain
`trainset/chemical_composition`, `valset/chemical_composition`, and
`testset/chemical_composition` variables, but logs do not provide per-record
vectors. No three-structure stored-field comparison could be conducted.
`real_composition_audit.json` records each exact path and its status.

The repository code supports the code-level conclusion that `chemical_composition`
is a raw 118-element atomic-number count histogram, feature index Z-1 is
element Z, and no normalization is applied before the MLP. This still needs
three archived-record comparisons for independent validation. Public curated
composition records could be obtained without running calculations, but none
were acquired here; if added later they must be labeled real-composition
probes, not training-domain validation. No training/validation split or domain
labels were invented.

## Test 4 — Controlled elemental substitutions

Seven exact-count paths include Nb-Ta, Mo-W, Ti-Zr dioxide, Fe-Co monoxide,
and C/H, C/H/N, C/H/O composition-space probes. The organic paths are explicitly
abstract fixed-size elemental substitutions, not claimed molecular series.
The oxide paths are composition probes; no structural stability is asserted.
Full weights and adjacent JS/TV values are in `substitution_paths.csv`.

{chr(10).join(path_lines)}

No branch transition is interpreted as a phase transition. “Gradual” versus
“abrupt” refers only to the per-step measured weight distance on the chosen
discretized path.

## Test 5 — Similarity controls

All first-round compositions are fixed at 100 atoms. `matched_similarity.csv`
contains pair-level atomic-fraction L1, JS divergence, total variation, and
dominant-branch agreement. It separates same-size small/large composition
distances and provides matched within-/between-family pairs in common
0.1-wide L1 bins. Same-stoichiometry, different-size pairs from Test 1 are
also recorded.

- Same-size small-distance pairs (L1 <= 0.5): within-family
  n={small_within['pair_count']}, mean JS
  {format_metric(small_within['mean_js_bits'], '.4f')} bits, mean TV
  {format_metric(small_within['mean_total_variation'], '.4f')}, dominant-branch agreement
  {format_metric(small_within['dominant_branch_agreement_fraction'], '.1%')}; between-family
  n={small_between['pair_count']}, mean JS
  {format_metric(small_between['mean_js_bits'], '.4f')} bits, mean TV
  {format_metric(small_between['mean_total_variation'], '.4f')}, dominant-branch agreement
  {format_metric(small_between['dominant_branch_agreement_fraction'], '.1%')}.
- Same-size large-distance pairs (L1 >= 1.5): within-family
  n={large_within['pair_count']}, mean JS
  {format_metric(large_within['mean_js_bits'], '.4f')} bits, mean TV
  {format_metric(large_within['mean_total_variation'], '.4f')}, dominant-branch agreement
  {format_metric(large_within['dominant_branch_agreement_fraction'], '.1%')}; between-family
  n={large_between['pair_count']}, mean JS
  {format_metric(large_between['mean_js_bits'], '.4f')} bits, mean TV
  {format_metric(large_between['mean_total_variation'], '.4f')}, dominant-branch agreement
  {format_metric(large_between['dominant_branch_agreement_fraction'], '.1%')}.
- Distance-matched within-/between-family comparisons:
  {matched['matched_pair_count']} matched pair sets across bins
  {matched['matched_common_distance_bins']}; mean within-minus-between JS
  {matched['matched_mean_within_minus_between_js_bits'] if matched['matched_mean_within_minus_between_js_bits'] is not None else 'not estimable'}
  bits, composition-bootstrap 95% CI
  {matched['matched_composition_bootstrap_ci95']}.
- Same stoichiometry but different counts: {matched['same_stoichiometry_different_size_pair_count']}
  pairs, mean JS {matched['same_stoichiometry_different_size_mean_js']:.4f} bits.
- Permutation shuffles complete 16-weight vectors among fixed-size composition
  records; bootstrap resamples composition records, not pair rows.

If matched common support is small, do not interpret within-/between-family
differences as controlled chemical-family effects. Constructed motif families
remain synthetic descriptors.

## Test 6 — Training-domain proximity

`domain_proximity.csv` was not generated because no training composition
reference vectors are accessible. Exact-composition occurrence, element-set
coverage, nearest-training-composition distance, and in-/out-of-domain labels
are therefore unavailable.

## Branch mapping provenance

The ordered config output heads are `branch-0` through `branch-15`. The launcher
defines the ordered dataset aliases; the run log records the corresponding
16-name `multi_model_list`; training code enumerates that list and maps each
name to `torch.tensor([[i]])`; and the loaded HydraGNN checkpoint has branch
parameter keys 0 through 15. The full evidence appears in
`branch_mapping_provenance.json`. This verifies the structural zero-based
mapping, but does not establish that each synthetic chemical family is
specialized to its named dataset.

## Conclusions and unresolved blockers

- **Atom count at fixed stoichiometry:** changes the input and, as reported in
  the series, may materially change the full routing distribution.
- **Softmax saturation at realistic sizes:** underflow is measurable; its
  prevalence in the training regime cannot be decided without archived atom
  counts.
- **Chemistry after count control:** the 100-atom permutation and matched-pair
  results are the strongest available synthetic evidence; still no real
  composition validation.
- **Substitutions:** path-level responses are measured; abruptness is defined
  by adjacent JS/TV, not physical transitions.
- **Consistent branch–chemistry associations:** not established for real
  datasets because synthetic family labels are not source-dataset labels.
- **Manuscript figure:** no figure is yet supported as evidence for real
  chemical specialization. A clearly labeled full-weight synthetic heatmap
  could serve only as a model-behavior illustration.
- **Remaining blockers:** archived structures and training composition
  distributions; three-record encoding audit; evidence that tested count sizes
  overlap the training range; real-record matched composition analysis.

Weights throughout are the MLP's learned preferences among source-specific
prediction heads. They are not indicators of physical fidelity, predictive
uncertainty, dataset-membership probabilities, or branch activation.
"""
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required output: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = ROBUSTNESS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if first_round._sha256(MLP_PATH) != "c984b338ee947534bbdedf1fc4dd2aea92bb846e3302638d6719b1e3f5ca7ca8":
        raise ValueError("MLP checkpoint hash differs from the first-round manifest")
    if first_round._sha256(BACKBONE_PATH) != "8532b03296a8ab4cfc3b37beb6d57b2091a50807b155fac93cb1882253d32867":
        raise ValueError("HydraGNN checkpoint hash differs from the first-round manifest")
    model, model_info = first_round._load_mlp(MLP_PATH)
    if (model_info["input_dim"], model_info["num_branches"], model_info["parameter_dtype"]) != (
        118,
        16,
        "torch.float64",
    ):
        raise ValueError(f"Unexpected MLP architecture/provenance: {model_info}")

    checkpoint_manifest = json.loads((ANALYSIS_DIR / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoint_manifest["rechecked_for_robustness"] = True
    checkpoint_manifest["architecture_discrepancy"] = None
    (ROBUSTNESS_DIR / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
    )
    branch_provenance = _branch_mapping_provenance()
    (ROBUSTNESS_DIR / "branch_mapping_provenance.json").write_text(
        json.dumps(branch_provenance, indent=2) + "\n", encoding="utf-8"
    )

    first_records, first_weights = _load_first_round()
    size_records = _size_series()
    size_matrix = np.vstack([record["vector"] for record in size_records])
    logits64_size, logp64_size, weights64_size, raw64_size, entropy64_size = _evaluate(
        model, size_matrix, torch.float64
    )
    if not np.isfinite(logits64_size).all():
        raise ValueError("Size-series logits contain nonfinite values")
    if not np.allclose(weights64_size, raw64_size, rtol=0, atol=2e-15):
        raise ValueError("Size-series torch.softmax and exp(log_softmax) mismatch")
    size_rows = _size_sensitivity(size_records, weights64_size, entropy64_size)
    _write_rows(ROBUSTNESS_DIR / "size_sensitivity.csv", size_rows)

    diagnostics_rows, diagnostics = _logit_diagnostics(model, first_records, size_records)
    _write_rows(ROBUSTNESS_DIR / "logit_diagnostics.csv", diagnostics_rows)

    substitution_records = _path_compositions()
    substitution_matrix = np.vstack([record["vector"] for record in substitution_records])
    _, _, substitution_weights, substitution_raw, _ = _evaluate(
        model, substitution_matrix, torch.float64
    )
    if not np.isfinite(substitution_weights).all() or not np.allclose(
        substitution_weights, substitution_raw, rtol=0, atol=2e-15
    ):
        raise ValueError("Substitution path softmax validation failed")
    substitution_rows, substitution_summary = _substitution_rows(
        substitution_records, substitution_weights
    )
    _write_rows(ROBUSTNESS_DIR / "substitution_paths.csv", substitution_rows)

    pair_rows, matched_summary = _pair_records(
        first_records, first_weights, size_records, weights64_size
    )
    _write_rows(ROBUSTNESS_DIR / "matched_similarity.csv", pair_rows)
    vectors = np.vstack([record["vector"] for record in first_records])
    permutation = _permutation_association(vectors, first_weights)
    permutation_json = {
        key: value
        for key, value in permutation.items()
        if key != "null_values"
    }

    real_audit = _real_composition_audit()
    (ROBUSTNESS_DIR / "real_composition_audit.json").write_text(
        json.dumps(real_audit, indent=2) + "\n", encoding="utf-8"
    )

    _figure_size_sensitivity(size_rows, figures_dir)
    _figure_logit_diagnostics(diagnostics_rows, figures_dir)
    _figure_substitution(substitution_rows, figures_dir)
    _figure_similarity(pair_rows, permutation, figures_dir)

    size_summary = _size_summary(size_rows)
    summary = {
        "scope": "second-round model robustness; synthetic inputs only; no training-domain classification",
        "seed": SEED,
        "first_round_composition_count": len(first_records),
        "size_series_composition_count": len(size_records),
        "substitution_path_point_count": len(substitution_records),
        "checkpoint_manifest": checkpoint_manifest,
        "logit_diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if key not in {"matrix", "logits64", "weights64", "weights32", "entropy64"}
        },
        "size_sensitivity": size_summary,
        "substitution_paths": substitution_summary,
        "similarity_controls": {
            "matched": matched_summary,
            "permutation": permutation_json,
            "matching_design": "pair atomic-fraction L1 bins of width 0.1 on 100-atom first-round synthetic compositions",
        },
        "real_composition_audit": real_audit,
        "branch_mapping_status": branch_provenance["status"],
        "domain_proximity": "not supported; no accessible reference training compositions",
    }
    (ROBUSTNESS_DIR / "summary_statistics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _report(ROBUSTNESS_DIR, summary, size_summary, substitution_summary)
    print(
        f"Completed robustness analysis: {len(first_records)} original probes, "
        f"{len(size_records)} size probes, {len(substitution_records)} path points; "
        f"association rho={permutation_json['observed_spearman_rho']:.4f}, "
        f"permutation p={permutation_json['two_sided_permutation_p']:.4g}."
    )


if __name__ == "__main__":
    run()
