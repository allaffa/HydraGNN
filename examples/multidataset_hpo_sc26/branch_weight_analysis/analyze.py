#!/usr/bin/env python3
"""Run a synthetic, composition-only probe of the trained BranchWeightMLP.

This script deliberately does not load or run the HydraGNN backbone. The
generated ASE structures are composition carriers, not relaxed or validated
materials. No training-domain labels are inferred because archived training
compositions are not available in this checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from scipy.stats import spearmanr
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
SC26_DIR = SCRIPT_DIR.parent
REPO_ROOT = SC26_DIR.parents[1]
DEFAULT_MLP = SC26_DIR / "mlp_branch_weights.pt"
DEFAULT_BACKBONE = (
    SC26_DIR
    / "multidataset_hpo-BEST6-fp64"
    / "multidataset_hpo-BEST6-fp64_epoch_97.pk"
)
TOTAL_ATOMS = 100
N_PER_FAMILY = 15
SEED = 260507

BRANCH_NAMES = (
    "Alexandria",
    "ANI1x",
    "MPTrj",
    "OC2020",
    "OC2022",
    "OC25",
    "ODAC23",
    "OMat24",
    "OMol25",
    "OMol25-neutral",
    "OMol25-non-neutral",
    "OPoly2026",
    "Nabla2DFT",
    "QCML",
    "QM7X",
    "transition1x",
)

FAMILY_ELEMENTS = {
    "synthetic C-H motifs": ("C", "H"),
    "synthetic C-H-N-O motifs": ("C", "H", "N", "O"),
    "synthetic C-H-N-O-S-P motifs": ("C", "H", "N", "O", "S", "P"),
    "synthetic transition-metal oxides": ("Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "O"),
    "synthetic mixed-metal oxides": ("Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Nb", "Mo", "O"),
    "synthetic silicate/oxide motifs": ("Li", "Na", "K", "Mg", "Ca", "Al", "Si", "O"),
    "synthetic transition-metal chalcogenides": ("Ti", "Nb", "Mo", "W", "S", "Se", "Te"),
    "synthetic binary refractory alloys": ("V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re"),
    "synthetic ternary refractory alloys": ("Ti", "V", "Zr", "Nb", "Mo", "Hf", "Ta", "W"),
    "synthetic multicomponent alloys": ("Fe", "Co", "Ni", "Cr", "Mn", "Al", "Mo", "Nb"),
    "synthetic semiconductor motifs": ("Si", "Ge", "Sn", "P", "As", "Sb", "S", "Se", "Te"),
    "synthetic ionic/halide motifs": ("Li", "Na", "K", "Mg", "Ca", "Al", "F", "Cl", "Br", "I", "O"),
    "synthetic rare-earth oxides": ("Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd", "O"),
}

BRANCH_COLORS = plt.get_cmap("tab20").colors[: len(BRANCH_NAMES)]
BRANCH_LABELS = [f"branch-{i} {name}" for i, name in enumerate(BRANCH_NAMES)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_composition(family: str, rng: np.random.Generator) -> dict[str, int]:
    if family == "synthetic C-H motifs":
        elements = ["C", "H"]
        weights = rng.dirichlet([3.0, 7.0])
    elif family == "synthetic C-H-N-O motifs":
        elements = ["C", "H", "N", "O"]
        weights = rng.dirichlet([3.0, 4.0, 1.0, 2.0])
    elif family == "synthetic C-H-N-O-S-P motifs":
        elements = ["C", "H", "N", "O", "S", "P"]
        weights = rng.dirichlet([3.0, 4.0, 1.0, 2.0, 0.8, 0.5])
    elif family == "synthetic transition-metal oxides":
        metal = str(rng.choice(["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]))
        elements = [metal, "O"]
        weights = rng.dirichlet([2.0, 3.0])
    elif family == "synthetic mixed-metal oxides":
        metals = rng.choice(
            ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Nb", "Mo"],
            size=2,
            replace=False,
        ).tolist()
        elements = [str(metals[0]), str(metals[1]), "O"]
        weights = rng.dirichlet([1.5, 1.5, 4.0])
    elif family == "synthetic silicate/oxide motifs":
        cations = rng.choice(["Li", "Na", "K", "Mg", "Ca", "Al"], size=2, replace=False)
        elements = [str(cations[0]), str(cations[1]), "Si", "O"]
        weights = rng.dirichlet([1.0, 1.0, 1.5, 5.5])
    elif family == "synthetic transition-metal chalcogenides":
        metals = rng.choice(["Ti", "Nb", "Mo", "W"], size=1 + int(rng.integers(0, 2)), replace=False)
        chalcogen = str(rng.choice(["S", "Se", "Te"]))
        elements = [str(element) for element in metals] + [chalcogen]
        weights = rng.dirichlet([1.5] * len(metals) + [3.0])
    elif family == "synthetic binary refractory alloys":
        elements = rng.choice(["V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re"], size=2, replace=False).tolist()
        weights = rng.dirichlet([2.0, 2.0])
    elif family == "synthetic ternary refractory alloys":
        elements = rng.choice(["Ti", "V", "Zr", "Nb", "Mo", "Hf", "Ta", "W"], size=3, replace=False).tolist()
        weights = rng.dirichlet([2.0, 2.0, 2.0])
    elif family == "synthetic multicomponent alloys":
        n_elements = int(rng.integers(4, 7))
        elements = rng.choice(
            ["Fe", "Co", "Ni", "Cr", "Mn", "Al", "Mo", "Nb"],
            size=n_elements,
            replace=False,
        ).tolist()
        weights = rng.dirichlet(np.full(n_elements, 2.0))
    elif family == "synthetic semiconductor motifs":
        n_elements = int(rng.integers(2, 5))
        elements = rng.choice(
            ["Si", "Ge", "Sn", "P", "As", "Sb", "S", "Se", "Te"],
            size=n_elements,
            replace=False,
        ).tolist()
        weights = rng.dirichlet(np.full(n_elements, 2.0))
    elif family == "synthetic ionic/halide motifs":
        cations = rng.choice(["Li", "Na", "K", "Mg", "Ca", "Al"], size=1 + int(rng.integers(0, 2)), replace=False)
        anions = rng.choice(["F", "Cl", "Br", "I", "O"], size=1 + int(rng.integers(0, 2)), replace=False)
        elements = [str(element) for element in cations] + [str(element) for element in anions]
        weights = rng.dirichlet(np.full(len(elements), 2.0))
    elif family == "synthetic rare-earth oxides":
        metals = rng.choice(["Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Gd"], size=1 + int(rng.integers(0, 2)), replace=False)
        elements = [str(element) for element in metals] + ["O"]
        weights = rng.dirichlet([1.5] * len(metals) + [4.0])
    else:
        raise ValueError(f"Unknown synthetic family: {family}")

    if len(set(elements)) != len(elements):
        raise ValueError(f"Composition generator produced duplicate species: {elements}")
    raw_counts = np.ones(len(elements), dtype=np.int64)
    raw_counts += rng.multinomial(TOTAL_ATOMS - len(elements), weights)
    return {element: int(count) for element, count in zip(elements, raw_counts)}


def _make_atoms(
    counts: dict[str, int],
    rng: np.random.Generator,
    arrangement: int = 0,
) -> Atoms:
    numbers = np.array(
        [atomic_numbers[element] for element, count in counts.items() for _ in range(count)],
        dtype=np.int64,
    )
    rng.shuffle(numbers)
    if arrangement:
        numbers = np.roll(numbers, arrangement)
    grid = np.array(list(itertools.product(range(5), range(5), range(4))), dtype=np.float64)
    positions = grid * 2.5
    return Atoms(
        numbers=numbers,
        positions=positions,
        cell=[12.5, 12.5, 10.0],
        pbc=True,
    )


def _composition_vector(atoms: Atoms) -> np.ndarray:
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    if numbers.size == 0 or np.any((numbers < 1) | (numbers > 118)):
        raise ValueError("Generated structure has an empty or unsupported element list")
    # Match the dataset adapters: histogram bins are atomic numbers 1 through 118.
    vector = np.bincount(numbers, minlength=119)[1:119].astype(np.float64)
    if vector.shape != (118,) or vector.sum() != len(atoms):
        raise ValueError("Generated composition does not match the 118-count encoding")
    return vector


def _make_suite() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    samples: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for family in FAMILY_ELEMENTS:
        for index in range(N_PER_FAMILY):
            for attempt in range(1000):
                counts = _family_composition(family, rng)
                atoms = _make_atoms(counts, rng)
                vector = _composition_vector(atoms)
                key = tuple(vector.astype(np.int64))
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise RuntimeError(f"Could not generate a unique composition for {family}")
            fractions = {
                chemical_symbols[i + 1]: float(value / vector.sum())
                for i, value in enumerate(vector)
                if value > 0
            }
            sample_id = f"SYN-{len(samples) + 1:03d}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "family": family,
                    "formula": atoms.get_chemical_formula(mode="hill"),
                    "source_dataset": "synthetic_generated",
                    "source_record_id": f"{sample_id}-composition-carrier",
                    "domain_classification": "unknown; training coverage unavailable",
                    "training_element_set_match": "unknown",
                    "atom_count": int(len(atoms)),
                    "atomic_fractions": fractions,
                    "vector": vector,
                    "atoms": atoms,
                }
            )
    expected_count = N_PER_FAMILY * len(FAMILY_ELEMENTS)
    if len(samples) != expected_count:
        raise AssertionError(f"Expected {expected_count} unique compositions, got {len(samples)}")
    return samples


def _load_mlp(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"MLP checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "mlp_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint dictionary with 'mlp_state_dict'")
    state = checkpoint["mlp_state_dict"]
    weight_keys = sorted(
        (key for key in state if key.startswith("net.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if not weight_keys:
        raise ValueError("MLP checkpoint has no net.*.weight tensors")
    dims = [int(state[key].shape[0]) for key in weight_keys]
    input_dim = int(state[weight_keys[0]].shape[1])
    layers: list[nn.Module] = []
    in_dim = input_dim
    for index, key in enumerate(weight_keys):
        output_dim = int(state[key].shape[0])
        layers.append(nn.Linear(in_dim, output_dim))
        if index < len(weight_keys) - 1:
            layers.append(nn.ReLU())
        in_dim = output_dim
    model = nn.Sequential(*layers)
    model_state = {
        key[len("net.") :]: value
        for key, value in state.items()
        if key.startswith("net.")
    }
    if len(model_state) != len(state):
        raise ValueError("MLP checkpoint contains unexpected keys outside net.*")
    parameter_dtype = next(
        value.dtype for value in state.values() if isinstance(value, torch.Tensor)
    )
    model.to(dtype=parameter_dtype)
    model.load_state_dict(model_state, strict=True)
    dtypes = sorted({str(value.dtype) for value in state.values() if isinstance(value, torch.Tensor)})
    if len(dtypes) != 1:
        raise ValueError(f"Expected one parameter dtype, got {dtypes}")
    model.eval()
    return model, {
        "input_dim": input_dim,
        "hidden_dims": dims[:-1],
        "num_branches": dims[-1],
        "parameter_dtype": dtypes[0],
        "epoch": int(checkpoint.get("epoch", -1)),
        "torch_dtype": parameter_dtype,
    }


def _predict(model: nn.Module, matrix: np.ndarray) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    inputs = torch.as_tensor(matrix, dtype=dtype, device="cpu")
    with torch.inference_mode():
        logits = model(inputs)
        weights = torch.softmax(logits, dim=-1)
    return weights.detach().cpu().numpy()


def _metrics(weights: np.ndarray) -> dict[str, Any]:
    order = np.argsort(weights)[::-1]
    cumulative = np.cumsum(weights[order])
    positive = weights > 0
    entropy = float(-np.sum(weights[positive] * np.log(weights[positive])))
    return {
        "dominant": int(order[0]),
        "top3_ids": order[:3].astype(int).tolist(),
        "top3_sum": float(weights[order[:3]].sum()),
        "entropy": entropy,
        "effective_branches": float(np.exp(entropy)),
        "support90": int(np.searchsorted(cumulative, 0.90, side="left") + 1),
        "support95": int(np.searchsorted(cumulative, 0.95, side="left") + 1),
        "max_weight": float(weights[order[0]]),
    }


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = (first + second) / 2.0
    mask1 = first > 0
    mask2 = second > 0
    ratio1 = np.divide(first, midpoint, out=np.ones_like(first), where=midpoint > 0)
    ratio2 = np.divide(second, midpoint, out=np.ones_like(second), where=midpoint > 0)
    # A positive subnormal can underflow when halved; its limiting ratio is 2.
    ratio1[mask1 & (midpoint == 0)] = 2.0
    ratio2[mask2 & (midpoint == 0)] = 2.0
    kl1 = np.sum(first[mask1] * np.log2(ratio1[mask1]))
    kl2 = np.sum(second[mask2] * np.log2(ratio2[mask2]))
    return float(max(0.0, (kl1 + kl2) / 2.0))


def _branch_id(branch_index: int) -> str:
    return f"branch-{branch_index}:{BRANCH_NAMES[branch_index]}"


def _csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_paths(model: nn.Module) -> list[dict[str, Any]]:
    paths = [
        (
            "Nb-Ta binary alloy probe",
            {"Nb": 70, "Ta": 30},
            {"Nb": 30, "Ta": 70},
        ),
        (
            "Ti-Fe oxide substitution probe",
            {"Ti": 25, "O": 75},
            {"Fe": 25, "O": 75},
        ),
        (
            "Nb-Ta-V ternary alloy probe",
            {"Nb": 60, "Ta": 25, "V": 15},
            {"Nb": 20, "Ta": 30, "V": 50},
        ),
    ]
    output: list[dict[str, Any]] = []
    for name, start, end in paths:
        matrix = []
        fractions = []
        for t in np.linspace(0.0, 1.0, 21):
            vector = np.zeros(118, dtype=np.float64)
            for element in set(start) | set(end):
                atomic_number = atomic_numbers[element]
                vector[atomic_number - 1] = (1.0 - t) * start.get(element, 0) + t * end.get(element, 0)
            matrix.append(vector)
            fractions.append(
                {
                    chemical_symbols[i + 1]: float(value / TOTAL_ATOMS)
                    for i, value in enumerate(vector)
                    if value > 0
                }
            )
        weights_matrix = _predict(model, np.vstack(matrix))
        for index, (t, vector, fraction, weights) in enumerate(
            zip(np.linspace(0.0, 1.0, 21), matrix, fractions, weights_matrix)
        ):
            metric = _metrics(weights)
            output.append(
                {
                    "path": name,
                    "step": index,
                    "t": float(t),
                    "interpolation_type": "composition-space probe; not a discrete structure",
                    "atomic_fractions_json": json.dumps(fraction, sort_keys=True),
                    "composition_counts_json": json.dumps(
                        {
                            chemical_symbols[i + 1]: float(value)
                            for i, value in enumerate(vector)
                            if value > 0
                        },
                        sort_keys=True,
                    ),
                    **{f"branch_{i:02d}_{BRANCH_NAMES[i]}": float(value) for i, value in enumerate(weights)},
                    "dominant_branch": _branch_id(metric["dominant"]),
                    "top3_branch_ids": ";".join(_branch_id(i) for i in metric["top3_ids"]),
                    "top3_cumulative_weight": metric["top3_sum"],
                    "weight_entropy": metric["entropy"],
                    "effective_branches": metric["effective_branches"],
                }
            )
    return output


def _pairwise_rows(samples: list[dict[str, Any]], weights: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first_index, second_index in itertools.combinations(range(len(samples)), 2):
        first = samples[first_index]
        second = samples[second_index]
        kind = "chemically_related_synthetic_family" if first["family"] == second["family"] else "chemically_distinct_synthetic_families"
        fractions_a = first["vector"] / first["vector"].sum()
        fractions_b = second["vector"] / second["vector"].sum()
        wa = weights[first_index]
        wb = weights[second_index]
        rows.append(
            {
                "pair_id": f"{first['sample_id']}__{second['sample_id']}",
                "pair_kind": kind,
                "sample_a": first["sample_id"],
                "sample_b": second["sample_id"],
                "family_a": first["family"],
                "family_b": second["family"],
                "composition_l1_distance": float(np.abs(fractions_a - fractions_b).sum()),
                "jensen_shannon_divergence_bits": _js_divergence(wa, wb),
                "total_variation_distance": float(0.5 * np.abs(wa - wb).sum()),
                "dominant_branch_agreement": bool(np.argmax(wa) == np.argmax(wb)),
            }
        )

    return rows


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    pair_kind: str,
    rng: np.random.Generator,
) -> dict[str, float] | None:
    frame = pd.DataFrame(rows)
    frame = frame[frame["pair_kind"] == pair_kind]
    if frame.empty:
        return None
    cluster_values: dict[str, list[float]] = {}
    for row in frame.to_dict("records"):
        cluster_values.setdefault(row["family_a"], []).append(row["jensen_shannon_divergence_bits"])
        if row["family_b"] != row["family_a"]:
            cluster_values.setdefault(row["family_b"], []).append(row["jensen_shannon_divergence_bits"])
    means = np.array([np.mean(values) for values in cluster_values.values()], dtype=np.float64)
    if len(means) < 2:
        return None
    samples = rng.choice(means, size=(2000, len(means)), replace=True).mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_unit": "family-cluster means; descriptive for this generated suite",
        "cluster_count": int(len(means)),
    }


def _save_figure(path_without_suffix: Path) -> None:
    plt.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()


def _plot_heatmap(
    samples: list[dict[str, Any]],
    weights: np.ndarray,
    output_dir: Path,
    n_representatives: int | None,
) -> None:
    ordered = sorted(range(len(samples)), key=lambda index: (samples[index]["family"], samples[index]["formula"], samples[index]["sample_id"]))
    if n_representatives is not None:
        chosen: list[int] = []
        for family in FAMILY_ELEMENTS:
            chosen.extend([index for index in ordered if samples[index]["family"] == family][:2])
            if len(chosen) >= n_representatives:
                break
        ordered = chosen[:n_representatives]
    data = weights[ordered]
    labels = [
        f"{samples[index]['formula']} [S]"
        for index in ordered
    ]
    height = max(6.0, 0.26 * len(ordered))
    fig, axis = plt.subplots(figsize=(13, height))
    image = axis.imshow(data, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(np.arange(len(BRANCH_NAMES)), BRANCH_LABELS, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axis.set_title(
        "Synthetic composition probes: BranchWeightMLP routing\n"
        "Rows grouped by generated composition motif; [S] = synthetic, training domain unknown"
    )
    axis.set_xlabel("Dataset-specific branch")
    axis.set_ylabel("Generated formula (composition carrier)")
    for index in range(1, len(ordered)):
        if samples[ordered[index]]["family"] != samples[ordered[index - 1]]["family"]:
            axis.axhline(index - 0.5, color="white", linewidth=1.2)
    fig.colorbar(image, ax=axis, label="Softmax weight", fraction=0.025, pad=0.02)
    fig.tight_layout()
    _save_figure(output_dir / ("fig1_branch_weights_heatmap" if n_representatives else "fig1_branch_weights_heatmap_full"))


def _plot_paths(paths: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    frame = pd.DataFrame(paths)
    branch_columns = [f"branch_{i:02d}_{BRANCH_NAMES[i]}" for i in range(len(BRANCH_NAMES))]
    path_info: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, (path_name, group) in zip(axes, frame.groupby("path", sort=False)):
        values = group[branch_columns].to_numpy(dtype=np.float64)
        relevant = np.argsort(values.mean(axis=0))[::-1][:5]
        x = group["t"].to_numpy(dtype=np.float64)
        for branch_index in relevant:
            axis.plot(
                x,
                values[:, branch_index],
                color=BRANCH_COLORS[branch_index],
                linewidth=2,
                label=BRANCH_LABELS[branch_index],
            )
        other = np.clip(1.0 - values[:, relevant].sum(axis=1), 0.0, 1.0)
        axis.plot(x, other, color="0.35", linestyle="--", linewidth=1.4, label="Other (remaining 11)")
        dominant = group["dominant_branch"].tolist()
        path_info[path_name] = {
            "top_five_branches": [_branch_id(int(index)) for index in relevant],
            "dominant_start": dominant[0],
            "dominant_end": dominant[-1],
            "dominant_switch_count": int(sum(a != b for a, b in zip(dominant, dominant[1:]))),
            "maximum_effective_branches": float(group["effective_branches"].max()),
        }
        axis.set_title(path_name)
        axis.set_xlabel("Interpolation fraction t")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Branch weight")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle("Composition-space interpolation probes (100-count vectors; not discrete structures)")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    _save_figure(output_dir / "fig2_composition_paths")
    return path_info


def _plot_similarity(rows: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    groups = [
        ("same_composition_different_generated_arrangement", "Same composition\n(different arrangement)"),
        ("chemically_related_synthetic_family", "Related generated\nmotif family"),
        ("chemically_distinct_synthetic_families", "Different generated\nmotif families"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    values = [frame.loc[frame["pair_kind"] == key, "jensen_shannon_divergence_bits"].to_numpy() for key, _ in groups]
    axes[0].boxplot(values, showfliers=False, widths=0.6)
    axes[0].set_xticks(range(1, len(groups) + 1), [f"{label}\n(n={len(value)})" for (_, label), value in zip(groups, values)])
    axes[0].set_ylabel("Jensen-Shannon divergence (bits)")
    axes[0].set_title("Routing-distance distributions")
    axes[0].grid(axis="y", alpha=0.2)
    plotted = frame[frame["pair_kind"] != "same_composition_different_generated_arrangement"]
    if len(plotted) > 2500:
        plotted = plotted.sample(2500, random_state=SEED)
    colors = np.where(
        plotted["pair_kind"].eq("chemically_related_synthetic_family"),
        "#4c78a8",
        "#f58518",
    )
    axes[1].scatter(
        plotted["composition_l1_distance"],
        plotted["jensen_shannon_divergence_bits"],
        c=colors,
        alpha=0.3,
        s=9,
        linewidths=0,
    )
    axes[1].set_xlabel("L1 distance between normalized atomic fractions")
    axes[1].set_ylabel("Jensen-Shannon divergence (bits)")
    axes[1].set_title("Composition distance vs. routing distance")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Synthetic-only comparison; pairs share samples and are not independent observations")
    fig.tight_layout()
    _save_figure(output_dir / "fig3_routing_similarity")


def _plot_concentration(samples: list[dict[str, Any]], metrics: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "family": sample["family"],
                "effective_branches": metric["effective_branches"],
                "max_weight": metric["max_weight"],
                "top3_sum": metric["top3_sum"],
            }
            for sample, metric in zip(samples, metrics)
        ]
    )
    families = list(FAMILY_ELEMENTS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    for axis, column, label in zip(
        axes,
        ("effective_branches", "max_weight", "top3_sum"),
        ("Effective number of branches", "Maximum branch weight", "Top-three cumulative weight"),
    ):
        data = [frame.loc[frame["family"] == family, column].to_numpy() for family in families]
        axis.boxplot(data, showfliers=False)
        axis.set_xticks(range(1, len(families) + 1), [family.replace("synthetic ", "") for family in families], rotation=70, ha="right", fontsize=7)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Weight concentration by generated composition motif; training-domain classes unavailable")
    fig.tight_layout()
    _save_figure(output_dir / "fig4_branch_concentration")


def _plot_family_means(samples: list[dict[str, Any]], weights: np.ndarray, output_dir: Path) -> None:
    family_means = np.vstack(
        [weights[[sample["family"] == family for sample in samples]].mean(axis=0) for family in FAMILY_ELEMENTS]
    )
    fig, axis = plt.subplots(figsize=(14, 6))
    bottoms = np.zeros(len(FAMILY_ELEMENTS))
    for index, name in enumerate(BRANCH_NAMES):
        axis.bar(
            np.arange(len(FAMILY_ELEMENTS)),
            family_means[:, index],
            bottom=bottoms,
            color=BRANCH_COLORS[index],
            label=f"branch-{index} {name}",
        )
        bottoms += family_means[:, index]
    axis.set_xticks(np.arange(len(FAMILY_ELEMENTS)), [family.replace("synthetic ", "") for family in FAMILY_ELEMENTS], rotation=45, ha="right")
    axis.set_ylabel("Mean softmax weight (15 compositions per motif family)")
    axis.set_title("Branch utilization across balanced generated motif families")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    _save_figure(output_dir / "fig5_branch_family_utilization")


def _write_report(
    output_dir: Path,
    model_info: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    top_family_lines = []
    for family, stats in summary["family_statistics"].items():
        top_family_lines.append(
            f"- **{family}:** most frequent dominant branch {stats['most_frequent_dominant_branch']} "
            f"({stats['most_frequent_dominant_count']}/15); mean weight on that branch "
            f"{stats['mean_weight_on_most_frequent_dominant']:.4f}; mean effective support "
            f"{stats['mean_effective_branches']:.2f}."
        )
    related = summary["pairwise_statistics"]["related_js"]
    distinct = summary["pairwise_statistics"]["distinct_js"]
    same = summary["pairwise_statistics"]["same_composition_max_js"]
    path_lines = []
    for name, info in summary["composition_paths"].items():
        path_lines.append(
            f"- **{name}:** dominant branch {info['dominant_start']} at t=0 and "
            f"{info['dominant_end']} at t=1; {info['dominant_switch_count']} changes between adjacent "
            "sampled points; the five plotted branches are "
            + ", ".join(info["top_five_branches"])
            + "."
        )
    report = f"""# BranchWeightMLP composition-routing probe

## Scope and limitation

This is a reproducible **synthetic composition-probe analysis**, not the requested
production-data analysis. The user confirmed that the supplied MLP pairs with the
local HydraGNN epoch-97 checkpoint and asked to generate structures. There are no
archived validation/held-out structures or training composition distributions in
the local checkout. Consequently, the required three-archive encoding comparison,
empirical in-domain/novel/distant labels, dataset-specialization claims, and
manuscript-ready generalizations cannot be established here. All {summary['sample_count']} generated
100-atom ASE structures are composition carriers with deterministic simple-cubic
positions; no relaxation, DFT, HydraGNN backbone inference, or training was run.
They should not be mistaken for stable molecules, crystals, or sampled training
records.

## Model and checkpoint provenance

- MLP checkpoint: `../mlp_branch_weights.pt`
- MLP SHA-256: `{summary['checkpoint_manifest']['mlp_sha256']}`
- MLP checkpoint epoch field: {model_info['epoch']} (this is the checkpoint's saved
  epoch, not independent evidence of best validation performance).
- HydraGNN checkpoint: `../multidataset_hpo-BEST6-fp64/multidataset_hpo-BEST6-fp64_epoch_97.pk`
- HydraGNN SHA-256: `{summary['checkpoint_manifest']['backbone_sha256']}`
- Git commit: `{summary['checkpoint_manifest']['git_commit']}`
- Pairing: user-confirmed; the MLP file itself contains no HydraGNN checkpoint
  identifier.
- MLP dimensions: {model_info['input_dim']} -> {model_info['hidden_dims'][0]} ->
  {model_info['hidden_dims'][1]} -> {model_info['num_branches']}; parameter dtype
  {model_info['parameter_dtype']}; softmax output.
- Branch order is recorded in `branch_mapping.csv` from the local SC26 launch
  metadata: {", ".join(BRANCH_NAMES)}.

## Composition encoding

The adapters construct `chemical_composition` as an 118-bin histogram of atomic
numbers 1 through 118, then cast it to float32. Feature index `Z-1` is the count
of element Z; absent elements have zero; counts are not normalized. Thus the
vector depends on the number of atoms in a structure. The inference reshape
changes layout to `[graphs, 118]` but does not normalize or otherwise transform
the counts before MLP inference. The generated inputs use the same integer-count
histogram over ASE atomic numbers, then the original checkpoint dtype (float64).

Archived-field validation on three saved structures was **not performed**:
there are no local archived datasets. `composition_encoding_audit.json` records
this explicitly; generated vectors are an implementation exercise, not an
independent archive audit.

## Evaluation construction and validation

- {summary['sample_count']} unique generated compositions, {summary['family_count']}
  balanced synthetic motif families, {N_PER_FAMILY} samples per family.
- Every probe has {TOTAL_ATOMS} atoms so the raw-count feature scale is held fixed
  across this synthetic suite.
- All records use source `synthetic_generated`; training combination coverage and
  domain class are `unknown`.
- All pairwise composition distances use L1 distance between atomic-fraction
  vectors. Routing total variation is one half the L1 distance between weight
  vectors; Jensen-Shannon divergence is computed with base-2 logarithms and is
  reported in bits. “Related” means members of the same generated template
  family, not a measured chemical-similarity label.
- A five-composition pilot passed shape, finite/nonnegative weight, normalization,
  and repeatability checks. Maximum softmax-sum error was
  {summary['pilot']['max_sum_error']:.3e}; maximum repeat difference was
  {summary['pilot']['max_repeat_difference']:.3e}.
- A composition-only arrangement check covered {summary['same_composition_structure_pairs']}
  same-composition pairs. Maximum observed JS divergence was {same:.3e}. This is
  a check of the composition-only implementation, not geometric transferability.
- Inference loaded only BranchWeightMLP and did not invoke the HydraGNN backbone.

## Quantitative routing results

The primary observation is the complete 16-dimensional softmax vector for each
input, saved in `branch_weights.csv` and visualized in the heatmaps. Read these
values as learned preferences among source-specific prediction heads. The
dominant-branch, top-three, entropy, and cumulative-support summaries are
secondary descriptions of that distribution; no branch is labeled activated or
inactive using a hard per-composition threshold.

- Overall mean effective support: {summary['global_concentration']['mean_effective_branches']:.3f}
  of 16 branches (range {summary['global_concentration']['min_effective_branches']:.3f} to
  {summary['global_concentration']['max_effective_branches']:.3f}).
- Mean maximum branch weight: {summary['global_concentration']['mean_max_weight']:.4f};
  mean top-three cumulative weight: {summary['global_concentration']['mean_top3_sum']:.4f}.
- Mean support size: {summary['global_concentration']['mean_support90']:.2f} branches
  for 90% cumulative weight and {summary['global_concentration']['mean_support95']:.2f}
  branches for 95%.
- {summary['weight_numerics']['exact_zero_count']} of
  {summary['weight_numerics']['total_weights']} computed float64 weights were
  represented as exact zero by softmax underflow. These are finite-precision
  zeros, not branch skipping or a per-composition activation rule; mathematical
  softmax weights are positive.
- Related-family pairs: n={related['pair_count']}, mean JS divergence
  {related['mean']:.4f} bits, median {related['median']:.4f}; family-cluster
  descriptive 95% bootstrap interval for the mean
  [{related['ci95_low']:.4f}, {related['ci95_high']:.4f}]; mean total-variation
  distance {related['mean_total_variation']:.4f}, dominant-branch agreement
  {related['dominant_branch_agreement']:.1%}.
- Different-family pairs: n={distinct['pair_count']}, mean JS divergence
  {distinct['mean']:.4f} bits, median {distinct['median']:.4f}; family-cluster
  descriptive 95% bootstrap interval for the mean
  [{distinct['ci95_low']:.4f}, {distinct['ci95_high']:.4f}]; mean
  total-variation distance {distinct['mean_total_variation']:.4f},
  dominant-branch agreement {distinct['dominant_branch_agreement']:.1%}.
- Same-composition/different-arrangement pairs: n={summary['same_composition_structure_pairs']},
  maximum JS divergence {same:.3e} bits.
- Spearman association between normalized-composition L1 distance and routing JS
  divergence over related and distinct composition pairs:
  rho={summary['pairwise_statistics']['spearman_rho']:.4f}. This is descriptive;
  pairs share compositions and the statistic is not an independent-pair
  significance test.
- Confidence intervals resample family-level mean routing distances, not
  individual pair rows; with generated families they are descriptive summaries,
  not population-inference intervals.

### Motif-family dominant branches

{chr(10).join(top_family_lines)}

### Continuous composition-space probes

Each path contains 21 points with a constant 100-count vector scale. Interpolated
fractional counts are composition-space probes, not discrete atomistic structures
or asserted phase transitions.

{chr(10).join(path_lines)}

## Figure interpretation

- **Figure 1:** heatmap of 26 representative generated compositions, ordered by
  synthetic motif family. It visualizes this checkpoint's response only; it is
  not evidence of routing on archived chemical records.
- **Supplementary full heatmap:** all {summary['sample_count']} generated compositions.
- **Figure 2:** five highest-mean branches along each of three fixed-total
  interpolation paths; remaining weight is shown as “Other.”
- **Figure 3:** same-composition, within-motif, and between-motif routing
  distances. Pairwise rows overlap by construction, so boxes are descriptive.
- **Figure 4:** concentration metrics by generated motif family. It does not
  compare empirical domain classes.
- **Figure 5:** family-balanced mean branch weights, with 15 compositions per
  generated motif family.

## Scientific assessment

- **Are similar compositions routed similarly?** In this synthetic suite, the
  measured relationship is the reported Spearman rho and the within-/between-
  family JS summaries above. These generated labels do not establish the result
  for manuscript data.
- **Which branches dominate for each chemical family?** The measured synthetic
  motif-family results are listed above and in `summary_statistics.json`. They
  are not dataset specialization evidence.
- **Does routing remain stable under small composition changes?** Figure 2 and
  `composition_paths.csv` give measured weight changes on three chosen paths.
  They establish behavior only along those constructed paths.
- **Are compositionally novel inputs routed differently from familiar ones?**
  This cannot be answered: training compositions are unavailable, so every
  generated record has unknown domain status.
- **Is weighting concentrated or diffuse?** The synthetic-suite effective
  support, maximum weights, top-three sums, and support sizes are in
  `routing_metrics.csv` and `summary_statistics.json`.
- **Established versus hypothesized:** established here are the checkpoint's
  deterministic softmax outputs for the exact generated count vectors and the
  arithmetic summaries. Dataset specialization, physical confidence, prediction
  accuracy, and behavior on in-domain or out-of-domain production data remain
  untested.

Softmax weights are learned preferences among source-specific prediction heads,
not probabilities that a composition belongs to a dataset, measures of physical
fidelity, uncertainty estimates, or indicators of DFT accuracy. A high or low
entropy must not be interpreted as confidence without independent calibration
evidence.

## Manuscript text and reviewer response (provisional only)

**Section 5.5 draft:** “The BranchWeightMLP maps an 118-component elemental-count
vector to a 16-branch softmax distribution. In a synthetic, fixed-size
composition-probe check, branch weights varied across tested compositions and
along selected interpolation paths; because archived structures and training
composition distributions were unavailable for this check, these results do not
establish dataset specialization or behavior on in-domain and novel materials.”

**Reviewer 2, Comment 5:** “We performed a composition-only checkpoint probe and
report the complete branch-weight distributions, concentration measures, and
composition-path responses. The probe used generated structures and therefore
does not substitute for analysis of the manuscript's archived evaluation data.
We will report claims about chemical specialization and domain coverage only
after validating the exact composition vectors against archived records and
comparing them with the training composition distribution.”

**Figure recommendation:** Do not use the present synthetic-only figures as
manuscript evidence. After rerunning with audited archived records, use the
representative branch-weight heatmap as the likely main-text figure and place
composition-path and pairwise-similarity plots in Supporting Information.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run(output_dir: Path, mlp_path: Path, backbone_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, model_info = _load_mlp(mlp_path)
    if model_info["input_dim"] != 118:
        raise ValueError(f"Checkpoint input dimension is {model_info['input_dim']}, expected 118")
    if model_info["num_branches"] != 16:
        raise ValueError(f"Checkpoint output dimension is {model_info['num_branches']}, expected 16")

    samples = _make_suite()
    matrix = np.vstack([sample["vector"] for sample in samples])
    pilot_matrix = matrix[:5]
    pilot_weights = _predict(model, pilot_matrix)
    repeated_weights = _predict(model, pilot_matrix)
    max_repeat_difference = float(np.max(np.abs(pilot_weights - repeated_weights)))
    max_sum_error = float(np.max(np.abs(pilot_weights.sum(axis=1) - 1.0)))
    min_weight = float(pilot_weights.min())
    if (
        not np.isfinite(pilot_weights).all()
        or min_weight < 0
        or max_sum_error > 1e-12
        or max_repeat_difference > 1e-12
    ):
        raise ValueError(
            "Five-composition pilot failed: "
            f"min={min_weight}, sum_error={max_sum_error}, repeat_diff={max_repeat_difference}"
        )

    weights = _predict(model, matrix)
    if weights.shape != (len(samples), len(BRANCH_NAMES)):
        raise ValueError(f"Unexpected output shape {weights.shape}")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Model produced nonfinite or negative softmax weights")
    max_full_sum_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    if max_full_sum_error > 1e-12:
        raise ValueError(f"Model softmax weights do not sum to one: max error {max_full_sum_error}")
    metrics = [_metrics(row) for row in weights]

    checkpoint_manifest = {
        "mlp_checkpoint": str(mlp_path.relative_to(SC26_DIR)),
        "mlp_sha256": _sha256(mlp_path),
        "mlp_epoch": model_info["epoch"],
        "hydragNN_checkpoint": str(backbone_path.relative_to(SC26_DIR)),
        "backbone_sha256": _sha256(backbone_path),
        "mlp_hydragNN_pairing": "user-confirmed; MLP checkpoint does not embed base-checkpoint identity",
        "git_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "input_dim": model_info["input_dim"],
        "hidden_dims": model_info["hidden_dims"],
        "output_branches": model_info["num_branches"],
        "parameter_dtype": model_info["parameter_dtype"],
        "inference_precision": "float64; checkpoint parameter dtype preserved",
        "backbone_forward_used": False,
    }
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
    )

    encoding_audit = {
        "status": "archive comparison not performed; synthetic-only scope",
        "feature_definition": "118-bin atomic-number count histogram; vector index Z-1 stores count of element Z",
        "source_code_evidence": [
            "examples/ani1_x/train.py: bins=range(1, 118 + 2), histogram cast to float32",
            "examples/transition1x/train.py: same 118-bin histogram construction",
            "examples/multidataset_hpo_sc26/inference_fused.py: reshape to [graphs, features], then direct MLP input",
        ],
        "raw_counts": True,
        "normalized": False,
        "absent_elements_are_zero": True,
        "depends_on_unit_cell_or_structure_atom_count": True,
        "pre_mlp_transform": "layout reshape and dtype/device conversion only; no count normalization",
        "generated_structure_vectors": "computed from ASE atomic numbers using the same histogram bins",
        "archived_structure_checks_required": 3,
        "archived_structure_checks_completed": 0,
        "limitation": "No archived structures with stored chemical_composition are available locally; generated records cannot establish an independent stored-field match.",
    }
    (output_dir / "composition_encoding_audit.json").write_text(
        json.dumps(encoding_audit, indent=2) + "\n", encoding="utf-8"
    )

    mapping_rows = [
        {
            "branch_index": index,
            "branch_id": f"branch-{index}",
            "dataset_name": name,
            "mapping_evidence": "SC26 multi_model_list order; corroborated by run-4284444.log",
        }
        for index, name in enumerate(BRANCH_NAMES)
    ]
    _csv_write(output_dir / "branch_mapping.csv", mapping_rows)

    manifest_rows = []
    weight_rows = []
    metric_rows = []
    for sample, weight, metric in zip(samples, weights, metrics):
        manifest_rows.append(
            {
                "sample_id": sample["sample_id"],
                "chemical_formula": sample["formula"],
                "chemical_family": sample["family"],
                "source_dataset": sample["source_dataset"],
                "source_record_id": sample["source_record_id"],
                "domain_classification": sample["domain_classification"],
                "training_element_set_match": sample["training_element_set_match"],
                "atom_count": sample["atom_count"],
                "atomic_fractions_json": json.dumps(sample["atomic_fractions"], sort_keys=True),
                "generated_structure_note": "100-atom deterministic simple-cubic composition carrier; not relaxed or physically validated",
            }
        )
        row = {
            "sample_id": sample["sample_id"],
            "chemical_formula": sample["formula"],
            "chemical_family": sample["family"],
            "source_dataset": sample["source_dataset"],
            "domain_classification": sample["domain_classification"],
        }
        row.update(
            {f"branch_{index:02d}_{name}": float(value) for index, (name, value) in enumerate(zip(BRANCH_NAMES, weight))}
        )
        row.update(
            {
                "dominant_branch": _branch_id(metric["dominant"]),
                "top3_branch_identifiers": ";".join(_branch_id(index) for index in metric["top3_ids"]),
                "top3_cumulative_weight": metric["top3_sum"],
                "weight_entropy": metric["entropy"],
                "effective_number_of_branches": metric["effective_branches"],
                "support_90_size": metric["support90"],
                "support_95_size": metric["support95"],
            }
        )
        weight_rows.append(row)
        metric_rows.append(
            {
                "sample_id": sample["sample_id"],
                "chemical_family": sample["family"],
                "domain_classification": sample["domain_classification"],
                "dominant_branch": _branch_id(metric["dominant"]),
                "maximum_branch_weight": metric["max_weight"],
                "top3_cumulative_weight": metric["top3_sum"],
                "weight_entropy": metric["entropy"],
                "effective_number_of_branches": metric["effective_branches"],
                "support_90_size": metric["support90"],
                "support_95_size": metric["support95"],
            }
        )
    _csv_write(output_dir / "composition_manifest.csv", manifest_rows)
    _csv_write(output_dir / "branch_weights.csv", weight_rows)
    _csv_write(output_dir / "routing_metrics.csv", metric_rows)

    paths = _make_paths(model)
    _csv_write(output_dir / "composition_paths.csv", paths)
    pair_rows = _pairwise_rows(samples, weights)
    for sample_index, sample in enumerate(samples[:6]):
        counts = {
            chemical_symbols[element_index + 1]: int(count)
            for element_index, count in enumerate(sample["vector"])
            if count > 0
        }
        arrangement_rng = np.random.default_rng(SEED + 100 + sample_index)
        vectors = [sample["vector"]]
        for arrangement in (1, 2):
            vectors.append(_composition_vector(_make_atoms(counts, arrangement_rng, arrangement)))
        arrangement_weights = _predict(model, np.vstack(vectors))
        if not np.allclose(arrangement_weights, arrangement_weights[0], atol=1e-12, rtol=0):
            raise ValueError("Same-composition arrangement check failed")
        for left, right in itertools.combinations(range(len(vectors)), 2):
            js_value = _js_divergence(arrangement_weights[left], arrangement_weights[right])
            pair_rows.append(
                {
                    "pair_id": f"{sample['sample_id']}-arr{left}__{sample['sample_id']}-arr{right}",
                    "pair_kind": "same_composition_different_generated_arrangement",
                    "sample_a": f"{sample['sample_id']}-arr{left}",
                    "sample_b": f"{sample['sample_id']}-arr{right}",
                    "family_a": sample["family"],
                    "family_b": sample["family"],
                    "composition_l1_distance": 0.0,
                    "jensen_shannon_divergence_bits": js_value,
                    "total_variation_distance": float(
                        0.5
                        * np.abs(
                            arrangement_weights[left] - arrangement_weights[right]
                        ).sum()
                    ),
                    "dominant_branch_agreement": bool(
                        np.argmax(arrangement_weights[left])
                        == np.argmax(arrangement_weights[right])
                    ),
                }
            )
    _csv_write(output_dir / "pairwise_similarity.csv", pair_rows)

    # Average branch-weight vectors equally across the 12 balanced motif families.
    family_statistics: dict[str, Any] = {}
    for family in FAMILY_ELEMENTS:
        indices = [index for index, sample in enumerate(samples) if sample["family"] == family]
        family_weights = weights[indices]
        dominant_ids = [metrics[index]["dominant"] for index in indices]
        counts = np.bincount(dominant_ids, minlength=len(BRANCH_NAMES))
        most_frequent = int(np.argmax(counts))
        family_statistics[family] = {
            "sample_count": len(indices),
            "most_frequent_dominant_branch": _branch_id(most_frequent),
            "most_frequent_dominant_count": int(counts[most_frequent]),
            "mean_weight_on_most_frequent_dominant": float(family_weights[:, most_frequent].mean()),
            "mean_effective_branches": float(np.mean([metrics[index]["effective_branches"] for index in indices])),
            "mean_max_weight": float(np.mean([metrics[index]["max_weight"] for index in indices])),
            "mean_top3_sum": float(np.mean([metrics[index]["top3_sum"] for index in indices])),
            "mean_branch_weights": {
                _branch_id(index): float(value)
                for index, value in enumerate(family_weights.mean(axis=0))
            },
        }

    pair_frame = pd.DataFrame(pair_rows)
    related_frame = pair_frame[pair_frame["pair_kind"] == "chemically_related_synthetic_family"]
    distinct_frame = pair_frame[pair_frame["pair_kind"] == "chemically_distinct_synthetic_families"]
    ci_rng = np.random.default_rng(SEED + 2)
    related_ci = _cluster_bootstrap_ci(pair_rows, "chemically_related_synthetic_family", ci_rng)
    distinct_ci = _cluster_bootstrap_ci(pair_rows, "chemically_distinct_synthetic_families", ci_rng)
    same_frame = pair_frame[pair_frame["pair_kind"] == "same_composition_different_generated_arrangement"]
    all_comp_pairs = pair_frame[pair_frame["pair_kind"] != "same_composition_different_generated_arrangement"]
    rho = spearmanr(
        all_comp_pairs["composition_l1_distance"].to_numpy(),
        all_comp_pairs["jensen_shannon_divergence_bits"].to_numpy(),
    ).statistic

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path_info = _plot_paths(paths, figures_dir)
    _plot_heatmap(samples, weights, figures_dir, n_representatives=26)
    _plot_heatmap(samples, weights, figures_dir, n_representatives=None)
    _plot_similarity(pair_rows, figures_dir)
    _plot_concentration(samples, metrics, figures_dir)
    _plot_family_means(samples, weights, figures_dir)

    summary = {
        "scope": "synthetic composition probes only; archived training and validation data unavailable",
        "sample_count": len(samples),
        "family_count": len(FAMILY_ELEMENTS),
        "samples_per_family": N_PER_FAMILY,
        "atom_count_per_sample": TOTAL_ATOMS,
        "domain_classification": "unknown for all synthetic records",
        "checkpoint_manifest": checkpoint_manifest,
        "pilot": {
            "compositions": 5,
            "minimum_weight": min_weight,
            "max_sum_error": max_sum_error,
            "max_repeat_difference": max_repeat_difference,
        },
        "global_concentration": {
            "mean_effective_branches": float(np.mean([metric["effective_branches"] for metric in metrics])),
            "min_effective_branches": float(np.min([metric["effective_branches"] for metric in metrics])),
            "max_effective_branches": float(np.max([metric["effective_branches"] for metric in metrics])),
            "mean_max_weight": float(np.mean([metric["max_weight"] for metric in metrics])),
            "mean_top3_sum": float(np.mean([metric["top3_sum"] for metric in metrics])),
            "mean_support90": float(np.mean([metric["support90"] for metric in metrics])),
            "mean_support95": float(np.mean([metric["support95"] for metric in metrics])),
        },
        "weight_numerics": {
            "exact_zero_count": int(np.count_nonzero(weights == 0)),
            "total_weights": int(weights.size),
            "samples_with_at_least_one_exact_zero": int(np.count_nonzero((weights == 0).any(axis=1))),
            "interpretation": "float64 softmax underflow; not branch skipping",
        },
        "pairwise_statistics": {
            "same_composition_pair_count": int(len(same_frame)),
            "same_composition_max_js": float(same_frame["jensen_shannon_divergence_bits"].max()),
            "related_js": {
                "pair_count": int(len(related_frame)),
                "mean": float(related_frame["jensen_shannon_divergence_bits"].mean()),
                "median": float(related_frame["jensen_shannon_divergence_bits"].median()),
                "mean_total_variation": float(related_frame["total_variation_distance"].mean()),
                "dominant_branch_agreement": float(related_frame["dominant_branch_agreement"].mean()),
                **(related_ci or {}),
            },
            "distinct_js": {
                "pair_count": int(len(distinct_frame)),
                "mean": float(distinct_frame["jensen_shannon_divergence_bits"].mean()),
                "median": float(distinct_frame["jensen_shannon_divergence_bits"].median()),
                "mean_total_variation": float(distinct_frame["total_variation_distance"].mean()),
                "dominant_branch_agreement": float(distinct_frame["dominant_branch_agreement"].mean()),
                **(distinct_ci or {}),
            },
            "spearman_rho": float(rho),
            "spearman_inference": "descriptive only; pair observations share compositions",
        },
        "same_composition_structure_pairs": int(len(same_frame)),
        "family_statistics": family_statistics,
        "composition_paths": path_info,
        "domain_comparison": "not possible without archived training composition distribution",
    }
    (output_dir / "summary_statistics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output_dir, model_info, summary)
    print(
        f"Wrote synthetic probe analysis for {len(samples)} compositions to {output_dir}; "
        f"mean N_eff={summary['global_concentration']['mean_effective_branches']:.3f}, "
        f"Spearman rho={rho:.4f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--mlp-checkpoint", type=Path, default=DEFAULT_MLP)
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_BACKBONE)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.mlp_checkpoint.resolve(), args.backbone_checkpoint.resolve())


if __name__ == "__main__":
    main()
