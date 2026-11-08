"""ASE calculator for the frozen fused HydraGNN/BranchWeightMLP model."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from torch_geometric.data import Batch, Data

SC26_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SC26_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SC26_DIR))

from inference_fused import (  # noqa: E402
    _fused_energy_forces,
    _predict_branch_energy_only,
    _reshape_composition,
    load_fused_stack,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import (  # noqa: E402
    get_radius_graph_pbc,
)

MLP_SHA256 = "c984b338ee947534bbdedf1fc4dd2aea92bb846e3302638d6719b1e3f5ca7ca8"
BACKBONE_SHA256 = "8532b03296a8ab4cfc3b37beb6d57b2091a50807b155fac93cb1882253d32867"
NUM_BRANCHES = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_verified_state(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        while normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        if normalized_key in normalized:
            raise ValueError(f"Duplicate {label} state key after prefix normalization: {normalized_key}")
        normalized[normalized_key] = value
    named_modules = dict(module.named_modules())
    for key, value in normalized.items():
        if key.endswith("graph_concat_projector.weight"):
            module_name = key[: -len(".graph_concat_projector.weight")]
            target = named_modules.get(module_name)
            if target is None or not hasattr(target, "_ensure_graph_concat_projector"):
                raise ValueError(f"Cannot materialize checkpoint layer for {key}")
            channel_dim = int(target.hidden_dim)
            graph_attr_dim = int(value.shape[1]) - channel_dim
            if graph_attr_dim <= 0:
                raise ValueError(f"Invalid graph-concat dimensions in checkpoint layer {key}")
            parameter = next(target.parameters())
            target._ensure_graph_concat_projector(
                graph_attr_dim=graph_attr_dim,
                channel_dim=channel_dim,
                device=parameter.device,
                dtype=parameter.dtype,
            )
    expected_keys = set(module.state_dict())
    missing = sorted(expected_keys - set(normalized))
    unexpected = sorted(set(normalized) - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"{label} checkpoint keys do not match constructed module; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    module.load_state_dict(normalized, strict=True)


def _graph_from_atoms(
    atoms: Atoms,
    *,
    dtype: torch.dtype,
    radius: float,
    max_neighbours: int,
    charge: float,
    spin: float,
) -> Data:
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    composition = np.bincount(atomic_numbers, minlength=119)[1:119]
    return get_radius_graph_pbc(
        radius=radius,
        max_neighbours=max_neighbours,
    )(
        Data(
            x=torch.as_tensor(atomic_numbers, dtype=dtype).reshape(-1, 1),
            atomic_numbers=torch.as_tensor(atomic_numbers, dtype=torch.long),
            pos=torch.as_tensor(
                np.asarray(atoms.positions, dtype=np.float64), dtype=dtype
            ),
            chemical_composition=torch.as_tensor(
                composition, dtype=torch.int64
            ).reshape(-1, 1),
            graph_attr=torch.as_tensor([charge, spin], dtype=dtype),
            natoms=torch.as_tensor([len(atoms)], dtype=torch.long),
            cell=torch.as_tensor(np.asarray(atoms.cell), dtype=dtype),
            pbc=torch.as_tensor(np.asarray(atoms.pbc), dtype=torch.bool),
        )
    )


class FusedHydraGNNCalculator(Calculator):
    """Evaluate the complete 16-head weighted energy and its coordinate gradient."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model: torch.nn.Module,
        mlp: torch.nn.Module,
        *,
        precision: str,
        param_dtype: torch.dtype,
        device: torch.device,
        autocast_ctx: Any,
        mlp_device: torch.device,
        mlp_autocast_ctx: Any,
        radius: float,
        max_neighbours: int,
        charge: float = 0.0,
        spin: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.mlp = mlp.eval()
        self.precision = precision
        self.param_dtype = param_dtype
        self.device = device
        self.autocast_ctx = autocast_ctx
        self.mlp_device = mlp_device
        self.mlp_autocast_ctx = mlp_autocast_ctx
        self.radius = float(radius)
        self.max_neighbours = int(max_neighbours)
        self.charge = float(charge)
        self.spin = float(spin)
        self.num_branches = int(getattr(model, "num_branches", 1))
        if self.num_branches != NUM_BRANCHES:
            raise ValueError(f"Expected {NUM_BRANCHES} model branches, got {self.num_branches}")
        if sum(1 for _ in mlp.parameters()) == 0:
            raise ValueError("BranchWeightMLP has no parameters")
        for parameter in (*model.parameters(), *mlp.parameters()):
            if parameter.requires_grad:
                raise ValueError("Model parameters must be frozen during relaxation")
        expected_dtype = torch.float32 if precision == "fp32" else torch.float64
        if param_dtype != expected_dtype:
            raise ValueError(
                f"Requested {precision}, but HydraGNN parameters use {param_dtype}"
            )
        if any(parameter.dtype != expected_dtype for parameter in model.parameters()):
            raise ValueError("HydraGNN parameters do not match the requested precision")
        if any(parameter.dtype != expected_dtype for parameter in mlp.parameters()):
            raise ValueError("BranchWeightMLP parameters do not match the requested precision")
        self.last_branch_weights: np.ndarray | None = None
        self.last_branch_energies: np.ndarray | None = None
        self.last_graph_edges = 0
        self.graph_build_count = 0
        self.last_graph_positions: np.ndarray | None = None

    def evaluate_atoms(self, atoms: Atoms) -> dict[str, Any]:
        torch.set_default_dtype(self.param_dtype)
        data = _graph_from_atoms(
            atoms,
            dtype=self.param_dtype,
            radius=self.radius,
            max_neighbours=self.max_neighbours,
            charge=self.charge,
            spin=self.spin,
        )
        self.graph_build_count += 1
        self.last_graph_edges = int(data.edge_index.size(1))
        self.last_graph_positions = np.array(atoms.positions, dtype=np.float64, copy=True)
        original_composition = np.bincount(
            atoms.get_atomic_numbers(), minlength=119
        )[1:119]

        batch = Batch.from_data_list([data]).to(self.device)
        for key in batch.keys():
            value = batch[key]
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=self.param_dtype)
        batch.pos = batch.pos.detach().requires_grad_(True)

        composition = _reshape_composition(batch)
        if composition.shape != (1, 118):
            raise ValueError(
                f"Expected one 118-element composition vector, got {tuple(composition.shape)}"
            )
        actual_composition = composition.detach().cpu().numpy()[0]
        if not np.array_equal(actual_composition, original_composition):
            raise ValueError("Graph composition does not match the exact atomic-number counts")

        with torch.enable_grad():
            with self.mlp_autocast_ctx:
                mlp_input = composition.to(
                    device=self.mlp_device,
                    dtype=next(self.mlp.parameters()).dtype,
                )
                logits = self.mlp(mlp_input)
                weights = F.softmax(logits, dim=-1).to(
                    device=self.device, dtype=self.param_dtype
                )
            if weights.shape != (1, NUM_BRANCHES):
                raise ValueError(f"Unexpected BranchWeightMLP output shape {tuple(weights.shape)}")
            if not torch.isfinite(weights).all() or not torch.allclose(
                weights.sum(dim=-1),
                torch.ones(1, device=self.device, dtype=self.param_dtype),
                atol=2e-6 if self.param_dtype == torch.float32 else 2e-14,
                rtol=0,
            ):
                raise FloatingPointError("Branch weights are nonfinite or do not sum to one")

            branch_energies: list[torch.Tensor] = []
            with self.autocast_ctx:
                for branch_id in range(self.num_branches):
                    branch_energy = _predict_branch_energy_only(
                        self.model, batch, branch_id
                    )
                    if branch_energy.numel() != 1:
                        raise ValueError(
                            f"Branch {branch_id} returned {branch_energy.numel()} energies "
                            "for one structure"
                        )
                    branch_energies.append(branch_energy.reshape(1))
                weighted_energy, forces = _fused_energy_forces(
                    branch_energies,
                    [weights[:, branch_id] for branch_id in range(self.num_branches)],
                    [],
                    [],
                    batch.pos,
                    batch.batch,
                )

        branch_energy_values = torch.stack(
            [energy.detach().reshape(()) for energy in branch_energies]
        )
        if (
            not torch.isfinite(weighted_energy).all()
            or not torch.isfinite(forces).all()
            or not torch.isfinite(branch_energy_values).all()
        ):
            raise FloatingPointError(
                f"Nonfinite {self.precision} energy or force prediction"
            )
        self.last_branch_weights = weights[0].detach().cpu().numpy().astype(np.float64)
        self.last_branch_energies = (
            branch_energy_values.detach().cpu().numpy().astype(np.float64)
        )
        return {
            "energy": float(weighted_energy.reshape(-1)[0].item()),
            "forces": forces.detach().cpu().numpy().astype(np.float64),
            "branch_weights": self.last_branch_weights.copy(),
            "branch_energies": self.last_branch_energies.copy(),
            "edge_count": self.last_graph_edges,
            "edge_index": data.edge_index.detach().cpu().numpy().copy(),
        }

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("ASE did not provide an Atoms object")
        result = self.evaluate_atoms(atoms)
        self.results["energy"] = result["energy"]
        self.results["forces"] = result["forces"]


def load_fused_calculator(
    *,
    logdir: Path,
    backbone_checkpoint: Path,
    mlp_checkpoint: Path,
    precision: str,
    charge: float,
    spin: float,
) -> tuple[FusedHydraGNNCalculator, dict[str, Any]]:
    expected_hashes = {
        "mlp": (mlp_checkpoint, MLP_SHA256),
        "backbone": (backbone_checkpoint, BACKBONE_SHA256),
    }
    for label, (path, expected) in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"{label} checkpoint SHA-256 mismatch: expected {expected}, got {observed}"
            )

    mlp_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    (
        model,
        mlp,
        config,
        device,
        autocast_ctx,
        param_dtype,
        num_branches,
        mlp_device,
        mlp_autocast_ctx,
        _unified_stack,
        _gnn_precision,
        _mlp_precision,
    ) = load_fused_stack(
        str(logdir),
        str(backbone_checkpoint),
        str(mlp_checkpoint),
        precision,
        precision,
        mlp_device_str,
    )
    backbone_state = torch.load(
        backbone_checkpoint, map_location=device, weights_only=True
    )["model_state_dict"]
    _restore_verified_state(model, backbone_state, label="HydraGNN")
    mlp_state = torch.load(
        mlp_checkpoint, map_location=mlp_device, weights_only=True
    )["mlp_state_dict"]
    _restore_verified_state(mlp, mlp_state, label="BranchWeightMLP")
    model.eval()
    mlp.eval()
    for parameter in (*model.parameters(), *mlp.parameters()):
        parameter.requires_grad_(False)
    if num_branches != NUM_BRANCHES:
        raise ValueError(f"Expected {NUM_BRANCHES} branches, got {num_branches}")
    architecture = config["NeuralNetwork"]["Architecture"]
    calculator = FusedHydraGNNCalculator(
        model,
        mlp,
        precision=precision,
        param_dtype=param_dtype,
        device=device,
        autocast_ctx=autocast_ctx,
        mlp_device=mlp_device,
        mlp_autocast_ctx=mlp_autocast_ctx,
        radius=float(architecture.get("radius", 5.0)),
        max_neighbours=int(architecture.get("max_neighbours", 20)),
        charge=charge,
        spin=spin,
    )
    provenance = {
        "precision": precision,
        "device": str(device),
        "mlp_device": str(mlp_device),
        "parameter_dtype": str(param_dtype),
        "mlp_parameter_dtype": str(next(mlp.parameters()).dtype),
        "number_of_branches": calculator.num_branches,
        "branch_output_order": [
            item["type"]
            for item in config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"]
        ],
        "checkpoint_sha256": {
            "hydragNN": BACKBONE_SHA256,
            "branch_weight_mlp": MLP_SHA256,
        },
        "state_dict_prefix_normalization": "Repeated leading 'module.' prefixes removed; strict key match then enforced",
        "radius_angstrom": calculator.radius,
        "max_neighbours": calculator.max_neighbours,
        "model_eval_mode": not calculator.model.training,
        "mlp_eval_mode": not calculator.mlp.training,
        "all_parameters_frozen": all(
            not parameter.requires_grad
            for parameter in (*calculator.model.parameters(), *calculator.mlp.parameters())
        ),
    }
    return calculator, provenance
