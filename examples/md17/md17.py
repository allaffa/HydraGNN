import os
import json
import torch
import torch_geometric
import argparse

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn


# Update each sample prior to loading.
def md17_pre_transform(data, compute_edges):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict energy (index 0 of 2 properties) for this run.
    data.y = data.energy / len(data.x)
    graph_features_dim = [1]
    node_feature_dim = [1]
    data = compute_edges(data)
    return data


# Randomly select ~1000 samples
def md17_pre_filter(data):
    return torch.rand(1) < 0.25


def main(model_type=None):
    # FIX random seed
    random_state = 0
    torch.manual_seed(random_state)

    # Set this path for output.
    os.environ.setdefault("SERIALIZED_DATA_PATH", os.getcwd())

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(__file__), "md17.json")
    with open(filename, "r") as f:
        config = json.load(f)

    verbosity = config["Verbosity"]["level"]
    arch_config = config["NeuralNetwork"]["Architecture"]

    # If a model type is provided, update the configuration
    if model_type:
        config["NeuralNetwork"]["Architecture"]["model_type"] = model_type

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()

    log_name = f"md17_test_{model_type}" if model_type else "md17_test"
    # Enable print to log file.
    hydragnn.utils.print.print_utils.setup_log(log_name)

    # Preprocess configurations for edge computation
    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

    # Fix for MD17 datasets
    torch_geometric.datasets.MD17.file_names["uracil"] = "md17_uracil.npz"

    dataset = torch_geometric.datasets.MD17(
        root="dataset/md17",
        name="uracil",
        pre_transform=lambda data: md17_pre_transform(data, compute_edges),
        pre_filter=md17_pre_filter,
    )
    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # Run training with the given model and md17 dataset.
    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MD17 example with an optional model type."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Specify the model type for training (default: None).",
    )
    args = parser.parse_args()
    main(model_type=args.model_type)
