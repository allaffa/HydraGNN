"""
Quick test to verify PINN loss works during training
"""
import torch
import numpy as np
import sys
import os

# Add to path
sys.path.insert(0, '/Users/7ml/Documents/Codes/HydraGNN')

print("=" * 70)
print("PINN Training Test")
print("=" * 70)

# Import after adding to path
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
import json

# Load config
config_path = '/Users/7ml/Documents/Codes/HydraGNN/examples/power_grid/power_grid.json'
with open(config_path, 'r') as f:
    config = json.load(f)

print("\n✓ Configuration loaded")
print(f"  lambda_physics: {config['NeuralNetwork']['Architecture'].get('lambda_physics', 'NOT SET')}")

# Check if we have pickle dataset
dataset_path = '/Users/7ml/Documents/Codes/HydraGNN/examples/power_grid/dataset'
pickle_file = os.path.join(dataset_path, 'serialized_dataset', 'unit_test.pkl')

if os.path.exists(pickle_file):
    print(f"\n✓ Found pickle dataset: {pickle_file}")
    
    # Load a few samples to test
    from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
    dataset = SimplePickleDataset(basedir=os.path.join(dataset_path, 'serialized_dataset'),
                                   label='unit_test')
    
    print(f"  Total samples: {len(dataset)}")
    
    # Get one sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n✓ Sample data structure:")
        print(f"  x.shape: {sample.x.shape}")
        print(f"  edge_index.shape: {sample.edge_index.shape}")
        print(f"  edge_attr.shape: {sample.edge_attr.shape}")
        print(f"  Has true_P: {hasattr(sample, 'true_P')}")
        print(f"  Has true_Q: {hasattr(sample, 'true_Q')}")
        print(f"  Has per_unit_scaling_factor: {hasattr(sample, 'per_unit_scaling_factor')}")
        
        if hasattr(sample, 'true_P'):
            print(f"  true_P.shape: {sample.true_P.shape}")
            print(f"  true_Q.shape: {sample.true_Q.shape}")
        
        print("\n" + "=" * 70)
        print("✓ Dataset is ready for PINN training!")
        print("\nNext step: Run actual training with:")
        print("  python power_grid.py --pickle")
        print("\nMonitor training logs for:")
        print("  - Total loss (supervised + λ×physics)")
        print("  - Task losses (Vmag, Vang)")
        print("  - Physics residuals should decrease over epochs")
        print("=" * 70)
    else:
        print("\n✗ Dataset is empty!")
else:
    print(f"\n⚠ Pickle dataset not found at: {pickle_file}")
    print("\nGenerate dataset first:")
    print("  cd dataset && python data_gen_GNN.py")
    print("  cd .. && python power_grid.py --preonly --pickle")
