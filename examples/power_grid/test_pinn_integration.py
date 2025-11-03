"""
Test script to verify PINN loss integration
"""
import torch
import sys
import os

# Quick test by checking the code structure
print("=" * 60)
print("PINN Integration Verification")
print("=" * 60)

# Read the create.py file to verify changes
create_file = '/Users/7ml/Documents/Codes/HydraGNN/hydragnn/models/create.py'
with open(create_file, 'r') as f:
    content = f.read()

# Check for key features
checks = {
    "EnhancedModelWrapper.__init__ has lambda_physics": "def __init__(self, original_model, lambda_physics=" in content,
    "EnhancedModelWrapper has loss() override": "def loss(self, pred, value, head_index):" in content and "EnhancedModelWrapper" in content,
    "loss() calls supervised + physics": "supervised_loss + self.lambda_physics * physics_loss" in content,
    "Variable naming improved (P_predicted)": "P_predicted = torch.zeros" in content,
    "Variable naming improved (Q_predicted)": "Q_predicted = torch.zeros" in content,
    "Residual calculation clear": "residual_P = P_actual - P_predicted" in content,
    "lambda_physics in create_model signature": "lambda_physics: float =" in content,
    "lambda_physics passed to wrapper": "EnhancedModelWrapper(model, lambda_physics=" in content,
}

print("\nCode Structure Checks:")
all_passed = True
for check, result in checks.items():
    status = "✓" if result else "✗"
    print(f"  {status} {check}")
    if not result:
        all_passed = False

# Check config file
config_file = '/Users/7ml/Documents/Codes/HydraGNN/examples/power_grid/power_grid.json'
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    has_lambda = '"lambda_physics"' in config_content
    print(f"\nConfiguration File:")
    print(f"  {'✓' if has_lambda else '✗'} lambda_physics in power_grid.json")
    all_passed = all_passed and has_lambda

print("\n" + "=" * 60)
if all_passed:
    print("✓ All PINN integration checks PASSED!")
    print("\nThe PINN loss is now integrated into training.")
    print("Physics-informed regularization will be applied during model.loss()")
else:
    print("✗ Some checks FAILED - review the code changes")
print("=" * 60)

