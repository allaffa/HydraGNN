# Anderson acceleration with AADL

HydraGNN can wrap every optimizer selected through
`hydragnn.utils.optimizer.select_optimizer` with AADL. Install it after the
machine-specific PyTorch environment:

```bash
python -m pip install -r requirements-aadl.txt
```

Enable acceleration inside the existing optimizer configuration:

```json
"Optimizer": {
  "type": "AdamW",
  "learning_rate": 0.001,
  "AADL": {
    "enabled": true,
    "acceleration_type": "anderson",
    "relaxation": 0.5,
    "wait_iterations": 100,
    "history_depth": 8,
    "store_each_nth": 1,
    "frequency": 20,
    "reg_acc": 1e-7,
    "sketch_fraction": 0.05,
    "sketch_policy": "backward_error",
    "safeguard": false
  }
}
```

All keys other than `enabled` are passed to `AADL.accelerate`. The wrapper is
applied after HydraGNN constructs SGD, Adam, Adadelta, Adagrad, Adamax, AdamW,
RMSprop, FusedLAMB, or their supported zero-redundancy form. PyTorch continues
to own optimizer state, AMP stepping, schedulers, and DDP communication.

## Safeguard limitation

HydraGNN's standard loop computes backward before calling `optimizer.step()`;
it does not use an optimizer closure. Consequently, loss-based AADL safeguards
cannot reevaluate candidates and `safeguard=true` is rejected. Algebraic
backward-error sketch checks remain available. Supporting loss safeguards
later requires a closure that faithfully reproduces HydraGNN's multi-head and
energy/force loss, including gradient-derived forces.

## LocalSGD

Anderson histories are rank-local during LocalSGD intervals. Whenever
HydraGNN's native Post-LocalSGD wrapper averages parameters—periodically or at
an epoch boundary—it resets AADL history. This prevents pre-synchronization
local iterates from being mixed with the newly averaged model. Optimizer-state
synchronization remains governed by HydraGNN's existing LocalSGD policy.
