# Concepts

## Base Model and Task Vector

For a base checkpoint `theta_base` and a task-specific checkpoint `theta_task`, a task vector is the parameter update:

```text
delta_task = theta_task - theta_base
```

Merge methods combine task vectors or task checkpoints, then apply a scalar `alpha` to interpolate from the base. This requires compatible parameter keys and tensor shapes.

## Merging and Rebasin

Merging combines independently specialized updates defined relative to the same base. Rebasin transports one task vector from a source base coordinate system to a target base coordinate system. They share checkpoint and task-vector interfaces, but the current rebasin evaluator transports one task vector at a time; a fully configurable multi-task merge-then-transport pipeline is future work.

## Preparation and Application

Methods can implement two phases:

- `prepare`: expensive, alpha-independent work such as SVDs, activation alignment, or gradient collection.
- `apply`: cheap construction of a result for a chosen alpha.

The evaluation code uses this split to avoid repeating preparation during alpha search. Change method parameters or checkpoint inputs when you need a new prepared state.

## Metrics

Vision entrypoints report raw test accuracy. Some analyses additionally report normalized values relative to a selected baseline; these are ratios, not percentages bounded by 100. Always label raw accuracy and normalization denominator separately.

## Checkpoint Compatibility

Local and hosted checkpoints are structurally validated by matching supported wrappers, normalizing common key prefixes, and checking tensor keys and shapes. Structural compatibility does not prove a checkpoint's claimed training data, split, seed, or base revision. Use the manifest's hashes and provenance metadata for released artifacts.
