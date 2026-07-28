# Methods Reference

All merge methods accept a common base checkpoint, fine-tuned checkpoints, optional task weights, and `alpha`. Expensive work belongs in `prepare`; `apply` then evaluates alpha candidates without rebuilding the method state.

## Merge Methods

| Name | Class | Summary | Key `method_params` |
|---|---|---|---|
| `weighted_average` | `WeightedAverageMerge` | Averages checkpoints, then interpolates from the base. | None. |
| `task_arithmetic` | `TaskArithmeticMerge` | Weighted sum of task vectors. | None. |
| `ties_merge` | `TIESMerge` | Prunes task vectors and resolves conflicting signs. | `topk`, `merge_func`. |
| `dare_merge` | `DAREMerge` | Randomly sparsifies each task vector before aggregation. | `drop_rate`, `seed`, `rescale`, `low_memory`. |
| `tsv_merge` | `TSVMerge` | Merges matrix updates through truncated singular-vector components. | `sv_reduction`, `max_rank`, `svd_dtype`, `vector_1d_merge`. |
| `isoc_merge` | `IsoCMerge` | Isotropizes the weighted matrix update. | `vector_1d_merge`. |
| `isocts_merge` | `IsoCTSMerge` | Separates common and task-specific matrix subspaces. | `common_space_fraction`, `vector_1d_merge`. |
| `cart_merge` | `CARTMerge` | Applies low-rank Cartesian task-vector composition. | `pruning_rank`, `scaling_coeffs`. |
| `pcb` / `pcb_merge` | `PCBMerge` | Balances flattened task vectors with intra/inter-task signals. | `clamp_min_ratio`, `clamp_max_ratio`, `att_ratio`, `lam`. |
| `actmerge` | `ActMerge` | Uses task-vector activation geometry for dense matrices. | Method-specific matrix options. |
| `dc_merge` | `DCMerge` | Dense task-delta composition. | Method-specific composition options. |
| `wudi` / `wudi_merge` | `WUDIMerge` | Solves a per-matrix WUDI objective. | `vector_1d_merge`. |

Matrix-oriented methods use their method-specific rule for 2D tensors. `vector_1d_merge: "average"` enables averaged 1D deltas where supported; the default is typically zero task-vector deltas.

## Transport Methods

| Name | Class | Summary |
|---|---|---|
| `identity` | `IdentityTransport` | Applies compatible task-vector tensors unchanged. |
| `orthogonal_shift` | `OrthogonalShiftTransport` | Removes the component aligned with the base-model shift. |
| `gradfix` | `GradFixRebase` | Uses target-model gradient signs to mask or correct task-vector components. |
| `theseus` | `TheseusRebase` | Builds activation-aligned coordinate transforms for matrix updates. |
| `transfusion` | `TransFusionRebase` | Uses Transformer-aware weight permutations. |
| `bico` | `BiCoRebase` | Estimates bilinear input/output coordinate maps from activations and gradients. |
| `bico_gradin` | `BiCoGradInRebase` | BiCo variant using input-side gradients for the input map. |

## Post-Merge Methods

| Name | Class | Summary |
|---|---|---|
| `adamerging` | `AdaMergingPostMerge` | Optimizes task or layer alpha values against an entropy objective. |
| `task_vector_finetune` | `TaskVectorFinetunePostMerge` | Fine-tunes each task delta after composition. |
| `merged_delta_finetune` | `MergedDeltaFinetunePostMerge` | Fine-tunes one merged visual delta. |
| `vision_head_probe` | `VisionHeadProbePostMerge` | Fine-tunes only final vision-head task-vector tensors. |

The source classes document method-specific behavior beside their implementations under `src/merge_and_rebase/{merge,rebase,postmerge}/methods/`.
