# steer4rebase → `steer` rebase method: porting recap

## What this is

`steer4rebase` (`/Users/fabiobozzoli/rebasin_linear/steer4rebase`) learns a **feature-space correction** that makes a frozen target CLIP vision encoder (B) behave like a fine-tuned source encoder (A), without touching B's weights. This has been ported into `merge-and-rebase` as a new rebase method named `steer`, usable from `vision_rebase.py` like any other method (`"method": "steer"` in a config).

It is architecturally different from every other rebase method in this repo (`gradfix`, `theseus`, `transfusion`, `bico`): those all produce a **weight-space delta** added to the target model's state dict. `steer` cannot — its correction depends on the target model's *intermediate activations* (for `block_ridge`) or is nonlinear (`global_mlp`), so no state-dict delta can represent it exactly. Instead, `steer` wraps the target classifier's `encode_image` at eval time to inject the correction directly into the forward pass. `vision_rebase.py` has a dedicated `steer_mode` dispatch branch for this (mirrors the existing `transfusion_mode`/`bico_mode` special-casing).

## What was ported

| steer4rebase piece | Ported as | Notes |
|---|---|---|
| `stage1.py` (`projection`, `target_corrections`) | `_stage1_projection`, `_stage1_target_corrections` | Unchanged math. |
| `stage2.py` (`ridge`, `global_ridge`, `ResidualMLP`/`global_mlp`, `block_ridge`) | `_ridge`, `_fit_global_mlp`, `_fit_block_ridge`/`_predict_block_ridge` | Unchanged math. |
| `data.py` (`few_shot`, `random_sample`, `group_blocks`) | `_few_shot`, `_random_sample`, `_group_blocks_concat`/`_group_blocks_sum_avg` | Only `concat`/`sum_avg` grouping ported (the two non-experimental strategies); `group_grid.py`'s V11/V12 distributed-weight variants are **not** ported. |
| `linearize.py` (`clip_vit_parameter_blocks`, `clip_resnet_parameter_blocks`, activation hooks) | `_clip_vit_parameter_blocks`, `_clip_resnet_parameter_blocks`, `_BlockActivationCapture` | Both ViT and ResNet CLIP visual backbones are supported as the **target** model. |
| `LinearizedModelV2`'s Taylor-linearized delta (functorch JVP) | Reuses `merge_and_rebase.utils.linearization.LinearizedModule` (already in this repo, `torch.func.jvp`-based) | No new dependency needed — this repo already had a modern equivalent. |
| `bea_utils.py` feature collection (`collect_standard_split_artifacts`, `collect_linearized_split_artifacts`) | `_collect_standard_split`, `_collect_linear_split` | Reuses this repo's `OpenClipClassifier`/loaders instead of the original CLI/dataloader plumbing. |
| `run.py` orchestration | `SteerRebase.prepare()` + `steer_correction_context()` | See below. |

Everything lives in one file: `src/merge_and_rebase/rebase/methods/steer.py` (same self-contained convention as `theseus.py`).

**Deferred / not ported:** `group_grid.py`'s V11/V12 distributed-weight block grouping, and the dead/commented-out experimental methods in `steering.py` (procrustes, learnable rotation, etc.).

**Bugs fixed along the way (pre-existing, not steer-specific):**
- `utils/linearization.py` skipped the `sdpa_kernel(MATH)` context needed for forward-mode AD when running on CPU — now applied unconditionally.
- `vision_rebase.py` read source/target ViT depth unconditionally, which would crash for any method (not just `steer`) using a ResNet target — now guarded.

## The two phases

1. **`prepare()`** (expensive, cached): computes A's fine-tuning delta and B's activations over the task's train/test splits, either as real forward-pass differences (`feature_regime="standard"`) or as a Taylor-linearized delta via `LinearizedModule` (`feature_regime="linear"` — required for `stage_2_strategy="block_ridge"`, since per-block deltas only make sense in the linearized decomposition). Results are cached to disk as individual `.pt` tensors under `<feature_cache_dir>/<source_tag>_to_<target_tag>/<task>/<regime>/<split>/`. Then it samples the few-shot support set, fits Stage 1 (`target_corrections`), and fits the chosen Stage 2 predictor (`global_ridge` / `global_mlp` / `block_ridge`), returning a `prepared` dict.
2. **Eval time**: `steer_correction_context(clf_target, prepared, alpha=...)` is a context manager that monkey-patches `clf_target.model.encode_image` to run the real forward pass, capture B's block activations via hooks, compute the Stage-2 correction, and add `alpha * correction` to the pooled feature before it's used for classification. This is what the alpha-sweep in `vision_rebase.py` scales — same "sweep alpha, pick best on val, report on test" loop as every other method, just applying a correction instead of a weight delta. Because there's no weight delta, the "untransported" baseline is unavailable for `steer` tasks, so the baseline reported is always `target_zeroshot` (uncorrected B) — which is exactly the right comparison for this method.

## `method_params` reference

| Parameter | Default | Values | Notes |
|---|---:|---|---|
| `feature_regime` | `"standard"` | `"standard"` \| `"linear"` | `"linear"` required for `block_ridge`. |
| `stage_2_strategy` | `"global_ridge"` | `"global_ridge"` \| `"global_mlp"` \| `"block_ridge"` | |
| `block_group_strategy` | `"concat"` | `"concat"` \| `"sum_avg"` | Only used by `block_ridge` when source/target depths differ (must be an integer ratio, e.g. 12→24). |
| `block_ridge_mode` | `"independent"` | `"independent"` \| `"smoothed_residual"` | Chained per-block residual smoothing, controlled by `rho`. |
| `rho` | `0.9` | `[0, 1]` | Only used when `block_ridge_mode="smoothed_residual"`. |
| `few_shot` | — | int | Shots per class. Exactly one of `few_shot` / `total_support_examples` must be set. |
| `total_support_examples` | — | int | Fixed-size random support set instead of per-class few-shot. |
| `stage1_lambda` | `1.0` | float | Stage 1 ridge regularization. |
| `ridge_lambda` | `1.0` | float | Stage 2 `global_ridge`/`block_ridge` regularization. |
| `mlp_hidden_dim` | `1024` | int | `global_mlp` only. |
| `mlp_epochs` | `100` | int | `global_mlp` only. |
| `feature_cache_dir` | `"src/.cache/steer_features"` | path | Where computed features are cached/looked up. |
| `force_recompute_features` | `false` | bool | Bypass the cache and recompute+overwrite. |
| `seed` | `42` | int | Support-set sampling / MLP init. |

## How to launch a run

Same entrypoint as every other rebase method:

```bash
python -m merge_and_rebase.eval.vision_rebase --config configs/vision8_steer.json
```

Two ready-made example configs:

- **`configs/vision8_steer.json`** — same-architecture pair (ViT-B-16 → ViT-B-16-plus-240), `block_ridge` + `smoothed_residual`.
- **`configs/vision_crossarch_steer.json`** — cross-architecture pair (ViT-B-16 → ViT-L-14, 12→24 blocks), exercises the `block_group_strategy="concat"` grouping path.

A ResNet target (e.g. `target_clip_model: "RN50"`) works with the same configs unchanged — `steer.py` detects the visual backbone type automatically and hooks `layer{1..4}` bottlenecks instead of transformer resblocks (`global_ridge`/`global_mlp` don't need per-block hooks at all, so they're backbone-agnostic either way).

To try the other Stage 2 strategies, edit `method_params.stage_2_strategy`:
- `"global_ridge"` — works with `feature_regime: "standard"` or `"linear"`.
- `"global_mlp"` — same, plus `mlp_hidden_dim`/`mlp_epochs`.
- `"block_ridge"` — **requires** `feature_regime: "linear"`.

First run for a given `(source, target, task, regime)` combination computes and caches features (slower); subsequent runs reuse the cache automatically unless `force_recompute_features: true` is set.

## Testing

`tests/test_steer_rebase.py` — 11 synthetic tests (no GPU/real checkpoints needed): Stage 1/2 math, disk-cache round-trip (including forced recompute), ViT and ResNet block-map correctness, and two full `prepare()` → `steer_correction_context()` end-to-end runs against a tiny fake CLIP model (including a cross-architecture dimension-safety check). Run with:

```bash
pytest tests/test_steer_rebase.py -v
```

**Not yet done:** a real end-to-end run against actual checkpoints/datasets (e.g. the GTSRB task referenced in the example configs) — the example configs' `tuned_ckpts` path is a placeholder copied from the `theseus` configs and should be checked against what's actually available before running for real.

---

# SVHN: preprocessing custom portato da `rebasin_linear`

## What this is

Il finetuning di SVHN in `rebasin_linear` usa un preprocessing su misura al posto di quello CLIP standard. È stato portato in `merge-and-rebase` come override **opt-in**, spento di default.

Le cifre SVHN sono 32×32: la pipeline CLIP le porta direttamente a 224 e ritaglia, sfocandole. La pipeline custom le ridimensiona a un `target_size` più piccolo (96) e le rimette al centro di una tela nera 224×224, mantenendo la cifra nitida, e applica una policy di augmentation al solo split di train.

| | pipeline CLIP standard | pipeline SVHN custom |
|---|---|---|
| geometria | `Resize(224)` + `CenterCrop(224)` | `Resize(96)` BICUBIC+antialias, poi `Pad` a 224 centrato con `fill=0` |
| train aug | nessuna | `AutoAugment(policy=SVHN)` fra resize e pad (default) |
| eval | stesso transform del train | stessa geometria, **senza** augmentation |

## What was ported

| pezzo di `rebasin_linear` | Ported as | Note |
|---|---|---|
| `src/datasets/svhn.py`, `train_preprocess` / `test_preprocess` | `build_svhn_transforms()` in `src/merge_and_rebase/data/svhn_preprocess.py` | Stesso ordine delle operazioni: convert RGB → resize → augment → pad → ToTensor → normalize. |
| `AUGMENTATION_CONFIG = 0..6` (costante di modulo) | dict `AUGMENTATIONS`, 7 varianti + `none` | Selezionabili per nome, senza editare il codice. |
| `transforms.Lambda(lambda im: im.convert("RGB"))` | `ConvertRGB` (`@dataclass(frozen=True)`) | La lambda non è picklabile e romperebbe i DataLoader worker sotto spawn. Stesso pattern di `EMNISTFixTransform`. |
| `normalize` con stats OpenAI hardcoded, `224` hardcoded | `_extract_normalize()` / `_infer_input_size()` | Stats e risoluzione lette dal `preprocess` del backbone, così `laion`/`datacomp` e i modelli 336px restano corretti. Fallback su stats OpenAI e 224. |

**Non portato** da `finetune_svhn.py`: `DeltaPredictor` / `LinearDeltaPredictor`, le loss prediction + commitment + predicted-CE, i predittori per blocco e la valutazione ridge di feature predictability. Solo il preprocessing.

## Dove si innesta

Un solo aggancio, in `build_vision_loaders` (`src/merge_and_rebase/data/vision_loaders.py`), accanto al fix EMNIST già presente:

```python
train_transform, eval_preprocess = maybe_svhn_transforms(hf_path, train_transform, eval_preprocess)
```

È l'unico punto del repo in cui l'identità del dataset (`hf_path`) e il transform si incontrano, e tutti i call site la attraversano: `finetune/train_vision.py`, `finetune/regularizers/_vision_collection.py`, `eval/vision_merge.py`, `eval/vision_rebase.py`, `eval/vision_linear_probe.py`, `eval/vision_logit_kl.py`, `eval/vision_connectivity.py`, `eval/vision_block_extension.py`. Di conseguenza training ed eval condividono la stessa pipeline senza che nessuno di quei file sia stato toccato.

Con l'override spento `maybe_svhn_transforms` restituisce gli argomenti invariati: comportamento identico a prima per ogni dataset, SVHN compreso.

## Come si accende

Dal più volatile al più permanente:

```bash
# una tantum
MR_SVHN_PREPROCESS=1 python -m merge_and_rebase.finetune.train_vision \
    --vision-config src/merge_and_rebase/finetune/configs/vision.yaml --datasets SVHN

# sweep di augmentation, senza toccare file
MR_SVHN_PREPROCESS=1 MR_SVHN_AUGMENTATION=randaugment_3_5 python -m merge_and_rebase.finetune.train_vision \
    --vision-config src/merge_and_rebase/finetune/configs/vision.yaml --datasets SVHN

# stabile: SVHN_CUSTOM_PREPROCESS = True in src/merge_and_rebase/data/svhn_preprocess.py
```

Non c'è un knob nello yaml di proposito: gli entrypoint di eval non leggono il config del finetune, e una variabile d'ambiente è l'unico canale che li raggiunge tutti in modo uniforme.

| Variabile | Costante di modulo | Default | Valori |
|---|---|---|---|
| `MR_SVHN_PREPROCESS` | `SVHN_CUSTOM_PREPROCESS` | `False` | `1`/`true`/`yes`/`on` — `0`/`false`/`no`/`off` |
| `MR_SVHN_TARGET_SIZE` | `SVHN_TARGET_SIZE` | `96` | int, `<=` input size del backbone |
| `MR_SVHN_AUGMENTATION` | `SVHN_AUGMENTATION` | `autoaugment` | vedi tabella sotto |

## Augmentation disponibili

| nome | `AUGMENTATION_CONFIG` originale | contenuto |
|---|---|---|
| `affine_jitter_strong` | 0 | `RandomAffine(8°, translate .08, scale .9-1.1)` + `ColorJitter(.2/.2/.15/.03)` |
| `affine_jitter_light` | 1 | `RandomAffine(4°, translate .04, scale .95-1.05, shear ±5)` + `ColorJitter(.1/.1/.075/.015)` |
| `autoaugment` | 2 **(default, era l'attiva)** | `AutoAugment(policy=SVHN, BICUBIC, fill=0)` |
| `randaugment_2_5` | 3 | `RandAugment(num_ops=2, magnitude=5)` |
| `randaugment_3_5` | 4 | `RandAugment(num_ops=3, magnitude=5)` |
| `randaugment_3_7` | 5 | `RandAugment(num_ops=3, magnitude=7)` |
| `photometric` | 6 | `RandomAffine(8°, shear ±10)` + `RandomEqualize(.2)` + `RandomInvert(.1)` + `RandomAutocontrast(.2)` + `ColorJitter(.15/.15/.1/.02)` |
| `none` | — | nessuna: il transform di train coincide con quello di eval |

L'augmentation è applicata **solo** allo split di train, e sempre quando l'override è acceso — indipendentemente da `data.train_preprocess` nello yaml, esattamente come faceva il codice originale. Vede solo il riquadro 96×96 e mai il bordo, perché il pad viene dopo.

## ⚠️ La regola che conta

**Il preprocessing cambia la geometria dell'input, quindi la stessa variabile d'ambiente va passata anche agli entrypoint di eval.** Accendere l'override in training e dimenticarsene in `vision_merge` / `vision_rebase` / `vision_linear_probe` / `vision_logit_kl` / `vision_connectivity` / `vision_block_extension` valuta il checkpoint su una distribuzione di input diversa da quella su cui è stato addestrato, e produce numeri sbagliati **in silenzio**.

In uno script sbatch basta esportarla una volta in testa:

```bash
export MR_SVHN_PREPROCESS=1
export MR_SVHN_AUGMENTATION=autoaugment

python -m merge_and_rebase.finetune.train_vision --vision-config ... --datasets SVHN
python -m merge_and_rebase.eval.vision_merge --config ...
```

Per lo stesso motivo i checkpoint SVHN prodotti prima dell'override non sono confrontabili con quelli prodotti dopo.

## Come verificare che sia attivo

1. A video compare una volta sola, alla costruzione dei loader:
   ```
   [SVHN] custom preprocess: target=96 input=224 augmentation=autoaugment
   ```
2. La zero-shot accuracy su SVHN **cambia** rispetto a un run con l'override spento (geometria diversa: è il segnale che sta agendo).
3. Controllo di coerenza train/eval: la test accuracy riportata a fine finetuning deve combaciare con la single-task accuracy che `vision_merge` (o `vision_linear_probe`) riporta sullo stesso checkpoint, lanciato con la stessa variabile d'ambiente. Una discrepanza grossa significa che un path di eval non sta passando per `maybe_svhn_transforms`.

Nota su `eval/vision_linear_probe.py`: fitta la testa su `loaders.train`, che con l'override acceso risulta augmentato. Per un probe deterministico usare `MR_SVHN_AUGMENTATION=none`.

## Testing

`tests/test_svhn_preprocess.py` — nessun modello, nessuna rete, nessuna GPU: immagine PIL sintetica e un finto `Compose` in stile open_clip con stats deliberatamente non-OpenAI. Copre il no-op a override spento e su altri dataset (è il test che protegge la non-regressione), la geometria del pad, il bordo costante a `-mean/std`, l'ereditarietà delle stats di normalizzazione, `none` == eval, gli errori su nome di augmentation sconosciuto e `target_size` troppo grande, e la picklabilità (fallisce se qualcuno rimette una `transforms.Lambda`).

```bash
pytest tests/test_svhn_preprocess.py -v
```

**Non ancora fatto:** un run reale end-to-end con l'override acceso, per confrontare la test accuracy finale col numero ottenuto in `rebasin_linear`.
