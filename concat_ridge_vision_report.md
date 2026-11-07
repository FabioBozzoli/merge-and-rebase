# Report: strategia Stage 2 `concat_ridge` (implementazione vision)

Riferimento: commit `d7c8b54` sul branch `steer4rebase` ("add concat_ridge stage 2 strategy").
Il commit precedente è `b018158`, e `git show d7c8b54` riporta il diff completo.

Questo documento descrive **solo** le modifiche fatte al codice di visione
(`src/merge_and_rebase/rebase/methods/steer.py` e i file collegati). L'unico ritocco a
`steer_text.py` è descritto al §6, ed è solo un adattamento di firma.

Scopo: permettere a un'altra sessione di portare la stessa strategia sul testo.

---

## 1. Cosa fa `concat_ridge`, in sintesi matematica

Le strategie Stage 2 già presenti erano due:
- **`global_ridge`**: una ridge `f_B[sel] → train_target`, che usa solo la feature finale di B (L2-normalizzata).
- **`block_ridge`**: una ridge per blocco, con target per-blocco costruiti da `delta_A_blocks`. Funziona solo in regime `linear`.

`concat_ridge` è **una singola ridge** il cui input è la concatenazione delle attivazioni pooled di **tutti** i blocchi di B, **inclusa la feature finale**:

```
X = [ h_1/s_1 , h_2/s_2 , … , h_L/s_L , f_B/s_{L+1} ]     (righe = support set selezionato)
s_b = sqrt( ||X_b||_F^2 / n )                             (calcolato sul support set, poi congelato)
W   = ridge(X, train_target, λ_eff)
correzione(x) = Σ_b  h_b(x) @ (W[rows_b] / s_b)
```

**Target**:
- È **identico** a quello di `global_ridge`: `train_target = delta_A[sel] @ logit_map.T @ pinv(W_B).T`. Il codice esistente lo calcola già prima del ramo Stage 2, e non è stato toccato.
- Non servono delta per-blocco. Per questo `concat_ridge` funziona **sia in regime `linear` sia in regime `standard`**.

**Blocchi**:
- Sono quelli prodotti dalla capture esistente `_activation_capture_for_visual(visual, block_granularity, token_pooling)`.
- Con `block_granularity="residual"` i blocchi sono `L+1`:
  - gli id `0..L-1` sono gli output delle resblock (pooled, raw);
  - l'id `L` è la feature finale L2-normalizzata (`final_ids=(L,)`, riempito da `_complete_captured_activations`).
- La feature finale è quindi già inclusa come ultimo blocco: non va aggiunta a parte.
- `global_ridge` è un caso annidato: corrisponde a usare solo le colonne di `f_B`.

**Scaling per blocco**:
- È **sempre attivo** e non c'è un flag per disattivarlo.
- Serve perché le attivazioni residue raw crescono di norma con la profondità, mentre la feature finale ha norma 1. Senza scaling, un λ unico regolarizzerebbe i blocchi in modo molto diseguale.

**Regolarizzazione**:
- Si usa il parametro esistente `ridge_lambda`.
- Se `block_ridge_lambda_scaling == "trace"`, vale `λ_eff = ridge_lambda · tr(XXᵀ)/n` della concatenazione già scalata. Dopo lo scaling questo valore è circa uguale al numero di blocchi.
- Con `"none"`, che è il default usato nella griglia, vale `λ_eff = ridge_lambda`.

**Equivalenza duale**:
- Siccome `n ≤ D`, `_ridge` usa automaticamente la forma duale, cioè un sistema `n×n` in float64.
- Il metodo equivale quindi a una kernel ridge con kernel `K(x,x') = Σ_b <h_b(x), h_b(x')> / s_b²`.
- Dimensioni dell'input: D = 12·768 + 512 = 9728 per ViT-B/16, e 24·1024 + 768 = 25344 per ViT-L/14.

**Grouping**:
- Non si applica. `block_group_strategy` viene ignorato, e lo stesso vale per `block_ridge_mode`/`rho`.
- Il numero di blocchi di B è quindi libero: funziona anche cross-arch (ViT-B → ViT-L) senza vincoli di rapporto di profondità.

---

## 2. Nuova funzione di fit: `_fit_concat_ridge` (steer.py, subito dopo `_predict_block_ridge`)

```python
def _fit_concat_ridge(
    blocks_train: Mapping[int, torch.Tensor],
    train_target: torch.Tensor,
    *,
    regularization: float,
    regularization_scaling: str = "none",
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if regularization_scaling not in {"none", "trace"}:
        raise ValueError(f"Unknown steer concat_ridge regularization_scaling: {regularization_scaling}")
    block_ids = sorted(blocks_train)
    if block_ids != list(range(len(block_ids))):
        raise ValueError(f"steer concat_ridge expects contiguous block ids from 0, got {block_ids}")
    n = train_target.shape[0]
    scales = [
        float(blocks_train[b].square().sum().div(n).sqrt().clamp_min(1e-12)) for b in block_ids
    ]
    features = torch.cat([blocks_train[b] / scale for b, scale in zip(block_ids, scales)], dim=1)
    effective_regularization = regularization
    if regularization_scaling == "trace":
        effective_regularization = regularization * float(features.square().sum()) / n
    coefficient = _ridge(features, train_target, effective_regularization)
    coefficients: list[torch.Tensor] = []
    offset = 0
    for b, scale in zip(block_ids, scales):
        width = blocks_train[b].shape[1]
        coefficients.append(coefficient[offset : offset + width] / scale)   # scaling incorporato
        offset += width
    info = {"input_dim": int(features.shape[1]), "block_scales": scales, "effective_lambda": effective_regularization}
    return coefficients, info
```

Punti chiave:
- L'input `blocks_train` contiene **già solo le righe del support set**, in float64.
- La funzione restituisce una lista di coefficienti per blocco con lo scaling già incorporato (`W_b' = W[rows_b]/s_b`). La predizione `Σ_b X_b_raw @ W_b'` è quindi matematicamente identica a `X_scaled @ W`.
- Si può usare `_predict_block_ridge(coefficients, blocks)` già esistente. In `prepare` però la predizione è scritta inline, in streaming per blocco (§4.6).

---

## 3. Cache delle feature: modifiche alle funzioni di supporto

### 3.1 `_load_cached_split`
Prima la firma era `need_blocks: bool`, che richiedeva sia `delta_A_blocks` sia `features_B_blocks`. Ora i due flag sono separati:

```python
def _load_cached_split(cache_dir, *, need_source_blocks: bool, need_target_blocks: bool):
    required = ["features_A", "delta_A", "features_B", "y_A"]
    if need_source_blocks: required.append("delta_A_blocks")      # serve solo a block_ridge
    if need_target_blocks: required.append("features_B_blocks")   # block_ridge e concat_ridge
```

### 3.2 `_load_or_compute_split`
- Il kwarg `need_blocks` è stato sostituito da `need_source_blocks` + `need_target_blocks`, che vengono passati a `_load_cached_split`.
- Dopo `compute_fn()` si salvano **solo i tensori nuovi**. Se il compute ha riusato tensori dalla **stessa** cache dir, li segnala con la chiave speciale `_reused_from`, e quei tensori non vengono riscritti:
  ```python
  reused_dir, reused_keys = data.pop("_reused_from", (None, ()))
  to_save = {k: v for k, v in data.items() if not (reused_dir == cache_dir and k in reused_keys)}
  _save_split_cache(cache_dir, to_save)
  ```

### 3.3 `_save_split_cache`: scrittura atomica
Serve perché più job concorrenti condividono la stessa cache. Ogni file viene scritto su un file temporaneo e poi rinominato:

```python
tmp = cache_dir / f".{name}.pt.{os.getpid()}.tmp"
torch.save(value, tmp)
os.replace(tmp, cache_dir / f"{name}.pt")
```

Per questo è stato aggiunto `import os` in testa al file.

### 3.4 `_reuse_mean_pooled_split`
- Nuovo kwarg `need_source_blocks: bool = True`, passato a `_load_cached_split(..., need_source_blocks=need_source_blocks, need_target_blocks=False)`. Con `False` accetta una cache standard, che non ha `delta_A_blocks`.
- Sul risultato imposta `reused["_reused_from"] = (mean_cache_dir, tuple(cached))`, così `_load_or_compute_split` non riscrive i file già presenti (§3.2).
- La validazione è invariata: label identiche e `features_B` ricalcolata allclose a quella in cache, altrimenti restituisce `None` e si fa il fallback all'estrazione completa.

### 3.5 `_collect_standard_split`: cattura opzionale dei blocchi di B
- Nuovi kwarg: `block_granularity: str | None = None` e `token_pooling: str = "mean"`.
- Con `block_granularity` non `None`, nello **stesso forward** di B usa `_activation_capture_for_visual` + `_target_batch_activations` per raccogliere i blocchi.
- Aggiunge al dict restituito `features_B_blocks: dict[int, Tensor[N, d_b]]`.
- Il tipo di ritorno passa da `dict[str, Tensor]` a `dict[str, Any]`.

### 3.6 Lettura dei blocchi di B da un'altra cache: `_attach_external_target_blocks` (nuova)
- **Motivazione**: B (pretrained) è lo stesso modello nei regimi linear e standard, quindi le sue attivazioni intermedie non dipendono dal regime di A.
- **Uso**: in regime standard si possono leggere i `features_B_blocks.pt` già presenti nella cache linear, senza estrarre nulla e senza scrivere nulla.

```python
_EXTERNAL_BLOCKS_MIN_COSINE = 0.995

def _attach_external_target_blocks(data: dict[str, Any], external_dir: Path, *, verbose: bool) -> None:
    # richiede features_B_blocks.pt, features_B.pt, y_A.pt in external_dir (altrimenti FileNotFoundError)
    # label: torch.equal(external y_A, data["y_A"])                          altrimenti ValueError
    # shape di features_B identica                                             altrimenti ValueError
    # cos riga-per-riga(features_B esterna, data["features_B"]).min() >= 0.995 altrimenti ValueError("... do not match ...")
    data["features_B_blocks"] = torch.load(external_dir / "features_B_blocks.pt", map_location="cpu", weights_only=True)
```

La soglia di 0.995 ha questa giustificazione empirica:
- feature ricalcolate su GPU o kernel diversi danno `1-cos ≈ 1e-5`;
- su ViT-L la riga peggiore misurata è 0.9989;
- un modello, una transform o un ordine di esempi diverso darebbe similarità molto più basse.

Si fa un log con il cos minimo e la media di `1-cos`.

---

## 4. Modifiche in `SteerRebase.prepare`

### 4.1 Nuovi parametri (in `method_params`)
```python
target_blocks_cache_dir: str | None = None,   # root di un'altra feature cache da cui leggere features_B_blocks
target_blocks_regime: str = "linear",         # regime (nome sottocartella) di quella cache
```

### 4.2 Validazione
- `stage_2_strategy` ammette anche `"concat_ridge"`, e il messaggio d'errore è stato aggiornato.
- **Nessun vincolo sul regime**: il controllo "block_ridge requires linear" resta solo per `block_ridge`.

### 4.3 Flag dei blocchi
Il flag `need_blocks` è stato sostituito da:

```python
need_source_blocks = stage_2_strategy == "block_ridge"
need_target_blocks = stage_2_strategy in {"block_ridge", "concat_ridge"}
cache_pooling = token_pooling if need_target_blocks else "mean"
external_target_blocks = (
    need_target_blocks and feature_regime == "standard" and target_blocks_cache_dir is not None
)
load_target_blocks = need_target_blocks and not external_target_blocks
```

- Il warning "token_pooling='cls' has no effect" ora scatta con `if not need_target_blocks`, mentre prima scattava con `stage_2_strategy != "block_ridge"`.
- Le due chiamate a `_load_or_compute_split` (train e test) passano `need_source_blocks=need_source_blocks` e `need_target_blocks=load_target_blocks`.

### 4.4 `_compute` in regime standard
Quando `load_target_blocks` è vero, cioè senza cache esterna:
1. Se non c'è `force_recompute_features`, prova `_reuse_mean_pooled_split` sulla cache standard, mean-pooled e possibilmente la stessa dir. Come `target_part` usa `_collect_target_blocks_split(...)` e passa `need_source_blocks=False`. Così aggiunge solo `features_B_blocks.pt` senza rifare i forward di A.
2. Altrimenti esegue l'estrazione completa con `_collect_standard_split(..., block_granularity=block_granularity if load_target_blocks else None, token_pooling=token_pooling)`.

### 4.5 Cache esterna
Dopo il caricamento di train e test, se `external_target_blocks` è vero:

```python
for split_name, data in ((train_split, train_data), ("test", test_data)):
    _attach_external_target_blocks(
        data,
        _cache_split_dir(target_blocks_cache_dir, source_tag, target_tag, task,
                         target_blocks_regime, split_name, block_granularity, cache_pooling),
        verbose=verbose,
    )
```

Il path segue lo stesso layout della cache principale: `<root>/<src>_to_<tgt>/<task>/<regime>[...]/<split>`.

### 4.6 Nuovo ramo Stage 2 (tra `global_mlp` e `block_ridge`)
```python
elif stage_2_strategy == "concat_ridge":
    # selezione delle righe PRIMA dell'upcast; pop per liberare memoria (GB su ViT-L)
    features_b_blocks_train = {b: v[selected].double() for b, v in train_data.pop("features_B_blocks").items()}
    num_target_blocks_total = len(features_b_blocks_train)
    concat_dim = sum(v.shape[1] for v in features_b_blocks_train.values())
    if verbose and concat_dim > 50_000:
        print(f"... WARNING: concat_ridge input has {concat_dim} dims; X needs "
              f"{int(selected.numel()) * concat_dim * 8 / 2**30:.1f} GiB in float64.")
    fitted, concat_info = _fit_concat_ridge(
        features_b_blocks_train, train_target,
        regularization=ridge_lambda,
        regularization_scaling=block_ridge_lambda_scaling,
    )
    coefficients = [c.to(dev) for c in fitted]
    # log: numero blocchi, input dim, lambda_eff, block scales

    def correction_fn(activations, *, _coefficients=coefficients):
        out = None
        for block_id, coefficient in enumerate(_coefficients):   # un blocco alla volta
            contribution = activations["blocks"][block_id].to(coefficient.device).double() @ coefficient
            out = contribution if out is None else out + contribution
        return out.to(dtype=activations["global"].dtype, device=activations["global"].device)
```

### 4.7 Diagnostica in cached space
```python
if need_target_blocks:
    cached_test_activations["blocks"] = dict(test_data["features_B_blocks"])   # niente .double() anticipato
```

- Prima c'era `{b: v.double() ...}`. Ora ogni `correction_fn` fa l'upcast per blocco, il che evita di raddoppiare la memoria su ViT-L.
- `stage2_test_acc` in cached space è calcolato come per le altre strategie. Nella griglia ha coinciso con la valutazione live (α=1) entro circa 0.3 punti.

---

## 5. `steer_correction_context` (eval live con hook)
```python
needs_blocks = prepared["stage_2_strategy"] in {"block_ridge", "concat_ridge"}
```
Il resto è invariato:
- gli hook catturano `activations["blocks"]` con la stessa granularità e lo stesso pooling usati nel fit;
- `_complete_captured_activations` mette la feature finale normalizzata nell'id `L`;
- la correzione viene sommata all'output L2-normalizzato di `encode_image` moltiplicata per alpha.

---

## 6. Adattamento minimo già fatto in `steer_text.py`
- La funzione condivisa `_load_or_compute_split` ha cambiato firma. Per questo in `steer_text.py` i `cache_args` ora passano `"need_source_blocks": need_blocks, "need_target_blocks": need_blocks` al posto di `"need_blocks": need_blocks`.
- È **solo** una compatibilità di firma: `concat_ridge` **non** è implementata sul testo.

Per il port sul testo vanno replicati, in modo analogo:
- §2: la funzione `_fit_concat_ridge` è riusabile direttamente, perché è agnostica alla modalità;
- §3.5–3.6: cattura dei blocchi di B in regime standard, o lettura da una cache esterna con la stessa validazione;
- §4: flag `need_source_blocks`/`need_target_blocks`/`load_target_blocks`, ramo Stage 2 e diagnostica;
- §5: hook live con `needs_blocks` esteso a `concat_ridge`.

---

## 7. Test aggiunti o modificati (`tests/test_steer_rebase.py`)
- Il test esistente di `_load_or_compute_split` è stato aggiornato ai nuovi kwarg.
- **`test_concat_ridge_matches_scaled_concat`**:
  - Usa blocchi sintetici con scale molto diverse.
  - Verifica che `_predict_block_ridge(_fit_concat_ridge(...))` coincida con `_ridge` sulla concatenazione scalata a mano.
  - Verifica che riscalare un blocco non cambi la predizione (invarianza dovuta allo scaling).
- **`test_concat_ridge_recovers_linear_map`**: con un target generato linearmente da più blocchi e λ piccolo, il fit recupera la mappa.
- **End-to-end** (`test_steer_prepare_and_correction_end_to_end`):
  - Aggiunti i casi `(concat_ridge, standard)` e `(concat_ridge, linear)`.
  - Il nome del task diventa `f"tiny_{strategy}_{regime}"`, per non condividere la cache tra i casi.
- **Helper aggiunti**:
  - `_prepare_tiny_steer(tmp_path, clfs, *, task, loaders=None, **params)`;
  - `_tiny_steer_models(target_depth=3)`.
  - Attenzione: `_tiny_loaders()` genera dati random nuovi a ogni chiamata, quindi per confrontare live e cached bisogna passare gli **stessi** loader.
- **`test_concat_ridge_adds_target_blocks_to_standard_cache`**:
  - Esegue prima `global_ridge` standard, poi `concat_ridge` sulla stessa cache.
  - Verifica che venga aggiunto solo `features_B_blocks.pt` e che gli mtime degli altri file restino invariati.
  - Verifica che la correzione live coincida con quella cached.
- **`test_concat_ridge_cross_arch_deeper_target`**: usa un target più profondo della sorgente, senza grouping.
- **`test_concat_ridge_standard_reads_target_blocks_from_linear_cache`**:
  - Verifica che la lettura da `target_blocks_cache_dir` funzioni.
  - Verifica che la cache standard non venga scritta (mtime invariati).
  - Verifica che una cache incoerente sollevi `ValueError("... do not match ...")`.
  - **Nota**: nel file questa funzione è definita **due volte** (righe ~854 e ~894). La seconda definizione sovrascrive la prima, quindi pytest esegue solo la seconda. La prima è codice morto da rimuovere, ma i due test verificano le stesse proprietà. Nel port sul testo va scritta una sola versione.

Stato: 63 test passano (steer + text).

Comando: `CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/src .venv/bin/pytest tests/test_steer_rebase.py`. Sul login node la GPU non è supportata da torch 2.11, quindi si gira su CPU.

---

## 8. Config, script e documentazione (vision)
- **`configs/vision8_steer_concat/make_configs.py`**:
  - Genera `concatridge_{linear,nonlinear}_{vitb,vitl}_fs{K}_seed{S}[_lamX].json`.
  - Importa `BASES/FEATURES/KEEP/SEEDS` da `configs/vision8_steer_global/make_configs.py`.
  - `method_params`: `stage_2_strategy="concat_ridge"`, `block_granularity="residual"`, `token_pooling="mean"`, `ridge_lambda=1.0` (modificabile con `--ridge-lambda`), `stage1_lambda=1.0`, `alpha=1` fisso, `force_recompute_features=False`.
  - Nel regime nonlinear aggiunge `target_blocks_cache_dir=/work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature` e `target_blocks_regime="linear"`.
- **`scripts/slurm/vision_concatridge.sh <linear|nonlinear> <vitb|vitl> <Dataset>`**:
  - L'indice dell'array è K; i 3 seed vengono eseguiti in sequenza.
  - Le run già completate (summary presente) vengono saltate.
  - `PYTHONPATH` punta allo snapshot del codice.
  - Imposta le env di SVHN (`MR_SVHN_PREPROCESS=1`, `MR_SVHN_TARGET_SIZE=96`, `MR_SVHN_AUGMENTATION=none`).
- **`scripts/slurm/submit_vision_concatridge.sh`**: risorse per coppia e dataset.
  - ViT-B: 8G e 40 minuti; SUN397 18G.
  - ViT-L: 12–24G e 1–1,5 ore.
  - RSS massimo osservato: 12.1 GB (ViT-L SUN397).
- **`steer4rebase_recap.md`**: aggiunte le righe su `concat_ridge` nelle tabelle dei parametri e delle strategie.
- **Analisi**:
  - `scripts/aggregate_steer_concat_comparison.py`: confronto con global/block ridge.
  - `scripts/analyze_stage1_energy.py` + `scripts/slurm/stage1_energy_ablation.sh`: ablation sulla soglia di energia della pinv di Stage 1. Esito: con `concat_ridge`, la differenza tra soglia 0.99 e pinv esatta è in media +0.03 punti, con massimo |0.6|; 0.99 resta il default.

---

## 9. Risultati di riferimento (vision, per validare un port)
Media sugli 8 dataset e sui 3 seed (33, 54, 89), accuratezza %, `ridge_lambda=1`, `stage1_lambda=1`, α=1:

| coppia | regime | K=1 | K=2 | K=5 | K=10 | K=20 | zero-shot |
|---|---|---|---|---|---|---|---|
| ViT-B/16 datacomp → ViT-B/16 laion2b | linear | 71.3 | 75.5 | 80.4 | 82.7 | 84.5 | 64.0 |
| | nonlinear | 71.1 | 73.5 | 77.2 | 79.4 | 81.0 | 64.0 |
| ViT-B/16 datacomp → ViT-L/14 datacomp | linear | 78.0 | 81.0 | 84.3 | 85.8 | 87.2 | 71.9 |
| | nonlinear | 76.5 | 78.6 | 81.2 | 82.6 | 84.0 | 71.9 |

In regime nonlinear, `concat_ridge` supera `global_ridge` di 3–4 punti in modo consistente.

**Cautela per il port**: a parità di `ridge_lambda`, la regolarizzazione relativa è circa (L+1) volte più debole rispetto alla global ridge, perché ogni blocco scalato ha norma quadratica media 1. Un eventuale sweep di λ non è ancora stato fatto.
