# Segment-split pooling dei regressori per-blocco di B (steer_text block_ridge)

> Il piano precedente (`linear_probe_stage: "pre"` + sweep nli6) è implementato e pushato: commit `0478777` e `07ee8ca` su `fork/block_ridge_lambda_tuning`.

## Context

In tutti i task NLI l'input è una coppia (premise, hypothesis). Con la pair encoding di T5 la sequenza è `premise </s> hypothesis </s>` ([text_loaders.py:348](src/merge_and_rebase/data/text_loaders.py:348)): nessun `[SEP]`, nessun `token_type_ids`, e i 6 dataset passano tutti per lo stesso `NLIExample` con campi `premise`/`hypothesis`, quindi non c'è niente di per-dataset da gestire.

Le attivazioni per-blocco di B che Stage 2 usa come **regressori** vengono poolate con una media unica su tutti i token reali (`masked_mean`, via `_TextBlockCapture`). Le due frasi collassano in un solo vettore, e l'informazione su come si distribuiscono premise e ipotesi non arriva mai alla ridge.

Obiettivo: far sì che ogni blocco residuale di B contribuisca `[GAP_premise ; GAP_hypothesis ; GAP_global]` invece del solo `GAP_global`.

Vincoli decisi con l'utente:

- **la larghezza della testa non cambia.** La correzione viene sommata all'input della testa ([steer_text.py:812](src/merge_and_rebase/rebase/text/steer_text.py:812)), quindi la feature poolata di B resta `D`. Lo split riguarda solo i regressori.
- **si tocca solo il target.** A, `w_a`, Stage 1 e i `heads.pt` esistenti restano invariati.
- composizione a 3 vie: il GAP globale **non** è ricostruibile linearmente dai due segmenti (è la loro media pesata per lunghezze che variano per esempio), quindi includerlo rende il nuovo spazio un superset stretto della baseline — la ridge può sempre recuperare il comportamento attuale.
- opt-in, default invariato (stessa scelta del piano precedente).

## Il confine train/eval che regge tutto

In entrambi i percorsi il blocco di indice `num_target_residual` **è** la feature poalata globale, assegnata separatamente dalla capture:

| | blocchi residuali (→ 3D) | blocco di output (resta D) |
|---|---|---|
| train | `capture.activations[b]`, [steer_text.py:383](src/merge_and_rebase/rebase/text/steer_text.py:383) | `out_b` da `_pooled_features`, [:384](src/merge_and_rebase/rebase/text/steer_text.py:384) |
| eval | `capture.activations`, [:815](src/merge_and_rebase/rebase/text/steer_text.py:815) | `blocks[num_residual] = feature`, [:813](src/merge_and_rebase/rebase/text/steer_text.py:813) |

La separazione esiste già come due assegnazioni distinte in tutti e due i siti: non va introdotta, va solo rispettata.

## Change

### 1. Pooling per segmento — `rebase/text/encoder_classifier.py`

Accanto a `masked_mean` ([:62](src/merge_and_rebase/rebase/text/encoder_classifier.py:62)), due helper nello stesso modulo (è il posto che già dichiara di essere l'unica definizione del pooling):

- ricavare le maschere di segmento da `input_ids`: segmento 1 = token prima del **primo** eos, segmento 2 = token tra primo e secondo eos, eos esclusi da entrambi. Nessun token speciale nuovo, nessuna ri-tokenizzazione.
- il pooling a 3 vie che concatena `masked_mean` sulle due maschere più `masked_mean` sulla `attention_mask` piena, riusando `masked_mean` invece di reimplementarne la logica (incluso il suo `clamp_min(1.0)`, che evita la divisione per zero su un segmento vuoto).

Aggiornare il docstring di `masked_mean`: l'invariante "feature poolata e attivazioni per-blocco poolate dalla stessa funzione" smette di valere di proposito, ed è esattamente il motivo per cui il blocco di output resta `D`.

### 2. `_TextBlockCapture` — `rebase/text/steer_text.py:225`

Un attributo in più accanto a `attention_mask` (es. `input_ids`, `None` = comportamento attuale). Quando è valorizzato, `_make_hook` ([:234](src/merge_and_rebase/rebase/text/steer_text.py:234)) usa il pooling a segmenti invece di `_masked_mean`.

### 3. I due siti che popolano la capture

- **train**, [steer_text.py:379](src/merge_and_rebase/rebase/text/steer_text.py:379): assegnare `capture.input_ids = batch_b["input_ids"]` accanto a `capture.attention_mask`.
- **eval**, `_mask_hook` ([:801](src/merge_and_rebase/rebase/text/steer_text.py:801)): leggere `kwargs.get("input_ids")`. È registrato `with_kwargs=True` e `TextLM.sequence_classification_accuracy` chiama `self.model(input_ids=..., attention_mask=...)` ([text_lm.py:386](src/merge_and_rebase/models/text_lm.py:386)), quindi `input_ids` è nei kwargs. **Da verificare allo stesso modo per `adapters.train_linear_probe_head`**, che in `linear_probe_stage="post"` fa forward sotto il correction context: se lì `input_ids` fosse posizionale, `kwargs.get` darebbe `None` e il pooling tornerebbe silenziosamente globale.

`_collect_standard_split` non cattura blocchi e non va toccato (block_ridge richiede già `feature_regime="linear"`).

### 4. Chiave della cache — `steer_text.py:551`

`features_B_blocks` cambia shape e lo schema di pooling non è nella chiave. `cache_args["feature_regime"]` finisce solo in `_cache_split_dir`, che lo usa come segmento di path ([steer.py:689](src/merge_and_rebase/rebase/methods/steer.py:689)); `_compute` legge `feature_regime` dallo scope, non da lì. Quindi basta passare una stringa decorata (es. `f"{feature_regime}__segpool"`) quando l'opzione è attiva: nessuna modifica di firma, nessun impatto sul path vision.

### 5. Knob e validazione — `SteerTextRebase.prepare`, `steer_text.py:448`

Nuovo `method_params.target_block_pooling: "global" | "segments"`, default `"global"`. Validare accanto agli altri controlli ([:480](src/merge_and_rebase/rebase/text/steer_text.py:480)):

- richiede `stage_2_strategy="block_ridge"` (con `global_ridge` esistono solo `activations["global"]`, l'opzione sarebbe un no-op silenzioso);
- richiede `model_kind="encoder_classification"` (il path decoder pools sull'eos, dove lo split non ha significato);
- incompatibile con `premise_hypothesis_template` valorizzato (stringa unica → un solo eos → secondo segmento vuoto).

Registrare `target_block_pooling` nei `diagnostics`/`artifacts` accanto a `block_group_strategy` ([:686](src/merge_and_rebase/rebase/text/steer_text.py:686)).

Sul boundary mancante (meno di due eos in una riga): **sollevare**, non degradare a globale. Sui 6 task con pair encoding un boundary assente significa che qualcosa è rotto a monte, e un fallback silenzioso produrrebbe numeri plausibili e sbagliati.

## Cosa NON va toccato, e perché

Verificato leggendo il codice:

- `_fit_block_ridge` e `_predict_block_ridge` ([steer.py:265](src/merge_and_rebase/rebase/methods/steer.py:265), [:285](src/merge_and_rebase/rebase/methods/steer.py:285)) indicizzano `blocks_train[block_id]` e fittano una ridge indipendente per blocco: larghezze diverse fra indici sono già supportate. I `train_targets` restano uniformi a `d_B`.
- `_group_blocks_concat` ([steer.py:294](src/merge_and_rebase/rebase/methods/steer.py:294)) fa `torch.cat(..., dim=1)` su blocchi residuali, tutti alla stessa larghezza → gruppi uniformi a `6D`. Il blocco di output non viene mai raggruppato: è assegnato a parte a `grouped_train[num_source_residual_blocks]` ([steer_text.py:666](src/merge_and_rebase/rebase/text/steer_text.py:666)).
- Stage 1, `w_a`/`w_b`, `logit_map`, la pinv, la testa, i `heads.pt`, il path vision, `text_rebase.py`: invariati.

Costo: gram per blocco residuale da `2D×2D` a `6D×6D` (6144 per t5-large — irrilevante); la cache delle feature per-blocco triplica in dimensione.

## Test

Un test, in `tests/test_text_encoder_classifier.py` (dove vive già la copertura del pooling encoder-only):

- su un batch costruito a mano con due lunghezze di segmento diverse e padding, verificare che le tre fette del vettore a 3 vie corrispondano a `masked_mean` calcolata sulle rispettive maschere, e che il padding non entri in nessuna delle tre.

Aggiungere in `tests/test_steer_rebase.py`, accanto a `test_block_ridge_fit_predict_roundtrip` ([:66](tests/test_steer_rebase.py:66)), un caso con larghezze per blocco **disomogenee** (residuali larghi, output stretto) che fa roundtrip fit→predict: è l'assunto strutturale su cui poggia tutta la modifica.

## Verifica end-to-end

1. **Non-regressione**: run breve senza il knob → `target_block_pooling="global"`. Le shape di `features_B_blocks` e l'accuracy finale devono coincidere con una run precedente. Bit-identico è l'obiettivo.
2. **La cache non si mescola**: due run consecutive, stesso task, una con e una senza il knob. Devono comparire due directory di cache distinte sotto `feature_cache_dir` e nessuna delle due deve stampare `using cached features` per l'altra.
3. **Smoke segments**: 1 task (`rte`), `max_samples_per_task` piccolo, `linear_probe_epochs: 20`. Verificare nei log che `stage2_test_acc` venga calcolato senza errori di shape e che la width dei regressori sia quella attesa.
4. **Train/eval coerenti**: è il rischio principale. `stage2_test_acc` è calcolato in spazio cached ([steer_text.py:732](src/merge_and_rebase/rebase/text/steer_text.py:732)) mentre la colonna `rebased` passa dall'hook live. Se il pooling a segmenti è attivo da una parte sola i due numeri divergono in modo netto: confrontarli è il controllo che intercetta un `input_ids` non arrivato all'hook.
5. **Confronto**: stesso config `global` vs `segments` su `rte` e `mnli` — è il numero che motiva la modifica.
