# t5-base (NTK) → roberta-base: test accuracy, mean over seeds 33/54/89, per dataset

Mean over the three seeds of `roberta_seed_table.md`, 4 decimals. `sick` has no 1000-shot column. The `AVG` row is the mean over the datasets (5 datasets at 1000 shots).

## MNLI

| Method | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|
| Zero-shot | 0.3772 | 0.3772 | 0.3772 | 0.3772 |
| Linear probing | 0.4352 | 0.4600 | 0.4752 | 0.4991 |
| Steering global MLP | 0.4696 | 0.4956 | 0.5059 | 0.5343 |
| Steering attention+segments | 0.5793 | 0.5983 | 0.6063 | 0.6393 |

## QNLI

| Method | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|
| Zero-shot | 0.6322 | 0.6322 | 0.6322 | 0.6322 |
| Linear probing | 0.6513 | 0.6687 | 0.6909 | 0.6911 |
| Steering global MLP | 0.6800 | 0.7020 | 0.7128 | 0.7331 |
| Steering attention+segments | 0.7731 | 0.7804 | 0.7950 | 0.8046 |

## RTE

| Method | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|
| Zero-shot | 0.5462 | 0.5462 | 0.5462 | 0.5462 |
| Linear probing | 0.5181 | 0.5462 | 0.5368 | 0.5422 |
| Steering global MLP | 0.5301 | 0.5422 | 0.5422 | 0.5448 |
| Steering attention+segments | 0.6118 | 0.6198 | 0.6212 | 0.6386 |

## SCITAIL

| Method | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|
| Zero-shot | 0.6717 | 0.6717 | 0.6717 | 0.6717 |
| Linear probing | 0.7231 | 0.7413 | 0.7567 | 0.7759 |
| Steering global MLP | 0.7044 | 0.7172 | 0.7243 | 0.7443 |
| Steering attention+segments | 0.7887 | 0.7980 | 0.8198 | 0.8363 |

## SNLI

| Method | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|
| Zero-shot | 0.4572 | 0.4572 | 0.4572 | 0.4572 |
| Linear probing | 0.4957 | 0.5007 | 0.5015 | 0.5111 |
| Steering global MLP | 0.5181 | 0.5380 | 0.5485 | 0.5565 |
| Steering attention+segments | 0.7152 | 0.7348 | 0.7511 | 0.7683 |

## SICK

| Method | 200 | 300 | 500 |
|---|---|---|---|
| Zero-shot | 0.6683 | 0.6683 | 0.6683 |
| Linear probing | 0.7844 | 0.7869 | 0.7998 |
| Steering global MLP | 0.7696 | 0.7763 | 0.7752 |
| Steering attention+segments | 0.7778 | 0.7913 | 0.8004 |

## AVG

| Method | 200 | 300 | 500 | 1000 |
|---|---|---|---|---|
| Zero-shot | 0.5588 | 0.5588 | 0.5588 | 0.5369 |
| Linear probing | 0.6013 | 0.6173 | 0.6268 | 0.6039 |
| Steering global MLP | 0.6120 | 0.6285 | 0.6348 | 0.6226 |
| Steering attention+segments | 0.7076 | 0.7204 | 0.7323 | 0.7374 |
