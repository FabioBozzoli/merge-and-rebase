# t5-base (NTK) → roberta-base: test accuracy per dataset and seed

Test accuracy, 4 decimals, one section per dataset: a row per (few-shot, method), a column per seed and the mean over seeds. Zero-shot is the fixed nearest-mean RoBERTa head (same for every seed); the linear probe picks its lr on val. `sick` has no 1000-shot row (only 606 class-2 rows in the train pool). The last section is the mean over the datasets, per seed.

## MNLI

| Shots | Method | Seed 33 | Seed 54 | Seed 89 | Mean |
|---|---|---|---|---|---|
| 200 | Zero-shot | 0.3772 | 0.3772 | 0.3772 | 0.3772 |
| 200 | Linear probing | 0.4289 | 0.4400 | 0.4367 | 0.4352 |
| 200 | Steering global MLP | 0.4650 | 0.4639 | 0.4800 | 0.4696 |
| 200 | Steering attention+segments | 0.5872 | 0.5800 | 0.5706 | 0.5793 |
| 300 | Zero-shot | 0.3772 | 0.3772 | 0.3772 | 0.3772 |
| 300 | Linear probing | 0.4694 | 0.4522 | 0.4583 | 0.4600 |
| 300 | Steering global MLP | 0.4856 | 0.5033 | 0.4978 | 0.4956 |
| 300 | Steering attention+segments | 0.6056 | 0.5917 | 0.5978 | 0.5983 |
| 500 | Zero-shot | 0.3772 | 0.3772 | 0.3772 | 0.3772 |
| 500 | Linear probing | 0.4844 | 0.4689 | 0.4722 | 0.4752 |
| 500 | Steering global MLP | 0.5039 | 0.5150 | 0.4989 | 0.5059 |
| 500 | Steering attention+segments | 0.6133 | 0.6028 | 0.6028 | 0.6063 |
| 1000 | Zero-shot | 0.3772 | 0.3772 | 0.3772 | 0.3772 |
| 1000 | Linear probing | 0.5011 | 0.4961 | 0.5000 | 0.4991 |
| 1000 | Steering global MLP | 0.5383 | 0.5433 | 0.5211 | 0.5343 |
| 1000 | Steering attention+segments | 0.6461 | 0.6394 | 0.6322 | 0.6393 |

## QNLI

| Shots | Method | Seed 33 | Seed 54 | Seed 89 | Mean |
|---|---|---|---|---|---|
| 200 | Zero-shot | 0.6322 | 0.6322 | 0.6322 | 0.6322 |
| 200 | Linear probing | 0.6544 | 0.6544 | 0.6450 | 0.6513 |
| 200 | Steering global MLP | 0.6811 | 0.6756 | 0.6833 | 0.6800 |
| 200 | Steering attention+segments | 0.7717 | 0.7700 | 0.7778 | 0.7731 |
| 300 | Zero-shot | 0.6322 | 0.6322 | 0.6322 | 0.6322 |
| 300 | Linear probing | 0.6683 | 0.6683 | 0.6694 | 0.6687 |
| 300 | Steering global MLP | 0.7083 | 0.6961 | 0.7017 | 0.7020 |
| 300 | Steering attention+segments | 0.7861 | 0.7756 | 0.7794 | 0.7804 |
| 500 | Zero-shot | 0.6322 | 0.6322 | 0.6322 | 0.6322 |
| 500 | Linear probing | 0.6939 | 0.6839 | 0.6950 | 0.6909 |
| 500 | Steering global MLP | 0.7211 | 0.7017 | 0.7156 | 0.7128 |
| 500 | Steering attention+segments | 0.7944 | 0.7939 | 0.7967 | 0.7950 |
| 1000 | Zero-shot | 0.6322 | 0.6322 | 0.6322 | 0.6322 |
| 1000 | Linear probing | 0.6906 | 0.6917 | 0.6911 | 0.6911 |
| 1000 | Steering global MLP | 0.7411 | 0.7300 | 0.7283 | 0.7331 |
| 1000 | Steering attention+segments | 0.8050 | 0.8033 | 0.8056 | 0.8046 |

## RTE

| Shots | Method | Seed 33 | Seed 54 | Seed 89 | Mean |
|---|---|---|---|---|---|
| 200 | Zero-shot | 0.5462 | 0.5462 | 0.5462 | 0.5462 |
| 200 | Linear probing | 0.5020 | 0.5382 | 0.5141 | 0.5181 |
| 200 | Steering global MLP | 0.5221 | 0.5462 | 0.5221 | 0.5301 |
| 200 | Steering attention+segments | 0.6386 | 0.5823 | 0.6145 | 0.6118 |
| 300 | Zero-shot | 0.5462 | 0.5462 | 0.5462 | 0.5462 |
| 300 | Linear probing | 0.5382 | 0.5622 | 0.5382 | 0.5462 |
| 300 | Steering global MLP | 0.5422 | 0.5823 | 0.5020 | 0.5422 |
| 300 | Steering attention+segments | 0.6627 | 0.5944 | 0.6024 | 0.6198 |
| 500 | Zero-shot | 0.5462 | 0.5462 | 0.5462 | 0.5462 |
| 500 | Linear probing | 0.4980 | 0.5944 | 0.5181 | 0.5368 |
| 500 | Steering global MLP | 0.5181 | 0.5542 | 0.5542 | 0.5422 |
| 500 | Steering attention+segments | 0.6024 | 0.6426 | 0.6185 | 0.6212 |
| 1000 | Zero-shot | 0.5462 | 0.5462 | 0.5462 | 0.5462 |
| 1000 | Linear probing | 0.5622 | 0.5542 | 0.5100 | 0.5422 |
| 1000 | Steering global MLP | 0.5221 | 0.5502 | 0.5622 | 0.5448 |
| 1000 | Steering attention+segments | 0.6185 | 0.6506 | 0.6466 | 0.6386 |

## SCITAIL

| Shots | Method | Seed 33 | Seed 54 | Seed 89 | Mean |
|---|---|---|---|---|---|
| 200 | Zero-shot | 0.6717 | 0.6717 | 0.6717 | 0.6717 |
| 200 | Linear probing | 0.7006 | 0.7444 | 0.7244 | 0.7231 |
| 200 | Steering global MLP | 0.6922 | 0.7311 | 0.6900 | 0.7044 |
| 200 | Steering attention+segments | 0.7917 | 0.7844 | 0.7900 | 0.7887 |
| 300 | Zero-shot | 0.6717 | 0.6717 | 0.6717 | 0.6717 |
| 300 | Linear probing | 0.7389 | 0.7544 | 0.7306 | 0.7413 |
| 300 | Steering global MLP | 0.7072 | 0.7317 | 0.7128 | 0.7172 |
| 300 | Steering attention+segments | 0.8044 | 0.8028 | 0.7867 | 0.7980 |
| 500 | Zero-shot | 0.6717 | 0.6717 | 0.6717 | 0.6717 |
| 500 | Linear probing | 0.7422 | 0.7600 | 0.7678 | 0.7567 |
| 500 | Steering global MLP | 0.7117 | 0.7289 | 0.7322 | 0.7243 |
| 500 | Steering attention+segments | 0.8278 | 0.8156 | 0.8161 | 0.8198 |
| 1000 | Zero-shot | 0.6717 | 0.6717 | 0.6717 | 0.6717 |
| 1000 | Linear probing | 0.7728 | 0.7783 | 0.7767 | 0.7759 |
| 1000 | Steering global MLP | 0.7383 | 0.7478 | 0.7467 | 0.7443 |
| 1000 | Steering attention+segments | 0.8339 | 0.8439 | 0.8311 | 0.8363 |

## SNLI

| Shots | Method | Seed 33 | Seed 54 | Seed 89 | Mean |
|---|---|---|---|---|---|
| 200 | Zero-shot | 0.4572 | 0.4572 | 0.4572 | 0.4572 |
| 200 | Linear probing | 0.4939 | 0.4939 | 0.4994 | 0.4957 |
| 200 | Steering global MLP | 0.5278 | 0.5261 | 0.5006 | 0.5181 |
| 200 | Steering attention+segments | 0.7178 | 0.7183 | 0.7094 | 0.7152 |
| 300 | Zero-shot | 0.4572 | 0.4572 | 0.4572 | 0.4572 |
| 300 | Linear probing | 0.5133 | 0.4878 | 0.5011 | 0.5007 |
| 300 | Steering global MLP | 0.5494 | 0.5261 | 0.5383 | 0.5380 |
| 300 | Steering attention+segments | 0.7461 | 0.7233 | 0.7350 | 0.7348 |
| 500 | Zero-shot | 0.4572 | 0.4572 | 0.4572 | 0.4572 |
| 500 | Linear probing | 0.5128 | 0.4906 | 0.5011 | 0.5015 |
| 500 | Steering global MLP | 0.5472 | 0.5506 | 0.5478 | 0.5485 |
| 500 | Steering attention+segments | 0.7539 | 0.7489 | 0.7506 | 0.7511 |
| 1000 | Zero-shot | 0.4572 | 0.4572 | 0.4572 | 0.4572 |
| 1000 | Linear probing | 0.5106 | 0.5150 | 0.5078 | 0.5111 |
| 1000 | Steering global MLP | 0.5522 | 0.5572 | 0.5600 | 0.5565 |
| 1000 | Steering attention+segments | 0.7683 | 0.7639 | 0.7728 | 0.7683 |

## SICK

| Shots | Method | Seed 33 | Seed 54 | Seed 89 | Mean |
|---|---|---|---|---|---|
| 200 | Zero-shot | 0.6683 | 0.6683 | 0.6683 | 0.6683 |
| 200 | Linear probing | 0.7894 | 0.7717 | 0.7922 | 0.7844 |
| 200 | Steering global MLP | 0.7589 | 0.7639 | 0.7861 | 0.7696 |
| 200 | Steering attention+segments | 0.7756 | 0.7689 | 0.7889 | 0.7778 |
| 300 | Zero-shot | 0.6683 | 0.6683 | 0.6683 | 0.6683 |
| 300 | Linear probing | 0.7822 | 0.7817 | 0.7967 | 0.7869 |
| 300 | Steering global MLP | 0.7778 | 0.7700 | 0.7811 | 0.7763 |
| 300 | Steering attention+segments | 0.7878 | 0.7933 | 0.7928 | 0.7913 |
| 500 | Zero-shot | 0.6683 | 0.6683 | 0.6683 | 0.6683 |
| 500 | Linear probing | 0.8056 | 0.7906 | 0.8033 | 0.7998 |
| 500 | Steering global MLP | 0.7839 | 0.7661 | 0.7756 | 0.7752 |
| 500 | Steering attention+segments | 0.8006 | 0.7994 | 0.8011 | 0.8004 |

## AVG over datasets (5 datasets at 1000 shots: no sick)

| Shots | Method | Seed 33 | Seed 54 | Seed 89 | Mean |
|---|---|---|---|---|---|
| 200 | Zero-shot | 0.5588 | 0.5588 | 0.5588 | 0.5588 |
| 200 | Linear probing | 0.5949 | 0.6071 | 0.6020 | 0.6013 |
| 200 | Steering global MLP | 0.6078 | 0.6178 | 0.6103 | 0.6120 |
| 200 | Steering attention+segments | 0.7137 | 0.7007 | 0.7085 | 0.7076 |
| 300 | Zero-shot | 0.5588 | 0.5588 | 0.5588 | 0.5588 |
| 300 | Linear probing | 0.6184 | 0.6178 | 0.6157 | 0.6173 |
| 300 | Steering global MLP | 0.6284 | 0.6349 | 0.6223 | 0.6285 |
| 300 | Steering attention+segments | 0.7321 | 0.7135 | 0.7157 | 0.7204 |
| 500 | Zero-shot | 0.5588 | 0.5588 | 0.5588 | 0.5588 |
| 500 | Linear probing | 0.6228 | 0.6314 | 0.6263 | 0.6268 |
| 500 | Steering global MLP | 0.6310 | 0.6361 | 0.6374 | 0.6348 |
| 500 | Steering attention+segments | 0.7321 | 0.7339 | 0.7309 | 0.7323 |
| 1000 | Zero-shot | 0.5369 | 0.5369 | 0.5369 | 0.5369 |
| 1000 | Linear probing | 0.6074 | 0.6071 | 0.5971 | 0.6039 |
| 1000 | Steering global MLP | 0.6184 | 0.6257 | 0.6237 | 0.6226 |
| 1000 | Steering attention+segments | 0.7344 | 0.7402 | 0.7377 | 0.7374 |
