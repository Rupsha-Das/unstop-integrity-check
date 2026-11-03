# Evaluation results

## peer

| TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|
| 6 | 0 | 0 | 5 | 1.0 | 1.0 |

## external

| TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 10 | 1.0 | 1.0 |

## ai

| TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|
| 1 | 1 | 0 | 9 | 0.5 | 1.0 |

## Pairs

| pair | relation | found |
|---|---|---|
| sub02_bravo.pdf <-> sub03_charlie.docx | exact_copy | yes |
| sub04_delta.docx <-> sub05_echo.docx | paraphrase | yes |
| sub01_alpha.pdf <-> pitch01_kilo.mp4 | cross_modal | yes |
| sub08_hotel.docx <-> quick_commerce.txt | external | yes |

Pair recall: **4/4**

## Template ignored as boilerplate

5/5 template files have non-empty excluded_boilerplate

2/2 degraded files skipped
