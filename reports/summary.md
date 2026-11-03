# Summary

| file | status | risk | peer | external | ai | top match |
|---|---|---|---|---|---|---|
| pitch01_kilo.mp4 | ok | HIGH | 100.0 | 0.0 |  | sub01_alpha.pdf (0.98) |
| pitch02_lima.mp4 | ok | LOW | 0.0 | 0.0 | 28.4 | - |
| pitch03_mike.mp4 | skipped | UNPROCESSED |  |  |  | - |
| sub01_alpha.pdf | ok | MEDIUM | 20.0 | 0.0 | 17.6 | pitch01_kilo.mp4 (0.98) |
| sub02_bravo.pdf | ok | HIGH | 100.0 | 0.0 | 49.8 | sub03_charlie.docx (1.0) |
| sub03_charlie.docx | ok | HIGH | 100.0 | 0.0 | 26.7 | sub02_bravo.pdf (1.0) |
| sub04_delta.docx | ok | HIGH | 80.0 | 0.0 | 32.5 | sub05_echo.docx (0.95) |
| sub05_echo.docx | ok | HIGH | 80.0 | 0.0 | 21.7 | sub04_delta.docx (0.95) |
| sub06_foxtrot.pdf | ok | MEDIUM | 0.0 | 0.0 | 76.3 | - |
| sub07_golf.pdf | ok | LOW | 0.0 | 0.0 | 22.9 | - |
| sub08_hotel.docx | ok | MEDIUM | 0.0 | 20.0 | 66.2 | quick_commerce.txt (0.9) |
| sub09_india.docx | ok | LOW | 0.0 | 0.0 | 40.8 | - |
| sub10_juliet.pdf | skipped | UNPROCESSED |  |  |  | - |

## Pairwise best chunk similarity (doc x doc)

Best single-chunk cosine similarity, *before* the boilerplate rule. Near-1.00 cells between unrelated submissions are the shared institutional prompt, which is excluded from every score (listed per report under *Ignored as common template*).

|  | pitch01_kilo.mp4 | pitch02_lima.mp4 | sub01_alpha.pdf | sub02_bravo.pdf | sub03_charlie.docx | sub04_delta.docx | sub05_echo.docx | sub06_foxtrot.pdf | sub07_golf.pdf | sub08_hotel.docx | sub09_india.docx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pitch01_kilo.mp4 | - | 0.62 | 0.98 | 0.61 | 0.61 | 0.59 | 0.59 | 0.46 | 0.41 | 0.53 | 0.44 |
| pitch02_lima.mp4 | 0.62 | - | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.54 | 0.46 | 0.57 | 0.61 |
| sub01_alpha.pdf | 0.98 | 0.70 | - | 1.00 | 1.00 | 1.00 | 1.00 | 0.54 | 0.51 | 0.55 | 0.62 |
| sub02_bravo.pdf | 0.61 | 0.70 | 1.00 | - | 1.00 | 1.00 | 1.00 | 0.62 | 0.52 | 0.62 | 0.68 |
| sub03_charlie.docx | 0.61 | 0.70 | 1.00 | 1.00 | - | 1.00 | 1.00 | 0.62 | 0.52 | 0.62 | 0.68 |
| sub04_delta.docx | 0.59 | 0.70 | 1.00 | 1.00 | 1.00 | - | 1.00 | 0.56 | 0.64 | 0.55 | 0.61 |
| sub05_echo.docx | 0.59 | 0.70 | 1.00 | 1.00 | 1.00 | 1.00 | - | 0.57 | 0.64 | 0.53 | 0.63 |
| sub06_foxtrot.pdf | 0.46 | 0.54 | 0.54 | 0.62 | 0.62 | 0.56 | 0.57 | - | 0.55 | 0.48 | 0.55 |
| sub07_golf.pdf | 0.41 | 0.46 | 0.51 | 0.52 | 0.52 | 0.64 | 0.64 | 0.55 | - | 0.49 | 0.53 |
| sub08_hotel.docx | 0.53 | 0.57 | 0.55 | 0.62 | 0.62 | 0.55 | 0.53 | 0.48 | 0.49 | - | 0.54 |
| sub09_india.docx | 0.44 | 0.61 | 0.62 | 0.68 | 0.68 | 0.61 | 0.63 | 0.55 | 0.53 | 0.54 | - |
