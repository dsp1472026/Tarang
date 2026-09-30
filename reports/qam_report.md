# Phase 3 - QAM (16 / 64 / 256QAM) validation

* **Generated known-payload frames:** 12/15 PASS (blind classification correct, payload byte-exact, CRC-16 valid).
* **Legacy Test-Data QAM captures (.iq + .wav):** classification 12/12; payload exact with expected payload 10/12; fully blind 7/12.

## Receiver

| Stage | Method |
|---|---|
| Front end | shared with PSK: AGC, burst, coarse CFO, RRC matched filter, Oerder-Meyr + Gardner timing |
| Rate / CFO | if the strobes fit no PSK/QAM lattice: coarse-CFO x symbol-rate hypotheses (integer sub-rates, coarse-to-fine amplitude-ring scan, CFO micro-scan), best lattice fit wins |
| Order | ALRT over BPSK..256QAM after sigma_A^2 (radial vs tangential spread) splits PSK/QAM; EVM per order (12.5/8/3.5 % limits), C40 and amplitude kurtosis reported as evidence. No PAPR. |
| Timing | roll-off x timing (x rate for < 200 symbols) scan scored by the lattice residual |
| Carrier | 4th-power line + joint phase / residual-frequency / scale fit (decision-directed complex LS from seeds), then decision-directed carrier + gain loops on confident symbols |
| Edges | symbols next to a capture edge that cuts the pulses: truncated-pulse least squares |
| Ambiguity | sync word (4 rotations x 4 Rs/4 aliases); captures without one: label-mapping hypotheses (Gray/natural x ascending/descending x I/Q order) x rotations x aliases, by expected payload or text |
| Bits | Gray per axis, max-log LLRs (ln P0/P1) |

## A. Generated frames

| Vector | Truth | Classified | EVM % | Ambiguity | CRC | Result | s |
|---|---|---|---|---|---|---|---|
| 16qam_clean | 16QAM | 16QAM | 1.76 | sync-word | valid | PASS | 37.1 |
| 16qam_impaired | 16QAM | 16QAM | 7.43 | sync-word | valid | PASS | 38.1 |
| 16qam_fracsps_bigcfo | 16QAM | 16QAM | 5.9 | sync-word | valid | PASS | 39.2 |
| 16qam_beta025_4800bd | 16QAM | 16QAM | 5.44 | sync-word | valid | PASS | 39.9 |
| 16qam_short_burst | 16QAM | 16QAM | 4.94 | sync-word | valid | PASS | 12.2 |
| 64qam_clean | 64QAM | 64QAM | 1.31 | sync-word | valid | PASS | 127.8 |
| 64qam_impaired | 64QAM | 64QAM | 3.03 | sync-word | valid | PASS | 78.9 |
| 64qam_fracsps_bigcfo | 64QAM | 64QAM | 2.89 | sync-word | valid | PASS | 80.0 |
| 64qam_beta025_4800bd | 64QAM | 64QAM | 3.34 | sync-word | valid | PASS | 51.1 |
| 64qam_short_burst | 64QAM | 256QAM | 2.46 | unresolved | fail | FAIL | 32.5 |
| 256qam_clean | 256QAM | 256QAM | 1.34 | sync-word | valid | PASS | 126.8 |
| 256qam_impaired | 256QAM | 64QAM | 5.48 | unresolved | fail | FAIL | 70.4 |
| 256qam_fracsps_bigcfo | 256QAM | 256QAM | 1.85 | sync-word | valid | PASS | 146.9 |
| 256qam_beta025_4800bd | 256QAM | 256QAM | 1.82 | sync-word | valid | PASS | 123.4 |
| 256qam_short_burst | 256QAM | 256QAM | 3.36 | text-heuristic | fail | FAIL | 35.5 |

## B. Legacy captures (no sync word; payload 'BRIGHT VIXENS JUMP; DOZY FOWL QUACK')

| File | Truth | Classified | Mapping found | Known payload | Blind | Blind text |
|---|---|---|---|---|---|---|
| `all_modulations_output/16QAM.iq` | 16QAM | 16QAM | Gray (standard) | PASS | FAIL | `'YXtLeo)\\ud|QK(U|qj\x07(lAUw(\\ALc)y|xH'` |
| `all_modulations_output/16QAM.wav` | 16QAM | 16QAM | Gray (standard) | PASS | FAIL | `'CSMFLT0WM\\EKR0OUIP.0DJ_]0GJVH0QUABN'` |
| `all_modulations_output/64QAM.iq` | 64QAM | 64QAM | Q-first, I natural-asc, Q natural-asc | PASS | PASS | `'BRIGHT VIXENS JUMP; DOZY FOWL QUACK'` |
| `all_modulations_output/64QAM.wav` | 64QAM | 64QAM | I-first, I natural-asc, Q natural-asc | FAIL (1 chars) | FAIL | `'FRIGHT VIXENS JUMP; DOZY FOWL QUACK'` |
| `all_modulations_output/256QAM.iq` | 256QAM | 256QAM | Gray (standard) | PASS | FAIL | `'c}dlfp5}lwibp?g{`~ 6ho}p5hdz`1wqbn'` |
| `all_modulations_output/256QAM.wav` | 256QAM | 256QAM | Gray (standard) | FAIL (1 chars) | FAIL | `'bsmdlv0um|gir0owkp.0fh\x7f}0ehtj0qwabn'` |
| `qam_pangram_all_files/16qam_bright_vixens.iq` | 16QAM | 16QAM | Gray (standard) | PASS | PASS | `'BRIGHT VIXENS JUMP; DOZY FOWL QUACK'` |
| `qam_pangram_all_files/16qam_bright_vixens.wav` | 16QAM | 16QAM | Gray (standard) | PASS | PASS | `'BRIGHT VIXENS JUMP; DOZY FOWL QUACK'` |
| `qam_pangram_all_files/64qam_bright_vixens.iq` | 64QAM | 64QAM | I-first, I gray-asc, Q gray-asc | PASS | PASS | `'BRIGHT VIXENS JUMP; DOZY FOWL QUACK'` |
| `qam_pangram_all_files/64qam_bright_vixens.wav` | 64QAM | 64QAM | I-first, I gray-asc, Q gray-asc | PASS | PASS | `'BRIGHT VIXENS JUMP; DOZY FOWL QUACK'` |
| `qam_pangram_all_files/256qam_bright_vixens.iq` | 256QAM | 256QAM | Gray (standard) | PASS | PASS | `'BRIGHT VIXENS JUMP; DOZY FOWL QUACK'` |
| `qam_pangram_all_files/256qam_bright_vixens.wav` | 256QAM | 256QAM | Gray (standard) | PASS | PASS | `'BRIGHT VIXENS JUMP; DOZY FOWL QUACK'` |
