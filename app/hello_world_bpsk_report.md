# SIH 26147 - Automated Signal Analysis Report

Generated 2026-09-29T23:03:37

## Result

* **Status:** UNVERIFIED
* **Decoded payload:** `HELLO WORLD`
* **Processing chain:** BPSK -> de-interleave diag: 29 x 3 diagonal (down-right), offset 0 -> Hamming(15,11)
* **Overall confidence:** 0.4

## Input

* File: `hello_world_bpsk.wav` (880 complex samples)
* Sample type: auto
* Sample rate: 12,000 Hz (WAV header (KNOWN))
* WAV 1 ch @ 48000 Hz
* real passband: analytic signal (FFT Hilbert), carrier ~6000.0 Hz (occupied-band centre), 97% bandwidth 1688 Hz, mixed to 0 Hz, decimated x4 -> 12000 Hz

## Modulation

* PSK / **BPSK** (confidence 1.0)
  * Envelope: CV 0.409, noise-corrected excess 0.409 -> varying envelope (log-odds FSK -6.0)
  * Sample cumulants |C20|=1.00 |C40|=1.59 C42=-1.59: x^2/x^4 keep a DC term -> linear modulation (log-odds -4.0)
  * IF kurtosis: gated 3.12 (raw 16.5); inconclusive (log-odds +0.0)
  * IF histogram (64 bins): 2 peak(s) at +0.00, +0.00 x BW, density at 0 Hz 0.54 of max (log-odds +0.0)
  * Cyclic lines: symmetric tone pair in x^1 -> FSK tones (log-odds +1.5)
  * M-th power line: strongest single line rho=1.00 in x^2 -> x^2 is a pure carrier: BPSK (any pulse shape) (log-odds -8.0)
  * Phase histogram (64 bins): 2 cluster(s) -> power-of-two phase set (PSK-like) (log-odds FSK -1.0)
  * Amplitude histogram (128 bins): 9 level(s) -> multi-level envelope (QAM-like) (log-odds FSK -0.5)
  * Family: linear (PSK/QAM) (P = 1.000)
  * Front end: 10.007 sps, 87 symbols; lattice residual 0.013 (best fit BPSK; < 0.15 = clean constellation)

## Parameters

| Parameter | Value | Unit | Source | Method | Confidence |
|---|---|---|---|---|---|
| Sample rate | 12,000 | Hz | KNOWN | WAV header; WAV 1 ch @ 48000 Hz; real passband: analytic signal (FFT Hilbert), carrier ~6000.0 Hz (occupied-band centre), 97% bandwidth 1688 Hz, mixed to 0 Hz, decimated x4 -> 12000 Hz |  |
| Passband carrier | 6,000 | Hz | ESTIMATED | occupied-band centre of the analytic WAV signal |  |
| Samples/symbol | 10.0008 | samples | ESTIMATED | -3 dB bandwidth + cyclic lines, hypotheses scored by strobe compactness | 1.0 |
| RRC roll-off | 0.5 |  | ESTIMATED | decision-directed min-EVM over candidates (PSD fit as initial value) | 0.71383 |
| Symbol rate | 1,199.9 | Bd | ESTIMATED | cyclic line + Oerder-Meyr refinement + decision-directed rate scan | 1.0 |
| Carrier offset | 0.00205104 | Hz | ESTIMATED | PSD -3 dB midpoint + 2th-power line + costas-bpsk PLL | 0.999987 |
| Es/N0 (MER) | 41.9695 | dB | ESTIMATED | decision-directed MER after sync |  |
| Modulation | BPSK |  | ESTIMATED | unified DSP decision tree (envelope, cumulants, IF histogram/kurtosis, cyclic lines; symbol-level ALRT over PSK/QAM) | 1.0 |
| Decoded payload | HELLO WORLD |  | ESTIMATED | text-heuristic; CRC not valid / no frame | 0.5 |

## Bitstream (L8)

* 88 bits (receiver output (text-heuristic)); randomness bit_balance 0.397727, binary_entropy 0.969606, byte_entropy 2.84535, randomness_score 0.344859
* Frame length estimate: 84
* Blind convolutional estimate: none

| Sync word | Offset | Hamming | Hard corr | LRT | GLRT | Threshold | Detected |
|---|---|---|---|---|---|---|---|
| 0xFAF33420 (CCSDS rate-1/2 ASM) | 8 | 8/32 | 0.5 | -320933.0 | -320955.0 | 3 | no |
| CCSDS ASM 0x1ACFFC1D | 45 | 10/32 | 0.375 | -398805.0 | -398828.0 | 3 | no |
| Telemetry marker 0xD3917E42 | 17 | 10/32 | 0.375 | -401234.0 | -401256.0 | 3 | no |
| PCM sync 0xEB90 | 9 | 4/16 | 0.5 | -160148.0 | -160159.0 | -1 | no |
| 0xF628 (IRIG/PCM) | 37 | 4/16 | 0.5 | -160089.0 | -160100.0 | -1 | no |
| Header 0xAA55 | 38 | 4/16 | 0.5 | -160045.0 | -160056.0 | -1 | no |

## De-interleaving (L9)

* Decision: **diag: 29 x 3 diagonal (down-right), offset 0** (not validated); 464 hypotheses tested

## FEC (L10)

* Decision: **Hamming(15,11)** - (15,11) binary linear block code, rate 11/15, systematic, corrects 1 error
* Family probabilities: Uncoded 0.99, Convolutional code 0.00, Block code (Hamming / RS) 0.00
  * Convolutional parity checks: best CC K=3 (7,7,5) r1/3 (phase 0) syndrome weight 0.370 (0.5 = random; 2 sigma)
  * Block-code checks: best Hamming(15,11) (offset 0) syndrome failures 0.800 (1 sigma); RS blocks decoded 0%
  * No sync word in clear
  * Hamming(15,11) at offset 8: syndrome failures 0.200 (7 sigma below random); 1 single errors corrected, 0 uncorrectable blocks

## Framing (L11)

* Layout: not validated
* CRC-valid frames: 0/0

| # | Start bit | Length | CRC | Valid | Text |
|---|---|---|---|---|---|

## Limitations

* FEC identification covers a library of convolutional codes (K 3-9, rate 1/2, 1/3, punctured 2/3, 3/4, 5/6), Reed-Solomon GF(2^8) profiles (p = 0x11D, fcr = 0), Hamming codes, LDPC codes with known parity-check matrices (IEEE 802.11 n=648 r3/4 and project-generated codes), blind LDPC parity-check recovery (needs more aligned error-free codewords than n) and RS + convolutional concatenated codes; turbo, BCH and polar codes are not identified.
* Pseudo-random interleavers: the LFSR address-generator family is searched and a rank-deficiency period without a structured interleaver is reported as a suspected PR interleaver; an arbitrary permutation is not reconstructed.
* Block interleavers larger than sqrt(N) bits are hypothesised from the burst length (a whole number of blocks) and accepted only when FEC + CRC validate.
* Convolutional (Forney) interleavers are searched for B <= 16 branches and M <= 16; diagonal interleavers over the same R x C hypotheses as block interleavers (both diagonal directions).
* A blind (ASSUMED) sample rate is a hypothesis: only samples/symbol is measured.
* Decoded content is VERIFIED only by a CRC; otherwise it is the best unverified hypothesis.
