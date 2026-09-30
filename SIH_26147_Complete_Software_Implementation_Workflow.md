# SIH 26147 — Software Implementation (as built)

**Problem Statement ID:** 26147 — Automated model for analysis of `.IQ` and `.wav` files with signal parameter extraction
**Organisation:** National Technical Research Organisation (NTRO) — Theme: Space Technology

This document describes **only what is implemented in this repository** (state of 2026-09-29): every
algorithm, threshold, decision rule and data path, taken from the source code. Where a feature is
partial, missing, or only used for testing, it is stated. Section 20 lists known limitations and
Section 21 lists code that exists but is **not** used by the application.

**Changes since the 2026-09-27 revision** (each is described in its section):

* **Multi-signal captures** (§3, §5.2): the capture spectrum is searched for every occupied band; with
  two or more signals each band is mixed to 0 Hz, filtered, decimated and analysed on its own. A header
  selector and a new *Signals* tab show every signal.
* **Image / file recovery** (§15A): a CCSDS-ASM file frame (`0x1ACFFC1D | LEN u32 | PAYLOAD | CRC-32`)
  is searched in every PSK / QAM / FSK decode, uncoded or K = 7 convolutional. Several CRC conventions,
  several transmitter bit labellings and a truncated tail are handled. Recovered images are shown in a
  new *Image* tab and can be saved.
* **Feed-forward linear receiver** (§8A): a receiver without loops for long, unscrambled bursts, where
  Gardner / Costas loops slip. It is used as a fallback when the loop receivers give no verified frame.
* **Tone-correlator FSK demodulator** (§9.7): used when the blind FSK tone model is wrong.
* **Oversampling reduction** (§4.6) and a corrected passband WAV front end (§4.4).
* **Rewritten SOI detector** (§5.1): a busy spectrum no longer hides signals, and filter skirts are no
  longer listed as signals.
* **GUI** (§17): 14 tabs, a colour theme, scrollable pages (fixes a `QWindowsWindow::setGeometry`
  error), Cancel that works during the run, and plots and forensics computed in the background so the
  window no longer shows *Not responding*.
* **Removed:** analogue demodulation (AM / FM / SSB / DSB-SC) and speech-to-text were added and then
  removed on request. The code (`demodulation/analog.py`, `demodulation/speech.py`), the Whisper model
  (`models/whisper`) and the `faster-whisper` package are gone. The application handles **digital**
  PSK / QAM / FSK signals only.
* **Large captures** (§20): distance matrices are now computed in blocks. A 73 MB WAV (18.3 M samples)
  previously failed in the order classifier with an 8.7 GB allocation.

Conventions used throughout the code and this document:

* Frequencies are handled in normalised units (cycles/sample, samples/symbol, cycles/symbol); they are
  converted to Hz only with the sample rate supplied for the capture.
* LLR convention: `llr = ln P(bit = 0) / P(bit = 1)`; hard bit = 1 when `llr < 0`.
* Bits are packed MSB-first unless stated otherwise.
* Every estimated quantity is stored as a `core.evidence.Parameter` (name, value, unit, source
  `KNOWN / ESTIMATED / ASSUMED / UNKNOWN`, method, confidence, list of `Evidence`). Each `Evidence` has a
  description, a source tag and a score in [0, 1].

---

## Contents

1. How to run
2. Repository map
3. End-to-end processing flow (what the GUI actually executes)
4. L1 — Ingestion (`.iq` sample formats, duration → sample rate, `.wav`)
5. L2 — Signal detection
6. L3/L4 — Modulation classification (FSK / PSK / QAM and order)
7. PSK receiver (L1–L6)
8. QAM receiver
8A. Feed-forward linear receiver (fallback for long unscrambled bursts)
9. FSK receiver (including the tone-correlator fallback, §9.7)
10. Unframed text payloads — the ASCII text model
11. L8 — Bitstream analysis
12. L9 — De-interleaving
13. L10 — FEC identification and decoding
14. L11 — Framing / protocol analysis
15. Joint post-demodulation search (L8–L11 together)
15A. Image / file-frame recovery (L11, all families)
16. Report generation
17. GUI — the 14 tabs
18. Synthetic test-signal generation (validation only)
19. Validation tools, tests and measured results
20. Known limitations and open issues
21. Code present but not used by the application

---

## 1. How to run

The Python virtual environment is in `.venv`. `requirements.txt` lists:

* **Core:** numpy, scipy, PyQt6, pyqtgraph, reedsolo, matplotlib.
* **Image payloads:** pillow (decodes and saves PNG / JPEG / … payloads, and does the Lanczos
  enlargement in the Image tab).
* **ML modulation-classifier models:** scikit-learn and onnxruntime. These models are not on the
  pipeline path (§21), but their modules import these packages.
* **Other listed packages:** torch, ldpc, h5py and pyyaml, plus pytest for the tests.

Install into the venv with:

```text
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`faster-whisper` is no longer required: speech-to-text was removed.

```text
.venv\Scripts\python.exe app\main.py          # GUI
.venv\Scripts\python.exe tools\demod.py --file <capture> [--fs <Hz>] [--dtype ci16_le|cf32_le]
                                               # command-line demodulation of one capture (L1-L6 only;
                                               # --dtype offers only the two legacy names)
.venv\Scripts\python.exe -m pytest -q tests\unit tests\integration
```

`gui/qt_compat.py` selects the first importable Qt binding in the order PyQt5, PyQt6, PySide6. It also
adds the Qt DLL directories of the virtual environment to the Windows DLL search path.

`app/main.py` starts the GUI under `if __name__ == "__main__":`. This guard is required: the GUI's
worker process (§17.3) is started with the `spawn` method, which re-imports the main module. Without the
guard, each worker would open a second window.

Image-decoding material lives in `Image Decoding/`:

* The user's transmitter script `c.py` (Space Downlink Simulator). Its indentation is flattened in this
  copy, so it does not run.
* Sample images: `Orginal.png` (3456 × 2234), `Demo.png`, `test1_encoded.png`.
* The original screenshot capture `Screenshot 2026-09-24 at 4.06.53 PM_qpsk.wav`.
* `test_results/`: the ten test signals of §19.1 with their sent and recovered images.

---

## 2. Repository map

| Package | Role in the application |
|---|---|
| `app/main.py` | Entry point: creates `QApplication` (Fusion style) and `gui.main_window.MainWindow`. |
| `gui/` | Main window, 14 tabs, widgets (plots, IQ setup dialog, tables), workers. `unified_worker.py` runs the pipeline in a `QThread`: signal splitting, the decode-validated order / rate / family check, and the image / feed-forward / FSK fallbacks (§3). `theme.py` holds the QSS colour theme and the pyqtgraph defaults. `widgets/background.py` (`BackgroundRunner`) computes plots and forensics off the GUI thread, in a thread or a worker process (§17.3). New tabs: `tabs/image_tab.py`, `tabs/signals_tab.py`. |
| `ingestion/` | `iq_formats.py` (sample formats + blind format detection), `capture.py` (loader for `.iq`/`.wav`, passband WAV front end, `reduce_oversampling`), `format_detector.py` (file type, header skip, sidecar/filename hints). |
| `detection/fft_detector.py` | Welch PSD energy detector: the SOIs shown in the Detection tab (§5.1). |
| `detection/channelizer.py` | Multi-signal band finder `find_signals` and per-band extraction `extract` (§5.2). |
| `dsp/` | AGC, burst detection, channel filter, DC estimate (`conditioning.py`); RRC pulse / matched filter / cubic interpolation (`filters.py`); constellations and Gray mapping (`constellations.py`); CPM phase/pulse models (`cpm.py`). |
| `estimation/` | `linear_mod_params.py` (PSD band, symbol-rate hypotheses, roll-off fit), `linear_rate_check.py` (ring-residual rate scan), `fsk_estimator.py` (FSK band, IF, rate, level fits, continuity, re-modulation match), `blind_sample_rate.py` (blind sample-rate hypotheses for `.iq` files). |
| `modulation/` | `unified_classifier.py` (FSK vs linear, linear branch), `signal_features.py`, `fsk_classifier.py` (envelope test), `linear_classifier.py` (PSK/QAM order), `psk_classifier.py` (M-th power line evidence). |
| `synchronization/` | `symbol_timing.py` (Oerder–Meyr, Gardner), `carrier_psk.py` (M-th power line, 2nd-order PLL). The other files are only imported for result containers (see §21). |
| `demodulation/` | `psk_receiver.py`, `psk.py` (ambiguity resolution, general / English / numeric message models), `qam_receiver.py`, `fsk_receiver.py`, `fsk_demod.py`, `ff_receiver.py` (feed-forward PSK/QAM receiver, §8A), `fsk_tones.py` (tone-correlator FSK over (M, h) hypotheses, §9.7), `demodulator.py` (result container). |
| `bitstream/` | `sync_search.py` (sync-word search), `entropy.py`, `correlation.py`, `patterns.py`, `framing.py` (burst / frame-length forensics), `forensics.py` (the Bitstream tab's analysis as one picklable function, run in the worker process). |
| `deinterleaving/` | `gf2.py` (GF(2) rank), `block.py` (block interleaver + blind period/offset), `convolutional.py` (Forney interleaver), `diagonal.py` (diagonal/helical interleaver), `pseudorandom.py` (LFSR address-generator and QPP permutations, PR presence evidence). |
| `fec/` | `convolutional.py` (codes, Viterbi with optional input priors, syndrome), `reed_solomon.py`, `block_codes.py` (Hamming), `ldpc.py` (LDPC construction, encoder, min-sum decoder, alignment), `dual.py` (GF(2) null space + sparse parity-check search), `concatenated.py` (RS + byte interleaver + convolutional, iterative decoding), `identify.py` (family scores, blind conv parameters). |
| `protocol/` | `crc.py` (CRC catalogue), `frames.py` (frame layout discovery by CRC). |
| `postdemod/analyzer.py` | Joint L8–L11 hypothesis search run after demodulation. |
| `postdemod/image_payload.py` | Image / file-frame search and payload identification (§15A). |
| `reporting/report.py` | Report builder (HTML / Markdown / JSON). |
| `conditioning/` | Manual bit-conditioning operations used by the Conditioning tab. |
| `synthetic/` | Test-signal generators and the test frame format. Not used to analyse captures, except `framing.py` (defines the test frame's sync word / CRC) and `iq_io.py` (reads `.iq` files). Image transmitter for tests: `image_framing.py`, `tx_image.py`, `rx_image.py`, `gen_test_image.py`, `run_image_pipeline.py`. |
| `tools/` | Validation scripts. `tests/` | pytest unit and integration tests. |

---

## 3. End-to-end processing flow

The GUI (`MainWindow.run_pipeline`) starts `UnifiedPipelineWorker(config)` in a background `QThread`.
`config` comes from the Input tab: file path, IQ data type, endianness, IQ order, byte offset, sample
rate and SOI threshold. It can also carry `multi_signal` (default `True`) and `postdemod_budget_s`
(default 600 s).

### 3.1 Capture level — `UnifiedPipelineWorker.work()`

```text
_load(config)                            L1  ingestion/capture.py -> complex baseband + sample rate
 └─ reduce_oversampling(cap)                 decimate to ~6 x the occupied band (§4.6)
find_signals(samples, fs)                L2  detection/channelizer.py: occupied bands, strongest first (§5.2)
FFTDetector.detect_sois(whole capture)   L2  SOIs of the whole capture (§5.1): Detection tab + report
if fewer than 2 bands:                       one signal: the whole capture is analysed as before
    _analyze(cap, sois = whole-capture SOIs)
else, for each band i (at most 16):
    extract(cap, band)                       mix to 0 Hz, FIR low-pass, decimate (§5.2)
    _analyze(sub-capture, band, sois = the whole-capture SOIs that overlap this band)
    (a failing signal is reported with its error; the others are still analysed)
result = first signal without an error, plus:
    signals        list of per-signal result dicts (the dict of §3.2 for each signal)
    bands          the SignalBand list
    capture_buffer the whole capture (Detection tab, Signals-tab spectrum, Input-tab summary)
    capture_sois   the whole-capture SOIs
    capture_notes_all
```

The SOIs are detected once, on the whole capture. Earlier they were detected per channelised signal,
and the filtered-away edges of the extracted band were listed as extra signals (filter skirts).

**Progress.** With N signals, the progress bar is split into N equal shares after the first 5 %.
Messages are prefixed `[Signal i/N @ ±f]`, and each signal's own messages (§3.4) are scaled into its
share.

**Cancellation.** `_check_cancel()` is called between layers and stages: before L2, classification,
the receivers and L8–L11; inside every fallback loop; inside the image-frame search (per hypothesis);
and from the L8–L11 progress callback. When Cancel was pressed, it raises `_Cancelled`.
`BaseWorker.run` then emits `cancelled` (not `error`), and the window shows "Operation cancelled by
user". A running numpy call is not interrupted, so a cancel takes effect at the next checkpoint
(normally within seconds; during the classification of a very large capture, minutes).

### 3.2 Signal level — `UnifiedPipelineWorker._analyze(cap, band, sois)`

```text
classify_capture(samples, fs)                 L3/L4 family FSK | PSK | QAM and order (§6)
FSK:
 ├─ receive_fsk()                             L3–L6 (§9)
 │   fitted h < 0.05 -> reclassify_as_psk() (constant-envelope PSK), continue as PSK
 └─ decode not verified (no CRC frame, no decisive text):
     find_image("FSK", r)                     image / file frame in the FSK bits (§15A)
     no CRC-valid frame -> _fsk_tone_fallback  tone correlator over (M, h) hypotheses (§9.7)
PSK / QAM:
 ├─ receive_psk() / receive_qam()             on the classifier's front end (§7, §8)
 ├─ verify_linear_order()                     order / rate check, bursts < 600 symbols (§3.3)
 ├─ image = find_image(family, r)             file-frame search, cached on the result (§15A)
 ├─ no image and no CRC-valid decode (rank < 2), or an image with a failing CRC:
 │    for sps in [estimated, nearest standard rate, nearest integer samples/symbol]:
 │        receive_feedforward(samples, fs, modulation, sps)      (§8A)
 │        adopt if its file frame is CRC-valid (or is the only frame found); stop at CRC-valid
 ├─ image found but CRC still failing, and < 50 000 symbols:
 │    loop receiver re-run at the exact-rate hypotheses; adopt only if the CRC validates
 └─ nothing verified at all, p(FSK) > 1e-3 and < 600 symbols:
      receive_fsk(); adopt only if it verifies (CRC, decisive text or an image frame)
image found  -> payload = "[image description]", CRC = the frame's CRC, "Recovered image" parameter;
                L8–L11 is skipped (layout, FEC and CRC already identified)
otherwise    -> hypotheses_from_receiver() + analyze()   L8–L11 joint search (§15)
                a CRC-valid L8–L11 result replaces the receiver's payload / CRC flag
```

**Exact-rate hypotheses** (`_exact_rate_hypotheses(fs, sps0)`) are built from the estimated
samples/symbol `sps0`:

* the nearest standard symbol rate from the list of 42 rates (50 Bd … 1 MBd), when the sample rate is
  known;
* the nearest integer samples/symbol.

A hypothesis is kept only if it lies within 7 % of the estimate and differs from it.

Each signal's result dict contains:

* `buffer`, `sois`, `parameters` (a dict of `Parameter`);
* `demod_result` (bits, LLRs, symbols, EVM or FSK error %), `payload_text`, `crc_valid`;
* `classification`, `postdemod`, `config`, `capture_notes`;
* `image` (an `ImagePayload` or `None`), `band` (a `SignalBand` or `None`) and `label`;
* the three legacy containers for the Sync tab (§17).

Each fallback that is adopted adds an `Evidence` item to the Modulation parameter:

| Evidence tag | Added when |
|---|---|
| `ORDER_CHECK` | the order / rate check chose another hypothesis |
| `FAMILY_CHECK` | the family was revised to FSK |
| `FF_RECEIVER` | the feed-forward receiver's frame was adopted |
| `IMAGE_RETRY` | a loop-receiver re-run at an exact rate made the CRC valid |
| `FSK_TONE_FALLBACK` | the tone correlator recovered the frame |

When a fallback is adopted, the reported modulation is the one that decoded the frame. For example,
after a tone-correlator decode it reads `2FSK (h = 0.50, tone correlator)`, not the blind fit's label.

**Why the fallbacks are ordered this way.**

* **Loop receivers first.** They are the verified path for short, framed bursts (§19). The feed-forward
  receiver runs only when they give nothing verified.
* **Feed-forward receiver next.** An unscrambled image contains long runs of identical symbols, for
  example a white background. These carry no timing or carrier transitions, so Gardner and Costas
  loops drift and slip. On the screenshot capture, lock was lost about 21 500 symbols in, and the rest
  of the image became colour noise (§8A).
* **Exact rate after that.** A rate estimate a few ppm off leaves bit errors that the exact rate does
  not. On a 991-symbol 64QAM burst, the estimate was 7.99990 samples/symbol instead of 8.
* **FSK fallback last, for short bursts only (< 600 symbols).** The FSK receiver's per-symbol
  Viterbi / BCJR scans run for hours over hundreds of thousands of symbols. A 2 M-sample QPSK capture
  with a binary, unverifiable payload hung here before this limit was added.

### 3.3 Decode-validated order / rate check (`verify_linear_order`)

**When it runs.** On short bursts the order classifier can pick a neighbour (8PSK → 64QAM,
16PSK → 16QAM), and dense QAM can lock at an integer fraction of the symbol rate. The check runs only
for bursts of fewer than 600 symbols (`ORDER_CHECK_MAX_SYMBOLS`), and only when the classified
hypothesis gives neither a CRC-valid frame nor a decisive text reading. It follows a fixed plan with no
time limit, so the result does not depend on machine load.

**Step 1 — other orders.** The other six PSK/QAM orders are tried on the classifier's front end, most
probable first.

**Step 2 — rate hypotheses.** Up to 4 symbol-rate hypotheses are tried, most plausible first:

* **(a) Bandwidth-consistent standard rates**, when the sample rate is known. These are the standard
  symbol rates consistent with the 99 %-power bandwidth B99: Rs between B99/1.5 and B99/0.8, nearest
  to B99/1.05 first. On the test generators, B99 measured 0.94–1.41 × Rs. The −3 dB width and the
  cyclic lines of a short repetitive text payload gave 0.2–1.4 × Rs on 256QAM, so they are not used.
* **(b) Multiples of the estimate:** × 2, × 3 and × ½, each with the nearest standard rate within 7 %.

All standard rates are reported as ASSUMED. Each rate hypothesis is tried with **all seven orders**, on
the receiver's own front end, which tests its own coarse-CFO hypotheses. A fresh classification of that
rate's symbols only sets the trial order: at the true rate, it called a 256QAM burst QPSK at 97 %.

**What counts as verified.**

* A CRC-valid frame ends the search.
* A text reading counts only if:
  * it is ≥ 8 bytes long;
  * it covers ≥ 50 % of the demodulated bits;
  * its trimmed text region (§10) scores ≥ −4.3 nats/char;
  * for a reading that the numeric model explains best, ≥ 90 % of the characters between the first and
    last digit are digits or separators. A wrong hypothesis on a digit payload produced
    `336f73263#1332f`.
* An image frame counts as verified: rank 2 if its CRC is valid, rank 1 if not.
* A sync word found with a failing CRC counts as evidence of the right rate / order, and is reported as
  such.

The adopted hypothesis and the number of hypotheses tried are recorded in the Modulation evidence. If
nothing verifies and the FSK probability is not negligible, the FSK receiver is run, and it is adopted
only if its decode verifies: short wideband 8FSK bursts can look linear to the envelope features.

`_linear_name` maps the receiver label `4QAM` to `QPSK` by exact match. An earlier `str.replace`
turned `64QAM` into `6QPSK`.

### 3.4 Progress messages (per signal)

| % | Stage |
|---|---|
| 15 | L2 |
| 30 | Classification |
| 55 | Receiver |
| 57 | Decode verification |
| 58–68 | Order / rate check |
| 64 | Feed-forward receiver |
| 66–69 | Image-frame search, FSK tone fallback, exact-rate retry, FSK check |
| 70–98 | L8–L11 |
| 100 | Done |

At capture level: 2 % (L1) and 4 % (band finding).

---

## 4. L1 — Ingestion

### 4.1 `.iq` sample formats (`ingestion/iq_formats.py`)

| Key | Accepted names | Bytes per complex sample | Conversion to float |
|---|---|---|---|
| `cu8` | CU8, uint8, u8, rtl | 2 | (v − 127.5) / 128 |
| `cs8` | CS8, CI8, SC8, int8, i8, s8 | 2 | v / 128 |
| `cs16` | CS16, CI16, SC16, int16, i16, s16 | 4 | v / 32768 |
| `cf32` | CF32, FC32, float32, complex64, f32 | 8 | as stored |
| `cf64` | CF64, FC64, float64, complex128, f64 | 16 | as stored |

Samples are interleaved I, Q. Multi-byte types accept an `_le` / `_be` suffix (default little-endian);
the legacy sidecar names `ci16_le` and `cf32_le` resolve to `cs16` / `cf32`. A trailing partial complex
sample is dropped. `normalize_datatype()` takes the first token of a GUI label (e.g. "CS16 / CI16 / SC16
(int16)" → `cs16`).

### 4.2 Blind sample-format detection

`FormatDetector.auto_detect_iq_format(path)` (called by the IQ setup dialog, and by the worker when the
data type is "auto") uses, in order:

1. **Sidecar metadata**: `<file>.sigmf-meta`, `<stem>.sigmf-meta`, `<stem>.json`, `<stem>_metadata.json`
   — the SigMF `core:datatype`, a `datatype` key, or the repository's own `format.files` block.
2. **File-name hints**: `_cu8`, `_cs8/_ci8/_int8`, `_cs16/_ci16/_sc16/_int16`, `_cf32/_fc32/_float32/_complex64`,
   `_cf64/_fc64/_float64/_complex128` (and `_be`).
3. A RIFF/WAVE header inside an `.iq` file → reported as CS16 with a warning.
4. **Text header skip**: a leading run of ≥ 16 printable bytes that contains a newline and starts with a
   letter or one of `# { % ; / < [` is skipped up to its last newline. (A density test is not used because
   most CU8 sample bytes are printable.)
5. **Byte-structure analysis** (`detect_iq_datatype`) on the whole data, or on three 256 KB regions
   (beginning, middle, end) of large files, majority vote:
   * **Floats**: the bytes are read as CF32 and CF64, little- and big-endian. A layout is plausible when all
     values are finite, fewer than 0.1 % of the non-zero magnitudes are denormal, the median magnitude is in
     (1e-15, 1e15), and the 99.9th percentile is within a few decades of the median
     (score = clip((6 − log10(p99.9/median)) / 2, 0, 1)). Score ≥ 0.5 selects that float type.
   * **8-bit vs 16-bit**: bytes are split into 4 lanes (one CS16 sample = two 8-bit samples). A constant
     component (e.g. Q ≡ 0) is ignored. CS16 is chosen when any of these holds: (a) the high byte equals the
     sign extension of the low byte in ≥ 60 % of pairs (low-level int16); (b) the low-byte lane changes
     randomly (mean |step mod 256| > 40 and > 1.6 × the high-byte lane); (c) the int16 reading is smoother:
     roughness (mean |Δ| / std) < 0.8 × the best 8-bit reading; (d) the low-byte histogram is close to
     uniform (total-variation distance below 1.5 × its sampling noise + 0.02) while the high byte is at least
     0.12 further from uniform. Both endiannesses are tested.
   * **CU8 vs CS8**: the offset that gives the smaller mean absolute step between consecutive samples wins
     (the wrong offset turns every zero crossing into a ≈ 255 jump).
   Result: legacy name (`uint8 / int8 / int16 / float32 / complex128`), endianness, header offset,
   confidence and the reason text shown in the dialog.

### 4.3 Sample rate of `.iq` files — three sources

A raw `.iq` file carries no sample rate. When an `.iq` file is selected, `gui/widgets/iq_duration_dialog.py`
(`IQSetupDialog`) opens, shows the detected type (user can override) and offers three sample-rate sources
(radio buttons):

1. **Recording duration** (default, exact) — described below; provenance `KNOWN`.
2. **Blind estimate from the signal** — `estimation/blind_sample_rate.py`, see §4.3.1; provenance `ASSUMED`.
3. **Unknown** — no sample rate; every result stays in normalised units (cycles/sample, samples/symbol);
   provenance `UNKNOWN` (if the file has a sidecar with a rate, the loader still uses it and marks it `KNOWN`,
   "sidecar metadata").

**Recording duration**: the user enters the duration with a unit (s, ms, µs, ns). The sample rate is computed as

```text
complex_samples = floor((file_size − header_offset) / bytes_per_complex_sample)
sample_rate     = complex_samples / duration_seconds
```

A warning is shown if the data size is not a multiple of the sample size, or if the rate is outside
1 Hz – 100 GHz (the user may proceed anyway). The dialog returns the canonical data-type tag
(e.g. `cs16_le`), the header offset and the rate; the Input tab copies them into its fields (rate with
10 significant digits). The sample rate is then marked `KNOWN` in the parameters (user-provided).
If the `.iq` file has a sidecar with a sample rate and no rate is supplied, `synthetic/iq_io.read_iq`
uses the sidecar (`format.sample_rate_hz`, or legacy `config.baseband_fs_hz`, `sample_rate`,
`baseband_sample_rate`).

`load_capture()` also supports: byte offset, endianness override, `IQ`/`QI` order (swap), and
`layout="planar"` (all I then all Q; not exposed in the GUI).

#### 4.3.1 Blind sample-rate hypotheses (`estimation/blind_sample_rate.py`)

What the samples determine is the symbol rate **relative** to the sample rate (samples/symbol); the absolute
sample rate cannot be measured from the samples alone (8 samples/symbol looks the same at 9.6 kHz / 1200 Bd
and at 1.92 MHz / 240 kBd). The estimator therefore:

1. reads up to 262 144 samples of the file in the selected sample format;
2. decides FSK vs linear with the classifier's log-odds (`unified_classifier.fsk_vs_linear`, §6.1);
3. measures samples/symbol blind — linear: the PSK front end (|x|² cyclostationary line with the −3 dB band
   prior, candidates scored by strobe compactness, Oerder–Meyr refinement, §7.1); FSK: the IF-transition
   spectral-line estimator (§9.2);
4. forms hypotheses fs = samples/symbol × each standard symbol rate (list of 29 rates from 50 Bd to 2 MBd,
   in a fixed preference order starting 1200, 2400, 4800, 9600 Bd), keeps 1 kHz ≤ fs ≤ 100 MHz, and when fs is
   within 1 % of a standard audio/SDR sample rate (list of 53 rates, 8 kHz … 61.44 MHz) proposes that standard
   value; hypotheses matching a standard rate are listed first, otherwise the preference order is kept.

The dialog shows the measured samples/symbol, the method and the ranked list; the user picks one. The chosen
rate is recorded with source `ASSUMED`, confidence 0.3 and the measured samples/symbol in the method text.
Measured on the 117 Test-Data `.iq` files with a known rate (sidecars not used by the estimator): the true
rate appears in the list for 110 (exactly, as the proposed standard value, for 108), within the first three
for 89 and first for 46 — the order within the list is a preference, not a measurement.

### 4.4 `.wav` files (`ingestion/capture.py`)

The sample rate comes from the WAV header. int16 / int32 / uint8 samples are scaled to [−1, 1).

**Stereo with different channels** (and no sidecar saying "passband" / 1 channel): left = I, right = Q.

**Mono, or stereo with identical channels:** treated as a **real passband** signal
(`_real_passband_to_baseband`):

1. Remove the mean.
2. Form the analytic signal z with the FFT Hilbert transform (`scipy.signal.hilbert`).
3. Shift the positive half-band [0, fs/2) to [−fs/4, fs/4): multiply by e^(−j2π·0.25·n), then
   decimate by 2 (`resample_poly`).
   * Why: `occupied_band` takes its noise floor from the quietest bins. On z itself it read the empty
     negative half as the floor and counted all the white noise of the positive half as signal. A
     2.4 kHz FSK signal at 9 kHz came out 22 kHz wide at 12 kHz. After the shift there is no empty
     region.
4. Carrier = centre of the 97 % occupied band of this signal (`occupied_band`, §4.5). In cycles/sample
   of the file rate: fc = 0.25 + centre / 2.
5. Mix to 0 Hz.
6. Decimate by a further `floor(1 / (6 × occupied bandwidth))` with `resample_poly`, for a total
   decimation of 2 × that. Its anti-alias filter is the only channel filter, so out-of-band noise
   remains for the receivers' SNR estimate.

The estimated carrier (Hz) is reported as the "Passband carrier" parameter. The receivers remove any
residual frequency offset themselves.

### 4.5 `occupied_band()` (used by the WAV loader, classifier and FSK receiver)

Welch PSD (segment ≤ 1024, two-sided). Noise level = median of the lower half of the PSD bins; excess =
max(medfilt3(PSD) − 2 × noise, 0); the band is the 1.5 %–98.5 % range of the cumulative excess (97 %
occupied band); centre = midpoint; SNR from the in-band power over the 10th-percentile floor.

### 4.6 Oversampling reduction (`reduce_oversampling`, applied to every capture by `_load`)

**Problem.** A capture sampled far above its bandwidth is slow to process, and its rate estimates can
fail. At about 40 samples/symbol (1200 Bd in a 48 kHz WAV / IQ file), the FSK receiver's rate estimate
locked onto twice the symbol rate and no frame was found. At 8–20 samples/symbol the same capture
decoded, and the classifier and receivers run several times faster.

**Method.**

1. Measure the occupied band (§4.5).
2. Compute the span = bw99 + 2 × |centre|, so that a residual carrier offset stays inside the band.
3. Decimation = floor(1 / (6 × span)), limited so that at least 4096 samples remain.
4. Decimate with `resample_poly` when the decimation is ≥ 2.

**Unchanged captures:**

* fewer than 8192 samples;
* occupied-band SNR below 3 dB (the band is then noise and says nothing about the rate);
* captures already near 6 × their band (e.g. 4 samples/symbol).

A capture note records the decimation, and `Capture.decimation` accumulates across the passband front
end, this step and channelisation (§5.2).

---

## 5. L2 — Signal detection

### 5.1 Signals of interest (`detection/fft_detector.py`, `FFTDetector.detect_sois`)

Run once on the **whole** capture (§3.1). The SOIs are shown in the Detection tab and counted in the
report.

**PSD.** Welch PSD (Hann, 1024-sample segments, 50 % overlap, two-sided, fft-shifted), in dB.

**Threshold.**

1. Light smoothing with a 3-bin median filter. This removes single-bin Welch fluctuations and keeps a
   tone's 3–4-bin main lobe.
2. Noise floor = median of the smoothed PSD; spread = MAD.
3. Threshold = floor + max(T, min(3 × MAD, T + 4 dB)), where T is the user's threshold (default 6 dB).
   The cap at T + 4 dB is new. In a busy spectrum (several signals, deep empty regions) the MAD is the
   spread of the signals themselves (about 30 dB), and 3 × MAD hid everything except the carrier
   lines.

**Runs of bins above the threshold.** Runs are merged when their gap is narrower than max(2 bins, half
the wider run). This joins the islands of one signal, such as a 2FSK tone line next to its main lobe.

**Filtering.** A run is dropped when it carries < 0.5 % of the detected power **and** peaks more than
40 dB below the strongest run. These are filter skirts, stop-band ripple, or the spectral leakage of a
neighbour.

**Reported per SOI:**

* power-weighted centre frequency;
* bandwidth = bins × bin width;
* SNR = peak − floor;
* evidence including its share of the detected power.

The time extent is the whole capture (no time segmentation).

### 5.2 Multi-signal band finding and extraction (`detection/channelizer.py`)

**`find_signals(x, fs)`** decides whether a capture holds several transmissions on different carriers,
and where they are.

1. **Spectrum.** Welch PSD with up to 8192-sample segments (a power of 2 ≤ N/16), two-sided,
   9-bin median smoothing. Noise floor = median bin level; this is valid while the signals occupy less
   than half of the band.
2. **Occupied bins.** A bin is occupied when it is more than 10 dB above the floor (`THRESH_DB`).
3. **Gap bridging.** Gaps narrower than 1 % of the sample rate (`MERGE_GAP`) are bridged.
4. **Island merging.** Neighbouring runs whose gap is under half the wider run's width are merged. This
   is **repeated until stable**. A noisy, lumpy spectrum (unscrambled image data at 14 dB) broke one
   signal into many pieces, and a small piece between a skirt island and the main body blocked a single
   left-to-right pass: the 14 dB QPSK test was split into two "signals" before this change. Separate
   stations (e.g. 8 kHz-wide channels 5 kHz apart) stay apart.
5. **Dropped bands:** narrower than 8 bins (`MIN_BINS`), or carrying < 1 % of the occupied power
   (`MIN_POWER_FRACTION`).
6. **Result.** At most 16 bands, strongest first. Each `SignalBand` has: index, centre (Hz, or
   cycles/sample when the rate is unknown), bandwidth, peak above floor, in-band SNR and power fraction.

**Splitting.** A capture is split only when at least two bands survive. Otherwise it is analysed whole,
as before.

**`extract(cap, band)`** creates the sub-capture for one band:

1. Mix the band centre to 0 Hz.
2. FIR low-pass (`firwin`, 101–2001 taps, applied with `oaconvolve`), cut-off at ± half the bandwidth
   widened by 15 % on each side (`EDGE_MARGIN`).
3. Decimate with `resample_poly` (Kaiser β = 5) to about 6 × the band.

The sub-capture keeps the file's notes, adds a "channelised …" note, and carries the absolute carrier
(the capture's carrier + the band centre).

### 5.3 Burst detection (used by the receivers)

`dsp/conditioning.detect_burst`:

1. Channel-filter around the band.
2. Smooth |x|² over about 4 symbols.
3. Floor = 2nd percentile.
4. If the median is less than 8 dB above the floor, the signal fills the capture. Otherwise, samples
   more than 6 dB above the floor form the burst, widened by half the window.

Noise power is taken from outside the burst.

---

## 6. L3/L4 — Modulation classification (`modulation/unified_classifier.py`)

`classify_capture(samples, fs)` returns family (FSK / PSK / QAM), modulation name, confidence, feature
dictionary and the evidence list.

### 6.1 FSK vs linear — sum of log-odds terms (`fsk_vs_linear`)

Features (`signal_features.compute_features`) are computed on the detected burst after removing the
occupied-band centre and a channel filter (± 1.0 × the 97 % band); the outer 5 % of samples are dropped.

| Term | Measurement | Log-odds added (positive = FSK) |
|---|---|---|
| Envelope | noise-corrected envelope CV: `excess = sqrt(max(cv² − cv_noise², 0))`, `cv_noise = sqrt(1/(2·SNR_inband))`; the smaller CV of the raw and IQ-balanced burst is used (`fsk_classifier.classify_family`) | `clip((0.19 − excess) / 0.02, −6, 6)` |
| Cumulants | sample-level |C20| = |E x²|/E|x|², |C40| (normalised) | −4 if max(|C20|, |C40|) > 0.15, else 0 |
| IF kurtosis | kurtosis of the instantaneous frequency, only samples whose amplitude > 0.5 × RMS | +1 if < 2.5; −1 if > 3.5; else 0 |
| IF histogram | 64-bin histogram (1–99 % range, Gaussian-smoothed), peaks ≥ 20 % | +1 if ≥ 2 peaks and density at 0 Hz < 0.5 of max; −1 if 1 peak and density at 0 Hz > 0.8; else 0 |
| Cyclic tone pair | two strongest lines of `|y|·exp(j·k·arg y)`, k = 1, 2: symmetric pair (second ≥ 0.5 of first, ≥ max(0.2, 3/√N)) | +1.5 if a pair exists |
| M-th power line | strongest single line of x² or x⁴ (when not a pair), strength ρ | −8 if ρ ≥ 0.85 (pure carrier: BPSK/QPSK of any pulse shape) |

`p_fsk = sigmoid(sum)`. Sum > 0 → FSK; the number of IF-histogram peaks (if ≥ 2) is kept as a tone-count
hint. Two textbook rules were found wrong on this data and are only weak, gated evidence: "IF kurtosis > 3
means FSK" (pulse-shaped PSK/QAM has IF spikes at envelope nulls, so it is *more* kurtotic) and "≥ 2 IF
peaks means FSK" (fires on about half of the PSK captures).

### 6.2 Linear branch (`classify_linear_capture`)

1. If the envelope excess < 0.12 (constant envelope, i.e. rectangular-pulse PSK), the symbol rate is taken
   from the **phase-jump rate**: the FSK rate estimator in phase-discontinuous mode on the IF.
2. **PSK front end** (`psk_front_end`, §7.1) → timing-recovered symbols `s_core`.
3. **Lattice residual**: for every hypothesis (BPSK, QPSK, 8PSK, 16PSK, 16QAM, 64QAM, 256QAM) the central
   ≤ 128 symbols are carrier-recovered (§6.3) and fitted; residual = √noise_var / (gain × d_min). A clean
   constellation gives < 0.15. With fewer than 20 symbols the residual is treated as ∞.
4. Coarse-CFO alternatives: the −10 dB band-edge midpoint and (≥ 1200 samples, ρ ≥ 0.05) the x⁴ line
   (`z = |x|·e^{j4·arg x}`, peak / 4). The one with the lowest lattice residual is kept.
5. If still > 0.15: roll-off × timing refinement (β ∈ {0.25, 0.35, 0.5, front-end β}, timing offset
   ±0.15 symbol).
6. If still > 0.15: hypothesis search —
   * integer sub-rates sps/k (k = 2…8, sps/k ≥ 2.2), each fine-tuned on the ring residual over ±4 %, under
     each coarse-CFO hypothesis;
   * early exit if one of the two best (after refinement) gives a clean lattice;
   * otherwise, for every CFO candidate (−3 dB midpoint, occupied-band centre, 0, power centroid, x² and x⁴
     line peaks), a constellation-agnostic **ring-residual rate scan** (`linear_rate_check.scan_rate`,
     2.2–64 samples/symbol, coarse-to-fine; distance of strobe amplitudes to the nearest ring of any
     hypothesis, in units of the local ring gap);
   * roll-off/timing refinement of the 3 best; CFO micro-scan ±0.012 cycles/sample in 0.002 steps around up
     to 4 rates.
   * **Pick rule** (`_pick`): best residual, but a candidate with an integer-fraction samples/symbol
     (faster rate) whose residual is within max(0.15, 1.25 × best + 0.02) is preferred (strobing at k × sps
     also lands on valid symbols).
   The adopted front end is used for the receiver.

### 6.3 PSK / QAM order (`linear_classifier.classify_linear`)

Input: `s_core` with 5 edge symbols dropped at each end, normalised to unit RMS.

* **Per-hypothesis carrier recovery and score** (`_score`):
  * PSK: M-th power line (frequency + phase), then a 2nd-order PLL (loop bandwidth 0.005).
  * QAM (`_qam_carrier`): 4th-power line restricted to ±0.03 cycles/symbol; a grid of residual
    frequencies (step sized so the drift stays inside the outermost decision region) plus ±4 steps around
    the line; phase seeds every 64/√M degrees over ±45° × 4 or 7 grid scales; each seed refined by 4
    iterations of decision-directed complex least squares (phase + scale) and a phase-slope frequency
    update; score = per-axis quantisation residual on ≤ 128 central symbols. Bursts > 256 symbols also get
    a decision-directed PLL.
  * Decision-directed LS gain fit, noise variance, then the average log-likelihood of an equiprobable
    Gaussian mixture over the hypothesis's points (this includes the ln(1/M) prior, so nested hypotheses do
    not over-fit).
* **PSK vs QAM (σ_A² test)**: on the 16PSK-corrected symbols, amplitude variance σ_A² > 0.05 **and** radial
  spread > 3 × tangential spread → QAM; otherwise PSK.
* **ALRT within the family**: posterior ∝ exp(min(N, 100) × (LL − LL_max)); winner = maximum posterior.
* Reported evidence (not deciding): |C40| and C42 of the winner vs theory (BPSK 2, QPSK 1, 8/16PSK 0,
  16QAM 0.68, 64QAM 0.619, 256QAM 0.605); for QAM, EVM per order vs 12.5 % / 8 % / 3.5 % and amplitude
  kurtosis vs theory (on this data kurtosis runs 16 < 64 < 256QAM, opposite to the textbook, so it is
  evidence only); for PSK, number of phase clusters visited (> M/2 required) and the M-th power line
  strength from `psk_classifier`.
* Confidence = posterior(winner) × (0.5 + 0.5 × mean score of the non-ALRT checks). The unified confidence
  is (1 − p_fsk) × this value.
* **Memory.** The classifier measures every symbol against every constellation point: the
  nearest-point search in the gain fit, and the mixture log-likelihood. So do `Constellation.nearest_labels`
  and `demap_soft` (max-log LLRs). All of these are computed in blocks of 65 536 symbols (`CHUNK` in
  `dsp/constellations.py`). The results are unchanged, but memory no longer grows with
  symbols × points: the 256QAM hypothesis on a 4.57 M-symbol capture needed a single 8.7 GB array and
  failed with `_ArrayMemoryError` (2026-09-29). The classifier still processes **all** symbols of the
  capture, so on large captures it is the slowest stage (§20).

### 6.4 FSK fit rejected as PSK

After the FSK receiver runs, if its fitted modulation index h < 0.05 (`fsk_fit_is_single_carrier`), the
signal is a single carrier with phase jumps: it is re-classified on the linear branch with the
constant-envelope (phase-jump) rate path.

---

## 7. PSK receiver (`demodulation/psk_receiver.py`, `psk.py`)

### 7.1 Front end (`psk_front_end`, shared with the classifier and the QAM receiver)

* **L1**: block AGC to unit power; `estimate_spectrum` (Welch: a 64-sample pass for the bandwidth, then
  ~8-symbol segments); burst detection; DC = mean of noise-only samples outside the burst (if there is no
  noise-only region, DC is left for the symbol-domain estimate, §7.3).
* **L3**: estimators use at most 32768 samples at the burst centre.
  * Coarse CFO = midpoint of the −3 dB band (raised-cosine PSD is half its plateau at ±Rs/2 for any
    roll-off, so the −3 dB bandwidth ≈ Rs).
  * **Symbol rate** (`select_symbol_rate`): candidate 1 = the −3 dB bandwidth refined on the |x|² cyclic
    line within ±15 % (zero-padded FFT + dense DTFT refinement); further candidates = the 8 strongest |x|²
    cyclic lines of the filtered and unfiltered burst and their 2× and 3× multiples, each refined within
    ±2 %. Each candidate is scored by the amplitude CV of the strobed matched-filter output (PSK is constant
    envelope at the right rate). The spectral estimate is kept if within 1.1 × best + 0.01; otherwise the
    highest rate within tolerance is chosen.
  * Initial roll-off: least-squares fit of a raised-cosine PSD model A·RC(f; β) + N, β ∈ {0.1 … 1.0}.
* **L5a**: RRC matched filter (span 10 symbols); **Oerder–Meyr** timing: phase of the symbol-rate line of
  |mf|²; the rate is refined over ±0.5 % (401 points) when estimated; then a **Gardner loop** (TED
  `Re{conj(y(t−T/2))·(y(t) − y(t−T))}` normalised by the average power, PI loop, Bn·T = 0.004, cubic
  Lagrange interpolation). `s_core` = contiguous symbols with amplitude > 0.5 × median.

### 7.2 Decision-directed refinement (`refine_dd`)

For each roll-off in {0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0} (or the known one): matched filter and
a common timing offset over ±0.1 symbol (21 points), scored by EVM after M-th-power carrier removal on up to
2048 symbols. Bursts shorter than 200 symbols also get a rate scan (±1.2 %, step 0.02/N).

### 7.3 Carrier recovery

* **M-th power feedforward** (`mth_power_line`): `z = |s|·exp(jM·arg s)`, FFT (16× zero-padded), parabolic
  peak refinement; frequency = peak / M, phase from the DTFT at that frequency, referenced to the block
  centre (Viterbi & Viterbi with amplitude exponent 1). Line strength ρ is reported.
* **2nd-order PLL** (`carrier_pll`, loop bandwidth 0.01, damping 0.707): Costas detector for BPSK
  (`Re·Im`) and QPSK (`sgn(Re)·Im − sgn(Im)·Re`), decision-directed (`Im(y·conj(d))`) for 8PSK/16PSK.
* **Residual DC** (decision-directed, `remove_dc_dd`): D = mean((y − â)·e^{jθ}) with θ the total phase
  removed from each symbol; kept only if EVM improves by more than 2 %.
* **Capture edges** (`equalize_edges`): if the capture cuts the signal (burst touches the file start/end),
  the symbols within 5 symbols of the edge are re-solved by least squares using the exact truncated-pulse
  response and the neighbours' decisions.
* Reported CFO = coarse + M-th power + PLL residual (+ alias correction, below).

### 7.4 Ambiguity resolution and demapping (`decode_psk`)

After carrier recovery the constellation is correct up to a rotation 2πr/S (S = M for M-PSK, 4 for square
QAM) and, because the M-th power estimate is ambiguous modulo Rs/M, up to a carrier **alias** a·Rs/M.
Strategies, in order:

1. **Sync word** (coherent): for every alias a, correlate with the modulated 96-bit test sync word
   (§18.1); |normalised correlation| ≥ 0.6 and ≤ 12/96 bit errors; the correlation phase gives r.
2. **Differential**: differential detection `y_n·conj(y_{n−1})` + sync search (rotation-immune).
   Sync hypotheses are ranked: CRC valid, then fewest sync errors, then smallest alias.
3. **Known payload** (only if an expected payload is supplied; the GUI does not supply one).
4. **Text heuristic** (captures without a sync word): every (bit-label mapping × alias × rotation) is
   scored with the message models (§10: best of general / English / numeric, bit alignment 0–7, binary
   header and trailer trimmed). Mappings: Gray counter-clockwise (standard), Gray clockwise, natural
   counter-clockwise, natural clockwise. Standard labels are tried first; the others only if the standard
   set is not decisive. Accepted by `text_decisive` (§10): ≥ 6 bytes; for ≥ 24 bytes score ≥ −7.0
   nats/char and ≥ 0.5 above the best *different* payload; shorter readings need ≥ −5.5 and a lead of
   0.5 × √(24 / n).
5. Otherwise "unresolved" (rotation 0).

Demapping: nearest point (hard) and max-log LLRs `(min_{s:b=1}|y−s|² − min_{s:b=0}|y−s|²)/σ²`,
σ² = EVM². QPSK whose bits follow square-QAM labels ("gray-cw") is reported as **4QAM**. Es/N0 is
estimated from the MER (−20·log10(EVM)).

Constellation conventions (`dsp/constellations.py`): M-PSK point at angle φ0 + 2πk/M carries label
gray(k) = k ⊕ (k >> 1); φ0 = π (BPSK: bit 0 → −1), π/4 (QPSK), 0 (8PSK, 16PSK). Square QAM: Gray per
axis, label = [I half | Q half], unit average power, phase symmetry 4.

---

## 8. QAM receiver (`demodulation/qam_receiver.py`)

Used for 16/64/256QAM (from the GUI, always with the classifier's front end and order).

1. **Roll-off × timing refinement** (`refine_timing`): β ∈ {0.2 … 1.0}, common timing offset ±0.2 symbol
   (7 points) on ≤ 256 central symbols, scored by the square-grid lattice residual; bursts < 200 symbols also
   get a rate scan (±0.6 %, 25 points).
2. **Coarse carrier and scale**: `_qam_carrier` (§6.3).
3. **Whole-burst refinement** (`grow_fit`): in centred windows of 256, 512, 1024 symbols and the whole burst,
   the confident symbols (|y − d| < 0.35 d_min) give a weighted least-squares fit of phase + frequency and
   a gain.
4. **Tracking** (`dd_track_center_out`): 2nd-order decision-directed carrier loop (bandwidth 0.004) plus a
   gain loop (μ = 0.01), run from the burst centre outwards in both directions, updated only on confident
   decisions.
5. **Edges**: truncated-pulse least squares as in §7.3.
6. **Decoding**: `decode_psk` with the standard mapping (sync word over 4 rotations × 4 aliases, no
   differential); if no sync word, `decode_mapping_hypotheses`: 4 axis mappings for I (Gray/natural ×
   ascending/descending) × 4 for Q × I-first/Q-first × 4 aliases × 4 rotations, scored by the expected
   payload (not used by the GUI) or by the message models (`best_text_region`, §10) — first at bit
   alignments that are multiples of gcd(8, bits/symbol) (a transmitted byte stream starts on a symbol
   boundary), then at every alignment if that is not decisive; accepted by `text_decisive` as in §7.4.
7. **Sync-aided refit** (`sync_aided_refine`, 2026-09-27): when the sync word is found but the frame CRC
   fails, the located sync word's 96 known bits (12 symbols on 256QAM) give a data-aided phase and
   frequency (weighted least-squares line through their phase errors) and gain; the burst is corrected,
   optionally refitted decision-directed (`grow_fit`), and decoded again. The result is used only if the
   CRC then validates. Motivation: on a 21-symbol 256QAM burst the DD loop left a residual carrier of
   0.64 °/symbol (phase −5° → +12° across the frame).

---

## 8A. Feed-forward linear receiver (`demodulation/ff_receiver.py`)

`receive_feedforward(x, fs, modulation, sps, beta=0.35, center=None) → FFResult` (modulation,
symbols, LLRs, hard bits, EVM, Es/N0 estimate, parameters, evidence). It has the same interface as the
loop receivers' results, so the image-frame search (§15A), the tabs and the report use it unchanged.

**Why it exists.** The loop receivers (Gardner timing + Costas / decision-directed carrier) need symbol
transitions. An image sent without a scrambler contains long runs of one byte value; for example, a
white slide background is 0xFF…. That means thousands of identical symbols: an unmodulated carrier with
no timing information. The loops then drift and slip. The 2 M-sample QPSK screenshot capture lost lock
about 21 500 symbols in: one 90° carrier slip plus whole-symbol timing slips, which turned the rest of
the image into colour noise. This receiver has **no loops**.

**Processing** (the input is normalised to unit RMS):

1. **Carrier frequency.**
   * Coarse: the centre of the occupied band (§4.5), or the `center` argument.
   * Fine: the M-th power spectral line (x² BPSK, x⁴ QPSK / QAM, x⁸ 8PSK, x¹⁶ 16PSK) over up to 2²²
     samples, with parabolic peak interpolation.
2. **Matched filter.** Root-raised-cosine (β = 0.35, span 10 symbols), `fftconvolve`.
3. **Timing (Oerder & Meyr per block).**
   * The square-law spectral line e^(−j2πn/sps) is summed over blocks of 256 symbols. The **global**
     sample index is used: with a non-integer sps, a per-block index restarts the symbol grid at every
     block, the estimates jump, and drift goes untracked.
   * Blocks whose line magnitude is below 0.2 × the median carry no transitions and are excluded.
   * The timing offsets of the remaining ("good") blocks are fitted with a **weighted straight line**:
     a sample-rate / symbol-rate mismatch is a linear drift. The fit extends to the burst edges; holding
     the first block's value mis-sampled the frame header of the 64QAM test.
   * The 5-block running mean of the residual is added, interpolated across the blocks without
     transitions.
   * Samples are taken with cubic Lagrange interpolation at t_k = k·sps + τ(k).
4. **Carrier phase (Viterbi & Viterbi, feed-forward).**
   * **PSK:** the M-th power of the unit-magnitude symbols is averaged over a sliding window of 128
     symbols. Its angle is unwrapped continuously (the M-fold ambiguity stays continuous) and divided by
     M. Identical symbols do not matter here: their M-th power is constant.
   * **QAM:** image data is far from equiprobable (a white background is one corner symbol), so two
     changes are needed.
     * *Scale* comes from matching |s| to the constellation's ring radii: 241 gains from 0.25 to 4,
       geometric, on up to 50 000 symbols. The mean power is not used: it shrank the grid and put
       corners on inner points.
     * *Phase* uses the 4th power of the **outer-corner ring only**: |s| above the midpoint of the two
       largest radii, averaged over 512 symbols, with windows that have no corner symbol bridged by
       interpolation. The middle-ring points' 4th power is not at the corners' angle, and with the data
       mix changing along the image that bias rotated whole stretches. The 16QAM screenshot showed clean
       rings but smeared phase and 20 % EVM before this change.
5. **Gain and phase refinement.** Three decision-directed passes: least-squares gain against the
   nearest constellation points, then phase correction by the angle of the 256-symbol running mean of
   s·conj(decision). The M-th power estimate is noisy on dense QAM because of the self-noise of the
   inner and middle rings; the decisions give a clean phase error.
6. **Output.** EVM, Es/N0 = −20·log10(EVM), max-log LLRs at that noise level, hard bits. The parameters
   are samples/symbol, symbol rate, carrier offset (band centre + line) and EVM. The evidence records
   the number of timing blocks and how many had transitions.

The residual M-fold phase ambiguity (and conjugation) is resolved downstream by the file-frame search
(§15A), as for the loop receivers.

**When it is used.** Only as the fallback of §3.2. The worker tries the estimated rate, the nearest
standard symbol rate and the nearest integer samples/symbol. A result is adopted when its file frame is
CRC-valid, or when it is the only frame found.

---

## 9. FSK receiver (`demodulation/fsk_receiver.py`, `fsk_demod.py`, `estimation/fsk_estimator.py`)

Handles 2/4/8-FSK, CPFSK, MSK, GFSK, GMSK and phase-discontinuous FSK.

### 9.1 L1–L2

Block AGC; occupied band; burst detection; DC from noise-only samples; blind **IQ-imbalance correction**
(Gram–Schmidt whitening of I/Q, valid because CPFSK is a proper signal) applied only if it lowers the
envelope CV by more than 5 %. Family evidence from the envelope test (§6.1). Channel filter ± 0.6 × band;
instantaneous frequency `f[n] = arg(y[n+1]·conj(y[n]))/2π`.

### 9.2 Symbol rate and continuity

* `estimate_rate`: two transition-sensitive transforms of the IF at smoothing scales W ∈ {1, 2, 4, 8}:
  T1 = |f_W[n+W] − f_W[n−W]| and T2 = (f_W − mean)²; line spectra averaged; up to 8 candidate lines. A
  dominant **power-law line pair** (x^P lines at ±tone, exact for P·h integer, e.g. MSK/GMSK) is ranked
  first. Harmonic relatives (×2, /2) of the top lines are interleaved. The winner is refined, and the symbol
  boundary found, by a decision-directed re-modulation scan (`dd_refine`).
* **Phase continuity**: mean |IF| per symbol-phase bin of the unfiltered IF; discontinuous if the peak bin
  exceeds the median by > 0.05 cycles/sample and > 6 standard errors. Scanned over ±1.2 % of the rate
  (97 points). Discontinuous signals use the jump-spike rate estimate.
* **Whole-burst timing tracking** (`track_rate`, bursts ≥ 2 × 250 symbols): boundary phase per 250-symbol
  segment, line fit to the offsets; the slope corrects the rate (rejected if |slope| > 0.003 or the RMS
  residual > sps/8).

### 9.3 Tone model (`fit_tone_model`, per rate hypothesis)

* IF integrate-and-dump strobes (central 60 % of each symbol, median).
* M ∈ {2, 4, 8}: equally spaced level fit (spacing = h, centre = residual CFO) with a Gaussian-mixture
  log-likelihood.
* Pulse hypotheses: rectangular, and (continuous phase, M ≤ 4) Gaussian with BT ∈ {1.0, 0.7, 0.5, 0.4, 0.3,
  0.25, 0.2}. Gaussian decisions use an ISI Viterbi (MLSE through the pulse's strobe ISI taps).
* Each (M, pulse) gets a coarse timing scan (8 offsets over one symbol), an h scan (rect: 0.9–1.1 ×
  spacing, 9 points; Gaussian: 0.9–1.9 ×, 11 points) and a fine timing scan (±1/32 symbol), scored by
  **re-modulation match** (per-symbol non-coherent correlation between the signal and a CPM model built from
  the decisions; for discontinuous signals per symbol).
* Selection: M by a minimum-description-length rule (a larger alphabet must improve the match by
  ln(M/2)·NSR/sps; smallest M within 0.001); pulse: rect unless a Gaussian beats it by more than
  max(0.0003, 0.15·NSR/√n_windows). Then 2 rounds of fine h / timing refinement.
* Variant name (`dsp/cpm.fsk_variant_name`): M = 2, continuous, h ≈ 0.5 (±0.03) → MSK (rect) / GMSK
  (Gaussian); other Gaussian → GFSK / MGFSK; otherwise "MFSK".

### 9.4 Demodulators

Each produces a per-symbol tone log-likelihood matrix LL (N × M):

* **Discriminator**: Gaussian model of the IF strobes around the tone centres.
* **Non-coherent correlator bank**: LL = 2|y_k|/N0 per tone.
* **Coherent max-log BCJR** over the CPM trellis (continuous phase and h = p/q with q ≤ 16; h is snapped to a
  fraction by minimum |h − p/q| + 0.003·q within 8 %): state = (accumulated phase j·π/q, previous L−1
  symbols); the carrier-phase reference comes from a model built from the non-coherent decisions, smoothed
  over 16 symbols; one decision-directed iteration.

### 9.5 Bit mapping and decoding (`decode_fsk`)

Mapping hypotheses: {Gray, natural (M > 2)} × {normal, spectrally inverted} × {absolute, differential}.
For each: bits and max-log LLRs from LL, sync-word search (≤ 12/96 errors); ranked CRC valid, fewest sync
errors, simplest hypothesis. Without a sync word: expected payload (not used by the GUI) or the message
models (`best_text_region`, §10: bit offset 0–7, header / trailer trimmed) over every hypothesis; the
runner-up is the best hypothesis with a *different* payload; accepted by `text_decisive` as in §7.4. The primary demodulator is the one with a
CRC-valid frame / sync word / fewest sync errors, preferring coherent-BCJR, then correlator, then
discriminator.

### 9.6 Backtracking over rates

The best-ranked rate gets a full fit first; if it gives no CRC-valid frame (or accepted text), the other
candidates (up to 10) are screened cheaply (IF-strobe decisions at 4 timing offsets + sync search), and up
to 4 get full fits. If nothing validates, the attempt with (CRC, sync word, best match) is kept, with one
extra rule: when the decisions of the chosen attempt repeat in aligned groups of k (≥ 90 %) and another
attempt has k × its samples/symbol, the slower attempt is chosen (2FSK with h = 1 at Rs produces the same
waveform as MSK at 2Rs with every symbol repeated; this happens with FEC-coded, random-looking data).

Reported: variant, samples/symbol, symbol rate, M, h, tone spacing, peak deviation, CFO, pulse/BT,
continuity, Es/N0 (from the re-modulation residual) and the FSK decision error %.

### 9.7 Tone-correlator fallback (`demodulation/fsk_tones.py`, `_fsk_tone_fallback` in the worker)

**Why it exists.** The blind FSK receiver fits its tone model from the instantaneous frequency. At few
samples per symbol, a continuous-phase h = 0.5 2FSK (MSK-like) signal shows smeared, multi-level
frequency tracks. The "CubeSat UHF" preset of the Space Downlink Simulator (96 kHz, 19.2 kBd,
5 samples/symbol) was fitted as 4FSK with h = 0.17, so every bit was wrong.

**When it runs.** The FSK decode is not verified and the image-frame search found no CRC-valid frame.

**Hypotheses.**

* M ∈ {2, 4, 8} plus the fitted M.
* h ∈ {0.5, 1.0} plus the fitted h, when the fitted h is more than 0.05 away from both.
* Each (M, h) pair at the estimated rate and at the exact-rate hypotheses of §3.2.

**Per hypothesis** (`tone_decisions`):

1. Tones f_k = (k − (M − 1)/2)·h·Rs around the occupied-band centre.
2. Every tone's mixed signal is integrated with a running sum (cumsum).
3. The per-symbol correlation energy is |S(t + sps) − S(t)|². The window edges are **fractional**:
   the cumsums are linearly interpolated at non-integer positions. Rounding the edges to whole samples
   made the result flip with the 10th decimal of sps: at a half-sample offset, `np.round` alternates
   (round-half-to-even), so sps = 5.0 decoded and 4.9999999988 did not.
4. Timing offset: the one of 8 fractions of a symbol that maximises the mean best-tone energy share,
   measured on the first 2¹⁸ samples.
5. Decision: the tone with the largest energy; natural-binary bits.

**Selection.** Each hypothesis's bits go through the file-frame search (§15A), including its FSK label
hypotheses (natural / Gray / inverse-Gray, tones in normal or reversed order). The first CRC-valid
frame wins, and the reported modulation becomes `<M>FSK (h = <h>, tone correlator)`. If only a
CRC-unverified image was found, it is kept.

---

## 10. Unframed text payloads — the message models (`demodulation/psk.py`)

Used by all three receivers when there is no sync word, to choose rotation / alias / label mapping / bit
alignment. Per-byte log-probability model: 25 % of the probability mass spread uniformly over all printable
ASCII (32–126, TAB, LF, CR), plus English letter frequencies (upper and lower case share each letter's
frequency), space 0.16, digits and common punctuation; non-printable bytes 1e-5 (all then normalised).
Per-byte score = log-probability − 5 × [byte not printable] − 1 × [byte outside letters, digits, space and
`.,;:!?'"-()\n`]. This is the **general** model.

Two further message models (2026-09-27); a reading's score is the **maximum** over the three, so they only
raise scores:

* **numeric**: 70 % digits, 12 % space, 12 % the separators `. , - + : / ( )`, 6 % any other printable byte,
  same non-printable penalty. Under the general model a digit is a rare character, so a wrong rotation that
  spells lowercase junk (`canlomfdge…`) out-scored the true `9876543210…`; under this model true digits score
  about −2.6 nats/char.
* **English**: the general score plus 0.7 × the mean pointwise-mutual-information bonus of case-folded
  letter/space pairs (add-one-smoothed letter bigram of a generic prose sample embedded in `psk.py`, not any
  test payload), −2 for a lowercase→uppercase switch inside a word. It separates 16/256QAM English from the
  near-text readings of wrong label mappings (`PACO I] BK\ WMTL…`, previously within 0.3 nats of the truth).

`best_text_region` tries bit alignments 0–7, skips a binary header (up to the last byte scoring ≤ −10,
i.e. non-printable, among the first 8) and cuts a trailer (from the first such byte among the last 8: guard /
ramp-down symbols, padding) for streams ≥ 16 bytes; the score is taken on the kept bytes, the reported payload
starts at the kept start and still contains the trailer. It is used by all three receivers (the QAM and FSK decoders previously scored
only the unskipped stream). The QAM decoder first searches alignments that are multiples of gcd(8, bits per
symbol), because a transmitted byte stream starts on a symbol boundary, then all alignments. Without this, an
I/Q-swapped 256QAM reading shifted by half a symbol spells the same digit string shifted by one nibble.

Acceptance (`text_decisive`): readings shorter than 6 bytes are never text; for ≥ 24 bytes, score ≥ −7.0 and
≥ 0.5 ahead of the best different payload; below 24 bytes, score ≥ −5.5 and a lead of 0.5 × √(24 / n).
Measured: real text with digits and special characters scores −2.6 … −6.6, random bytes below −10. Random
8–25-byte symbol streams read as text: 1/300 (8PSK) and 0/300 (16PSK), versus 24/300 and 16/300 with the
previous rule; 0/30 for 16/64/256QAM (`tests/unit/test_text_models.py`).

---

## 11. L8 — Bitstream analysis

Run in the joint search (§15) on the receiver's bit stream; also available manually on the current bits.

### 11.1 Sync-word search (`bitstream/sync_search.py`, after Chiani & Martini 2006)

Library: test-frame PN127/96 sync, CCSDS ASM 0x1ACFFC1D, 0xD3917E42, 0xFAF33420, 0xEB90, 0xF628,
Barker-13, 0xAA55. Words shorter than 16 bits are skipped by default (so Barker-13 is not searched). Every
word and its complement is slid over every bit offset. With s_i = ±1 and received LLRs l_i (hard bits give
l = ±1):

* hard correlation (L − 2·d_H)/L and Hamming distance d_H;
* soft correlation Σ s_i l_i / Σ |l_i|;
* **LRT** Σ [s_i l_i / 2 − ln cosh(l_i / 2)] (log-likelihood ratio of "sync here" vs random equiprobable
  data given exact LLRs);
* **GLRT** Σ (s_i l_i − |l_i|) / 2.

Detection threshold: the largest Hamming distance e such that (2 × number of positions) × P(Binomial(L, ½)
≤ e) ≤ 1e-3. Occurrences at or below the threshold (non-overlapping) are listed; the most common spacing is
the **frame period**. P(emulation) = chance of a random match this good anywhere; confidence High < 1e-6,
Medium < 1e-3.

### 11.2 Other L8 measurements

* Randomness (`bitstream/entropy.py`): ones fraction, binary entropy, byte entropy, randomness score.
* Bipolar autocorrelation up to lag 512 and its local peaks ≥ 0.15 (`bitstream/correlation.py`).
* Frame length estimate: sync recurrence period, else the strongest autocorrelation peak.
* GF(2) rank profile (§12.2) and the blind convolutional-code estimate (§13.4).
* Repeated 16–32-bit patterns and activity bursts (`patterns.py`, `framing.py`, manual re-run only).

---

## 12. L9 — De-interleaving (`deinterleaving/`)

### 12.1 Interleaver models

* **Block (rectangular) R × C**, period P = R·C: the transmitter writes a block row by row and reads it
  column by column (`blk.reshape(R, C).T`); the last block is zero-padded. "Write columns, read rows" on
  R × C is the same permutation as "write rows, read columns" on C × R, so searching ordered (R, C) pairs
  covers both orientations. De-interleaving works on whole blocks from a given bit offset.
* **Convolutional (Forney) B, M**: branch i delays its bits by i·M commutator cycles; the de-interleaver uses
  (B − 1 − i)·M; end-to-end delay (B − 1)·M·B bits. The transmitter flushes with that many zero bits; the
  receiver drops them after de-interleaving. Works on bits or LLRs (dtype preserved).
* **Diagonal (helical) R × C** (`diagonal.py`): per block the bits are written row by row into R × C and read
  along the wrapped diagonals — for j = 0 … C−1, for r = 0 … R−1: M[r, (j + s·r) mod C], s = +1
  (down-right) or −1 (down-left). This is a permutation for every R, C. The expression
  π(k) = ((k mod R)·C + (k mod R) + ⌊k/R⌋) mod (R·C) found in some descriptions is **not** a permutation
  (it maps two positions to the same one for every R, C ≥ 2, checked for 2 … 32) and is not used.
* **Pseudo-random** (`pseudorandom.py`):
  * LFSR address-generator family: an n = ⌈log2(P+1)⌉-bit Fibonacci LFSR with a primitive feedback polynomial
    (one per n = 2 … 16; maximal length verified at import) runs from a seed; each state s gives address s − 1;
    addresses ≥ P are skipped; the first P addresses are the permutation (out[i] = in[addr[i]]).
  * QPP permutation π(i) = (f1·i + f2·i²) mod K for given (K, f1, f2), with a validity check (no standard
    parameter table is included and QPP is not searched blindly).

### 12.2 Blind estimation

* **GF(2) rank** (`gf2.rank`): Gauss–Jordan elimination on rows packed into 64-bit words.
* **Rank-deficiency period search** (Burel & Gautier 2009): the stream is cut into rows of L bits
  (L = 2 … min(128, √N − 4), rows up to 2L + 16, only where rows ≥ L + 16); deficiency (L − rank)/L. Coded
  data keeps its parity constraints inside a row when L is a multiple of the interleaver period and the rows
  are aligned, random data gives full rank. Period = the smallest deficient L whose tested multiples are
  ≥ 75 % deficient (else the most deficient L); the alignment offset is the one of 0 … P−1 with the largest
  deficiency (this offset × deficiency curve is the plot in the De-interleaving tab).
* **K–S score** (Wee, Choi & Jeong 2021): ranks of non-overlapping L × L matrices vs the exact
  random-binary-matrix rank distribution; D = max |F_obs − F_random|; needs ≥ 3 matrices (N ≥ 3L²).
* Because the rank method needs N > P², larger interleavers are **hypothesised from the burst length**
  (`dimension_hypotheses`): all (R, C) with 2 ≤ R, C ≤ 128 such that a whole number of blocks fits the data
  span with ≤ 16 spare bits.
* **Convolutional interleaver** hypotheses: B = 2 … 16, M = 1 … 16 with flush length ≤ N/2.
* **Diagonal** hypotheses: the same (R, C) pairs as the block search, both diagonal directions.
* **Pseudo-random** hypotheses: LFSR permutations for the rank-deficiency period (all seeds when n ≤ 10) and
  for the data span taken as ONE interleaver block (P within 16 bits of the span, seeds 1 and all-ones).
* **PR presence**: if a rank-deficiency period exists and no structured interleaver validates, the report
  states that a pseudo-random interleaver of that period is suspected. An arbitrary permutation is **not**
  reconstructed; `fec/dual.py` can recover the (permuted) code's parity checks from enough aligned blocks,
  but the permutation itself is not derived from them.
* Every hypothesis is only accepted when FEC decoding and a CRC confirm it (§15).

---

## 13. L10 — FEC (`fec/`)

### 13.1 Convolutional codes (`fec/convolutional.py`)

Library (generators in octal, MSB = tap on the current input):

| Name | K | Generators | Rate |
|---|---|---|---|
| NASA/CCSDS | 7 | 171, 133 | 1/2 |
| | 3 | 7, 5 | 1/2 |
| | 4 | 15, 17 | 1/2 |
| | 5 | 23, 35 | 1/2 |
| | 6 | 53, 75 | 1/2 |
| IS-95/3GPP | 9 | 561, 753 | 1/2 |
| LTE-style | 7 | 133, 171, 165 | 1/3 |
| | 3 | 7, 7, 5 | 1/3 |
| DVB/CCSDS punctured | 7 | 171, 133, puncture 10/11 | 2/3 |
| DVB/CCSDS punctured | 7 | 171, 133, puncture 101/110 | 3/4 |
| DVB punctured | 7 | 171, 133, puncture 10101/11010 | 5/6 |

* Encoder: c_j = u * g_j (mod 2), zero-tail termination (K − 1 zeros), puncturing column by column.
* **Soft-decision Viterbi**: states = 2^(K−1); branch metric = ½ Σ LLR_j · (1 − 2c_j); punctured positions
  are erasures (LLR 0); known or unknown start state; traceback from state 0 (terminated) or the best state.
  Quality outputs: **re-encoding mismatch** (fraction of non-erased received hard bits that differ from
  re-encoding the decoded path; ≈ channel BER when right) and the normalised path metric.
  `decode_unknown_phase` tries every bit offset within one puncturing period and every puncturing phase.
* **Syndrome test** (unpunctured rate 1/n): for every pair (1, j), c_1 * g_j + c_j * g_1 = 0. The fraction
  of violated checks is ≈ 2 × BER when (code, phase) is right and ≈ 0.5 otherwise. Checks whose span is all
  zeros are skipped (the all-zero sequence satisfies every code: transmitter padding and interleaver flush
  would otherwise look like coded data). For codes whose generators have different weight parities, a
  complemented stream gives a weight near 1, so w > 0.5 is read as "inverted stream" (the decoder then uses
  −LLR).
* **Random baseline** for punctured codes: the re-encoding mismatch of Viterbi on 2400 random bits (fixed
  seed) is computed once per code; a punctured hypothesis counts only if 1 − mismatch/baseline ≥ 0.5
  (high-rate punctured codes fit random data fairly well).

### 13.2 Reed–Solomon (`fec/reed_solomon.py`, `reedsolo` backend)

Profiles over GF(2⁸), primitive polynomial 0x11D, first consecutive root 0: RS(255,223), RS(255,239),
RS(204,188) (shortened 255,239), RS(255,247). Messages longer than k bytes are split into k-byte blocks; the
last block is shortened. Blind decoding tries every profile × bit offset 0–7 × MSB/LSB-first byte packing;
the length of the final shortened block is searched (fewest corrections wins; skipped when all full
blocks already failed). Ranking: fraction of blocks decoded, then fewest corrected symbols. Note that
RS(204,188) and RS(255,239) are the same code (same parity length) and cannot be told apart; the first in
the list (RS(255,239)) is reported. The RS search is also run on the complemented stream (RS is not closed
under bit inversion).

### 13.3 Hamming codes (`fec/block_codes.py`)

Hamming(7,4), (15,11), (31,26), systematic G = [I | P]. Syndrome decoding with single-error correction.
Evidence: fraction of non-zero n-bit blocks with a non-zero syndrome at each alignment offset (≈ n·BER when
right, ≈ 1 − 2^−(n−k) when wrong; all-zero blocks skipped).

### 13.4 LDPC codes (`fec/ldpc.py`)

* Library: **IEEE 802.11 n = 648, rate 3/4 (Z = 27)** — prototype matrix taken from the MathWorks
  `ldpcEncode` / `ldpcDecode` documentation (block size 27, codeword 648, 486 information bits); two
  project-generated regular (3,6) Gallager codes, n = 96 and n = 504 (NOT standard codes). Other standard
  matrices (802.11 rate 1/2, 2/3, 5/6, 802.16e, DVB-S2, 5G NR) are not included; `from_prototype` /
  `LDPCCode(H)` accept them when a verified copy is available.
* Quasi-cyclic expansion: prototype entry s ≥ 0 → Z × Z identity shifted right by s; −1 → zero block.
* Systematic encoder: H is reduced over GF(2) with pivots taken from the last columns first; the non-pivot
  columns are the information positions (the first k columns for the 802.11 code); parity = M·u.
* Decoder: normalised min-sum (α = 0.75), flooding schedule, up to 50 iterations, early stop on a zero
  syndrome. Measured: the 802.11 code decodes error-free at Eb/N0 = 3 dB in the test (raw BER 3.5 %).
* **Codeword alignment** (blind frame synchronisation): for every offset 0 … n−1 the fraction of unsatisfied
  checks over up to 4 non-zero codewords; significance z = (0.5 − f)/(0.5/√checks); accepted when z ≥ 5
  (random data reaches about 3.5); the CRC decides.

### 13.5 Blind parity-check recovery (`fec/dual.py`)

For B aligned blocks of n bits (B > n) the null space of the block matrix over GF(2) is the dual code. Sparse
checks are searched information-set style: the null-space basis is put in reduced row-echelon form over a
random column order (40–60 orders), rows and pairwise sums of the 40 lightest rows with weight ≤ w_max are
kept, and a linearly independent subset forms H_est. In the analyser the candidate codeword lengths are the
three most rank-deficient row lengths of the stream (§12.2), the alignment offset comes from the rank
deficiency, and H_est is decoded with min-sum; information bits are taken from the leftmost information set
of H_est (correct when the code is systematic in its first k positions). Needs error-free blocks and more
codewords than n (e.g. ≥ 105 codewords of n = 96). Measured: 15 frames coded with an unknown (3,6) n = 96
code — all 46 checks recovered, all 15 frames CRC-valid.

### 13.6 Concatenated codes (`fec/concatenated.py`)

Outer RS (§13.2) → outer byte interleaver → bits → inner convolutional code (§13.1). Outer interleavers:
`forney:I,M` (Forney byte interleaver, I branches, branch j delays j·M bytes, flushed with (I−1)·M·I zero
bytes), `depth:I` (the data is split into I equal shortened codewords sent column by column, CCSDS-style
symbol interleaving), or none. Profiles: "DVB-S structure" = RS(204,188) + Forney I = 12, M = 17 + CC K = 7
(171,133) r1/2; "CCSDS structure" = RS(255,223) with depth 4 or 1 + the same inner code. These reproduce the
structure only: energy dispersal / randomisers, sync-byte handling and the CCSDS symbol inversion are not
modelled.

Decoding: (1) soft Viterbi of the inner code; (2) outer de-interleaving and RS decoding of every codeword;
(3) failed codewords are retried with erasures on the least reliable bytes (reliability = sum of |LLR| of the
coded bits each input bit drives), keeping a 4-symbol detection margin (f ≤ nsym − 4 and 2e + f ≤ nsym − 4,
e counted from the actual changes — without the margin, erasure decoding "succeeded" on wrong data);
(4) the bits of every decoded codeword are fed back to the Viterbi decoder as a-priori input LLRs (±30) and
steps 1–3 repeat (up to 3 iterations while the number of decoded codewords grows). Codewords that are all zero
before or after correction are ignored (padding and interleaver flush decode to the zero word under every RS
code and would otherwise validate wrong hypotheses). Blind outer search (`search_outer`): every RS profile ×
{none, depth 2 … 8, Forney with I·M = n} on the Viterbi output, ranked by the fraction of non-zero codewords
decoded (the code with more parity wins ties, since RS codes with the same first root are nested).
Measured: all three profiles decode error-free up to ≈ 10 % raw BER in the unit test; with σ = 0.85 noise the
feedback raised the decoded DVB-S-structure codewords from 20/30 to 27/30 (frames correct 3 → 5 of 6); no gain
for depth 1; the erasure step did not rescue any codeword in these tests (the reliability measure is coarse
without a soft-output Viterbi); no wrong decode was accepted.

### 13.7 FEC identification (`fec/identify.py`)

* `structure_scan(bits)` (a few ms): syndrome weight of every unpunctured convolutional code at every phase
  and the Hamming failure rate at every offset. Each best result is converted to a significance
  z = (random expectation − observed) / its standard deviation over the checks actually evaluated;
  strength = clip((z − 6) / 10, 0, 1) (0 below ~6σ, 1 above ~16σ).
* **Blind convolutional parameters** (Marazin, Gautier & Burel 2011): from the rank profile of the raw
  stream, the deficient row lengths are multiples of n; the first deficient L0 and the slope of the
  deficiency give k (slope ≈ (n−k)/n) and, for k = 1, K ≈ (n−1)(L0/n − 1) + 1. Verified on the library
  codes (correct rate and K for all non-degenerate codes; the (7,7,5) code has a repeated generator and
  gives K = 1). Valid only at low BER and without interleaving.
* **Family probabilities** (uncoded / convolutional / block): scores = conv strength, max(Hamming strength,
  RS decode fraction), and uncoded = 1 if a frame validates in clear, 0.5 if a sync word is visible in clear,
  otherwise 0.3 × (1 − max structure strength); normalised. When a CRC validates a decoded frame, the
  confirmed family (uncoded, convolutional, block, LDPC or concatenated) is set to 0.97 and the others share
  0.03.

---

## 14. L11 — Framing (`protocol/`)

### 14.1 CRC catalogue (`protocol/crc.py`)

Rocksoft parameters, table-driven: CRC-16/CCITT-FALSE, CRC-16/XMODEM, CRC-16/KERMIT, CRC-16/X-25,
CRC-16/ARC, CRC-16/MODBUS, CRC-32/ISO-HDLC, CRC-32C, CRC-8/SMBUS, CRC-8/MAXIM. All reproduce their check
values for "123456789" (`self_test`).

### 14.2 Frame layout discovery (`protocol/frames.py`)

For a detected sync word (all its occurrences), the layout is found by CRC validation. Layouts tested
(220): length field {16-bit big-endian, 16-bit little-endian, 8-bit, none} × every CRC × CRC stored big- or
little-endian × CRC coverage {length + payload, payload, sync + length + payload} (without a length field
the frame length is the sync spacing or the rest of the stream). A layout is accepted only if chance cannot
explain it: (number of layouts of that CRC width) × (2^−width)^(frames validated) must be ≤ 0.01 — so a
single frame can be validated by CRC-16/32 but not by CRC-8 alone. The layout validating the most frames
(wider CRC on ties) wins; every frame is then parsed (start bit, sync errors, length, payload, CRC received
and computed, valid). If no layout validates, frames are still reported as "unverified".

`Frame.crc_ok` also rejects a trivial frame: an all-zero (or empty) payload with a received CRC of 0
(`_trivial`). Under every CRC with zero init and zero xorout, such a frame "validates". An LDPC decoder
converging to the zero codeword produced it, and it was reported as CRC-valid before this check.

---

## 15. Joint post-demodulation search (`postdemod/analyzer.py`)

### 15.1 Bit-stream hypotheses (`hypotheses_from_receiver`)

* PSK/QAM: the receiver's own output first, then every other carrier rotation k·2π/S of the receiver's
  symbols (hard bits and max-log LLRs recomputed, noise variance from the receiver's Es/N0). The data span
  is found from **symbol amplitude**: first/last run of 3 symbols above 0.5 × the smallest constellation
  amplitude (the receivers also output a few guard/noise symbols at the burst edges).
* FSK: the receiver's output and its polarity inverse. The data span comes from |LLR| (first/last position
  where the 8-bit moving mean exceeds 0.35 × the median).

### 15.2 Search order

1. **L8 report** on the first hypothesis (§11).
2. **Stage A — raw stream**, for each hypothesis: sync search + frame/CRC analysis. If a frame validates:
   the RS decoder is still run (systematic RS keeps data readable) and reported if it decodes all blocks
   with a valid CRC; otherwise the stream is reported as uncoded. If a sync word is visible but the CRC
   fails, the body after the sync word is kept as a coded-block candidate (sync outside the code, CCSDS CADU
   style); the detected sync word is put back in front of the decoded body for framing.
3. **Stage B — code structure without de-interleaving**, on the data span and on the body after a sync:
   structure scan → Viterbi for the best unpunctured convolutional code (if it has strength; if its frames
   do not validate, the concatenated outer-layer search and iterative decoding of §13.6 follow) → Hamming
   decoding (if strength) → LDPC library codes (alignment z ≥ 5, min-sum) → RS search → punctured codes
   (Viterbi at every offset/phase, first hypothesis and bodies after a sync only). Each decoded stream goes through sync search + frame/CRC analysis; the
   first CRC-valid result ends the search. The slower searches are skipped once a decode validates.
4. **Stage C — de-interleaving**, for each hypothesis (hypotheses that are the bitwise complement of one
   already searched are skipped):
   * rank-deficiency period estimate on the first hypothesis; its factorisations are tried first;
   * then block (R, C) from the burst length, convolutional (B, M), diagonal (R, C, both directions),
     pseudo-random LFSR permutations (§12.2);
   * each hypothesis is screened at the data-span start with `structure_scan` + a sync search (a sync word in
     clear counts fully only if its frame validates, 0.5 otherwise); when it shows structure (strength > 0
     or z ≥ 6) its offset is refined over 0 … 32 bits starting 8 bits before the span start;
   * a candidate with strength ≥ 0.9 is decoded immediately (stage B decoders); a CRC-valid result ends the
     search, otherwise the search continues (a structured but wrong candidate, e.g. a divisor of the true
     period, does not stop it); at the end the best 4 candidates per hypothesis are decoded.
5. **Stage D** — if a rank-deficiency period was found and nothing validated: the pseudo-random-interleaver
   evidence is added, and blind parity-check recovery + min-sum decoding (§13.5) is tried for the three most
   deficient row lengths.
6. If nothing validates, the chain with the best (CRC frames, frames found, strength, simplicity) is reported
   as **unverified**.

Budget: at most 12 000 de-interleaver screenings (deterministic) and a wall-clock safety limit (600 s in the
GUI). The result object carries the L8 report, family decision, structure scan, de-interleaver candidates
(stream, type, parameters, offset, strength, detail, validated), period estimate, FEC candidates (stream,
family, code, setting, metric, result, CRC frames), the best chain, number of hypotheses tested, time and a
search log. The worker replaces the receiver's payload and CRC flag with the analyser's only when a CRC
validates.

---

## 15A. Image / file-frame recovery (`postdemod/image_payload.py`)

A received file (typically an image) is carried in a byte frame with a CCSDS attached sync marker:

```text
[PREAMBLE 0xAA x 4]  | SYNC 0x1ACFFC1D | LENGTH (u32, big-endian) | PAYLOAD | CRC-32
 (c.py only)            32 bits          bytes of PAYLOAD
```

The frame is sent either uncoded or with the K = 7 (171, 133) rate-½ convolutional code, which covers
the whole frame including SYNC (`synthetic/tx_image.py`). The generic L8–L11 search (§15) does not
cover this layout, because it tries 8/16-bit length fields, and it would report binary payloads only as
"no decisive text". So this frame is searched for directly, on every PSK / QAM / FSK decode (§3.2).

### 15A.1 Two transmitter conventions

| | Project transmitter (`synthetic/image_framing.py`, `tx_image.py`) | Space Downlink Simulator (`Image Decoding/c.py`) |
|---|---|---|
| CRC-32 coverage | SYNC + LENGTH + PAYLOAD | PAYLOAD only |
| Preamble | none | 0xAA × 4 before SYNC |
| FEC | optional K = 7 r½ | none |
| Pulse / IF | RRC 0.35; 48 kHz passband WAV or .iq | RRC 0.35, IF offset 0.25·fs; stereo I/Q WAV, real WAV or .iq |
| PSK bit labels | Gray | point = exp(j2π·gray(i)/M), i.e. the bits are the **inverse Gray** of the phase index |
| QAM bit labels | Gray (I/Q) | axis level = gray(i) per axis |
| FSK | continuous-phase 4FSK, h = 0.5 | tone index = natural binary; h = 0.5 (tone spacing Rs/2) |
| Last partial symbol | padded | dropped (`nb = len(bits) // k * k`), so up to k − 1 CRC bits are never sent |

### 15A.2 Payload contents (`identify_image`)

* **Encoded image:** PNG, JPEG, GIF, BMP, TIFF or WebP, recognised by its magic bytes. It is accepted
  only if Pillow decodes it **completely**, so a truncated or corrupt file fails.
* **5-byte header + encoded image:** W (u16), H (u16), C = 0, then an encoded image whose decoded size
  must equal W × H. This is c.py's "compress payload (PNG)" option.
* **5-byte header + raw pixels:** W (u16), H (u16), C ∈ {1, 3, 4}, then W·H·C bytes. The header must
  predict the payload length **exactly**. Both big- and little-endian are tried, and W, H ≤ 16384.
* Anything else is reported as an N-byte binary file.

`ImagePayload.file_bytes` removes the 5-byte header in front of an encoded image, so a saved PNG /
JPEG opens in any viewer. Raw pixels are saved as PNG. `describe()` gives, for example:
`RAW image 256x165x3, 126 725 bytes; CRC-32 VALID; FEC none; rotation +90 deg`.

### 15A.3 Search order (`find_image(family, r)`)

**1. Soft hypotheses** (`llr_hypotheses`).

* PSK / QAM: the receiver's LLRs, then every M-fold rotation of the receiver's symbols, with and without
  conjugation. The LLRs are recomputed with max-log demapping at the receiver's Es/N0. These rotations
  are needed because the receivers resolve the phase ambiguity with the project's PN127 sync word,
  which a file frame does not carry.
* FSK: the receiver's LLRs and their inversion.

For each hypothesis:

* **Uncoded.** The ASM is correlated over all hard bits (every offset, both polarities). A hit needs
  ≤ 2 of 32 bit errors.
* **Coded.** The first 32 768 LLRs are Viterbi-decoded (K = 7, unterminated, unknown start state) for
  both trellis phases, and the ASM is searched in that output. Only on a hit is the whole stream
  decoded.

**2. Hard label hypotheses** (`label_hypotheses`, uncoded). These cover other transmitters' bit
labellings on hard decisions. The ASM is searched in the first 2¹⁷ bits only, because it opens the
frame.

* **8PSK / 16PSK:** phase index p for every rotation (M) × conjugation (2), with inverse-Gray, Gray and
  natural labels. For BPSK / QPSK every Gray labelling is a rotation, so none are added.
* **QAM:** per-axis level indices for 4 rotations × conjugation, with inverse-Gray, Gray and natural
  labels, I-first or Q-first.
* **FSK:** tone positions in normal or reversed order, with the three labellings.

**3. Frame parse** (`_parse`). LENGTH must fit the stream. If the stream ends inside the CRC, by up to 7
bits (`CRC_TAIL_BITS`), the missing bits are padded. This happens when a burst ends with its last symbol,
or when c.py dropped the bits that did not fill a symbol (a 2FSK frame was 1–2 bits short).

**4. Acceptance** (`_accept`). A frame is accepted in either of these cases:

* **CRC valid.** The received CRC equals the CRC-32 over SYNC + LENGTH + PAYLOAD or over PAYLOAD. A
  CRC whose top ≥ 25 bits match (the last ≤ 7 bits were not transmitted) also counts; chance agreement
  is 2⁻²⁵, and the evidence says the tail was not transmitted.
* **CRC not valid, but the payload is a structurally complete image** (it decodes completely, or the raw
  header predicts the exact length). This is reported as "CRC not verified".

**5. Ranking.** The search returns the first CRC-valid frame. Otherwise it returns the best candidate by
(CRC valid, is an image, fewest sync errors).

The worker caches the result on the receiver result (`r._image_frame`), so repeated checks do not
search again.

**Evidence** added to the Modulation parameter:

| Tag | Content |
|---|---|
| `IMAGE_FRAME` | sync position, sync errors, hypothesis, FEC |
| `CRC` | coverage and value, or the mismatch with both computed values |
| `IMAGE` | format, size and why the structure is consistent |

A "Recovered image" (or "Recovered file") parameter is added. The decoded-payload text becomes
`[<describe()>]`.

### 15A.4 Image quality

The recovered image is the transmitted pixels, bit-exact when the CRC is valid (§19.1). Its resolution
is the resolution the transmitter sent. c.py scales the source image down to `max_dim` before
transmission, for example 256 px for the screenshot capture, so the source image's detail above that is
not in the signal. The Image tab's "Smooth (Lanczos)" view and "Save enlarged ×4" interpolate for
readability; they add no detail.

---

## 16. Report generation (`reporting/report.py`)

`build_report(results)` collects: input (file, sample type, sample rate and its source, capture notes),
number of SOIs, classification (family, modulation, confidence, up to 20 evidence items), all parameters
with provenance, demodulation summary, L8 (randomness, periodicity, frame length, blind conv estimate, sync
table), L9 (decision, validation, period estimate, candidates), L10 (family probabilities, decision,
evidence, candidates), L11 (layout, frames), search log, a fixed list of limitations, and the result:
status "VERIFIED (CRC valid)" / "UNVERIFIED" / "NOT DECODED", processing chain, overall confidence
(verified: min(0.99, 0.5 + 0.49 × classification confidence); else 0.4 × classification confidence).
Two sections are new:

* **`recovered_file`** (when an image / file frame was recovered, §15A): format, bytes, width, height,
  channels, CRC valid, CRC received and computed, FEC, hypothesis, sync bit, sync errors and evidence.
* **`signals`** (captures with two or more signals, §5.2): one row per signal with centre, bandwidth,
  SNR, modulation, result (image description, decoded text or error) and CRC status. It is rendered as
  a table in Markdown and HTML, with CRC-valid rows highlighted.

The per-signal report (the Report tab shows the selected signal) contains that signal's layers,
together with the capture's `signals` table.

Renderers: HTML (shown in the Report tab), Markdown, JSON; `save()` chooses by file extension. The report
has no analogue-audio or transcript sections (those features were removed).

---

## 17. GUI — the 14 tabs

### 17.1 Window

**Header.** The title "SIH 26147 · Automated Signal Analysis" and a subtitle describing the chain. The
subtitle has a minimum width of 10 px, so it may clip but never widens the window. For captures with
two or more signals, a **Signal selector** (combo box, one entry per signal with its label and
modulation) appears on the right; choosing a signal refills every tab with that signal's result.

**Theme** (`gui/theme.py`, applied by `apply_theme(app)` before any widget is built):

* A dark slate palette: background `#0b1220`, panels `#111a2e`, text `#e2e8f0`.
* Accents: teal `#2dd4bf` (primary), blue `#60a5fa`, amber `#fbbf24`; good / bad colours for status.
* Eight per-signal series colours.
* One Qt style sheet covers tabs (padding 6 × 9 px), group boxes, buttons (`objectName("primary")` for
  the main action), inputs, tables (teal selection highlight), progress bar, scroll bars and tooltips.
* Matching pyqtgraph defaults (background, foreground, antialiasing).
* In tab titles, `&` is written `&&` so Qt does not treat it as a mnemonic ("1. Input && Setup").

**Scrollable pages.** Every tab page is placed in a `QScrollArea` (`_add_tab`; `self._pages` maps each
tab to its page). Labels that show dynamic text use `wrap_label`, which wraps and has a minimum width,
so it never forces the window wider than the screen. A decoded binary payload is one long unbroken
string: its minimum width became the window's (126 618 px), and Windows reported
`QWindowsWindow::setGeometry: Unable to set geometry`. Long values are also shortened for display:
`short_text` shows the first 400 characters plus a count, and the Parameters tab truncates at 200
characters. The window's minimum width is now about 818 px.

### 17.2 Tabs

| # | Tab | Content (as implemented) |
|---|---|---|
| 1 | Input & Setup | File selection. For `.iq`, the IQ setup dialog (§4.3: sample format + sample-rate source: duration / blind estimate / unknown). Fields: file format, data type (auto + 5 SDR types), endianness, IQ order, byte offset, sample rate, centre frequency, SOI threshold. The sample-rate status line shows the source (KNOWN / ASSUMED / UNKNOWN). The WAV header rate is shown on selection, and a capture summary (the whole capture) after the run. |
| 2 | Signal Detection | PSD plot and spectrogram (waterfall, 512-sample Hann segments) of the **whole capture**, with the SOI table (§5.1). The spectrogram axes are x = frequency, y = time (previously transposed), and its colour levels come from the 20th / 99.8th percentiles, so empty filtered regions do not flatten the contrast. |
| 3 | Parameters | Table of every `Parameter` (value, unit, source, method, confidence) with an evidence inspector. Includes "Recovered image / file" when a frame was found. |
| 4 | Modulation | Detected modulation, confidence bar, diagnostic text, evidence table (including ORDER_CHECK / FAMILY_CHECK / FF_RECEIVER / IMAGE_RETRY / FSK_TONE_FALLBACK / IMAGE_FRAME / CRC / IMAGE entries), constellation of the demodulated symbols. |
| 5 | Sync | For PSK/QAM: three rows built from the receiver's results. "AGC" shows a fixed 1.0 (the block AGC is inside the receiver). "Costas loop" shows the receiver's **total** estimated carrier offset in Hz, although the label says residual. "Gardner TED" shows the number of output symbols; the label says linear interpolation, but the code uses cubic Lagrange. For FSK the tab points to the Parameters tab. |
| 6 | Demodulation | Constellation (1 sample/symbol), EVM (PSK/QAM) or FSK decision error %, decoded payload with CRC status (for an image: its description), first 512 bits. |
| 7 | Conditioning | Manual operations on the demodulated bits: bit offset 0–7, differential decoding, LFSR descrambler presets (PN9, PN11, PN15, DVB-S, CCSDS), bit order (MSB, LSB, swap). Shows bits, hex and ASCII. The result is displayed only, not fed to later tabs. |
| 8 | Bitstream | Sync-search table (Chiani metrics, threshold, occurrences, period, P(emulation), polarity), statistics / periodicity / structure text with hex preview, autocorrelation plot, rank-deficiency plot. "Re-run" analyses the current bits. When L8–L11 was skipped (image found), the forensics run on the receiver's bits in the worker process (§17.3) and "analysing N bits in the background…" is shown until they finish. |
| 9 | De-interleaving | Decision header, best hypothesis with evidence and search log, candidates table (validated row highlighted), Burel–Gautier offset × deficiency plot, rank profile. With an image, it shows "file frame recovered directly … see the Image tab". |
| 10 | FEC | Family probabilities (uncoded, convolutional, block, LDPC, concatenated) with next action, best hypothesis with evidence, decoding candidates, code library list. Same note as tab 9 for an image. |
| 11 | Framing | Layout (sync, length field, CRC, byte order, coverage, layouts tested), frames table, selected frame text and hex, evidence. Same note as tab 9 for an image. |
| 12 | Report | HTML report of the selected signal (with the capture's signals table), export to HTML / Markdown / JSON. |
| 13 | **Image** | The recovered image (`gui/tabs/image_tab.py`), described below. |
| 14 | **Signals** | Every signal in the capture (`gui/tabs/signals_tab.py`), described below. |

**Tab 13 — Image.** The header gives the description with CRC status. The **View** combo offers:

* *Smooth (Lanczos)*, the default: a Pillow Lanczos enlargement to fit, so text on a small transmitted
  image stays readable;
* *Pixels (nearest)*: the received pixels as blocks;
* *Native size*: 1:1.

The image label has an *Ignored* size policy, so a large image never enlarges the window. Other
controls and panels:

* **Save image…** writes the file exactly as sent: an encoded payload byte-exact (without the 5-byte
  header), raw pixels as PNG. The default name is `<capture>_recovered.<ext>`.
* **Save enlarged ×4…** writes a Lanczos ×4 PNG for viewing or reports; it adds no detail.
* The frame-details text shows format, size, bytes, CRC received / computed, FEC, hypothesis and sync
  position / errors, followed by the evidence list.
* Without an image, the tab says so.

**Tab 14 — Signals.**

* A header with the number of signals.
* The capture spectrum (Welch, up to 2²¹ samples, computed in the background) with each detected band
  shaded in its series colour and numbered.
* A table: #, centre, bandwidth, SNR, modulation, result (image description, "CRC VALID: …" /
  "unverified: …" text, "nothing decoded", or the error). CRC-valid or image rows are highlighted.
* Selecting a row (or the header combo) shows that signal in every other tab. The two stay in sync.
* When results are filled in, row 1 is selected with signals blocked, so signal 1 is not rendered a
  second time.

**Landing tab after a run:**

* Signals (tab 14) when the capture holds several signals;
* Image (tab 13) when the single signal carried an image;
* Signal Detection (tab 2) otherwise.

**Progress panel.** It shows the worker's messages (with the `[Signal i/N]` prefix) and has a
**Cancel** button that works during the run (§3.1). The worker stops at the next checkpoint, and the
window reports "Analysis operation was cancelled".

### 17.3 Keeping the window responsive (`gui/widgets/background.py`)

The pipeline itself always ran in a `QThread`. But filling the tabs after a run executed heavy numpy
work on the GUI thread, so Windows marked the window **"Not responding"** for several seconds. The
worst cases were:

* the Bitstream tab's forensics over about 1 M bits: sync-word library search, GF(2) rank profile and
  autocorrelation, about 8 s;
* the Detection tab's PSD + spectrogram of a 2–4 M-sample capture, about 1 s;
* the Signals tab's spectrum.

`BackgroundRunner` (one per tab or plot) moves this work off the GUI thread:

* **`submit(fn, on_done, on_error)`** runs `fn()` in a `QThread`, then calls `on_done(result)` on the
  GUI thread. Used for the Detection plots (`SpectrumPlot.compute` / `WaterfallPlot.compute` are static
  and return data; `apply()` draws) and the Signals spectrum.
* **`submit_process(fn, args, on_done, on_error)`** runs `fn(*args)` in a separate process: one shared
  `ProcessPoolExecutor(max_workers=1)` with the `spawn` context, started on first use. A 100 ms
  `QTimer` checks for completion. Used for the Bitstream forensics (`bitstream.forensics.run`).
  A thread was not enough here: long numpy calls hold Python's GIL, and the GUI still stalled for the
  length of each call (about 1.5 s). A process cannot block the GUI thread.
* **Generation counter.** Only the latest submission of each runner is applied. An older one that
  finishes late is dropped, so a new analysis or a signal switch never shows stale plots.

**Measured** (`shot_result.pkl`, the screenshot QPSK result with 1 013 938 bits; a 50 ms timer records
the longest gap between ticks while the result is shown):

| Version | Longest freeze |
|---|---|
| All tab work on the GUI thread | about 20 s |
| Thread-based background work | 1.5 s |
| Worker process for the forensics | 0.9–2.4 s in two runs, measured while a 70 MB pipeline test was using most of the machine's memory and CPU |

The remaining pause comes from drawing the plots and tables themselves (pyqtgraph and Qt on the GUI
thread), plus the first start of the worker process. The window keeps repainting between these steps.

---

## 18. Synthetic test-signal generation (validation only)

### 18.1 Test frame (`synthetic/framing.py`)

`SYNC (96 bits) | LEN (16 bits, big-endian) | PAYLOAD | CRC-16/CCITT-FALSE over LEN + PAYLOAD | zero pad`.
SYNC = first 96 bits of the PN127 m-sequence (x⁷ + x⁶ + 1, seed 0x7F); 96 is a multiple of 1, 2, 3, 4, 6, 8,
so the sync word occupies whole symbols for every modulation. The receivers use this sync word for
ambiguity resolution; captures without it are handled by the text model.

### 18.2 Generators

* `synthetic/generator.py` (PSK/QAM): Gray-mapped symbols (optionally differential), RRC pulses evaluated in
  continuous time (fractional samples/symbol, timing offset, clock ppm), CFO, phase, gain, DC offset, AWGN at
  a given Es/N0.
* `synthetic/fsk_generator.py`: CPFSK with rectangular or Gaussian pulses (h, BT), Gray/natural mapping,
  optional phase discontinuity and differential coding, CFO, timing offset, clock ppm, DC, IQ gain/phase
  imbalance, AWGN (per-sample SNR).
* `synthetic/channel_coding.py`: frame → FEC (any library convolutional code, RS profile, Hamming code, LDPC
  library code or `CONCAT:<profile>`) → interleaver (`block:RxC`, `conv:B,M`, `diag:RxC[,−1]`, `pr:P|all[,seed]`);
  sync word inside the coded block or outside it (uncoded, in front).
* `synthetic/iq_io.py` writes `.iq` (int16 and float32) + a JSON sidecar with a `truth` block (used only by
  validation tools).

---

## 19. Validation tools, tests and measured results

### 19.1 Image / file recovery (2026-09-28 / 29)

**Test signals.** Built with the signal path of the user's transmitter `Image Decoding/c.py` (Space
Downlink Simulator), re-implemented because the copy in the folder has flattened indentation and does
not run:

* frame `0xAA × 4 | 1ACFFC1D | LEN u32 | PAYLOAD | CRC-32(payload)`;
* payload = W/H/C header + raw RGB, or W/H/0 + PNG;
* c.py bit labels (§15A.1), RRC 0.35, IF offset 0.25·fs, 1 MHz sample rate.

Images: `Demo.png`, `Orginal.png` (scaled by the transmitter) and a generated test pattern.

**Where the files are.** Everything is in `Image Decoding/test_results/`:

* `<name>.wav` / `.iq` (the `.iq` has a JSON sidecar with its sample rate);
* `<name>_encoded.png`, the image sent;
* `<name>_recovered.png`, the image recovered;
* `summary_sent_vs_recovered.png`;
* `README.txt`.

Each signal was run through `UnifiedPipelineWorker`, fully blind. Only the file was given; the `.iq`
case also used its sidecar's sample rate.

| Test | Modulation | Image sent | Payload | File | SNR | Result | Time |
|---|---|---|---|---|---|---|---|
| demo_qpsk_raw | QPSK | 256×191 | raw RGB | stereo I/Q WAV | – | bit-exact, CRC valid | 172.9 s |
| demo_8psk_png | 8PSK | 512×383 | PNG | stereo I/Q WAV | – | bit-exact, CRC valid | 68.6 s |
| orig_16qam_raw | 16QAM | 256×165 | raw RGB | stereo I/Q WAV | 30 dB | bit-exact, CRC valid | 401.7 s |
| orig_qpsk_png1024 | QPSK | 1024×662 | PNG | stereo I/Q WAV | – | bit-exact, CRC valid | 225.3 s |
| pattern_64qam_png | 64QAM | 256×160 | PNG | stereo I/Q WAV | – | bit-exact, CRC valid | 281.5 s |
| pattern_bpsk_raw | BPSK | 96×60 | raw RGB | real passband WAV | 25 dB | bit-exact, CRC valid | 36.2 s |
| demo_2fsk_png | 2FSK | 200×149 | PNG | stereo I/Q WAV | – | bit-exact, CRC valid | 271.2 s |
| orig_4fsk_png | 4FSK | 160×103 | PNG | stereo I/Q WAV | 20 dB | bit-exact, CRC valid | 146.7 s |
| pattern_256qam_raw | 256QAM | 96×60 | raw RGB | stereo I/Q WAV | – | bit-exact, CRC valid | 124.0 s |
| orig_qpsk_raw_noisy | QPSK | 256×165 | raw RGB | `.iq` (cf32 + sidecar) | 14 dB | bit-exact, CRC valid | 151.9 s |

**10 / 10 bit-exact with a valid CRC.** The times are wall-clock on the development laptop. Most of the
time is spent in classification and in the fallbacks tried before the frame verifies.

**What each fix contributed:**

| Fix | Test it fixed |
|---|---|
| Feed-forward receiver with global-index timing + linear drift fit | 64QAM: an edge bit error |
| Outer-ring QAM carrier and ring-radius gain | 16QAM: EVM 20 % → clean |
| Tone correlator with fractional windows | 2FSK: previously fitted as 4FSK, h = 0.17 |
| CRC-tail padding | 2FSK: frame 1–2 bits short |
| Iterative band merging | 14 dB QPSK: previously split into two "signals" |

**User's screenshot capture** (`Image Decoding/Screenshot 2026-09-24 at 4.06.53 PM_qpsk.wav`, about
2 M samples, QPSK, a 256×165 raw RGB image of a slide):

* Before: the loop receivers lost lock about 21 500 symbols in, and the lower part of the image was
  colour noise.
* Now: the feed-forward receiver recovers it **bit-exact with a valid CRC** (`test_results/Screenshot_qpsk_recovered.png`,
  plus a ×4 Lanczos copy).
* The text on the slide is only as sharp as the 256 px the transmitter sent. `orig_qpsk_png1024`
  shows the same slide sent at 1024 px.

### 19.2 Large capture (73 MB WAV)

`Orginal.png` at 768 × 496, raw RGB (payload 1 142 789 bytes), QPSK 250 kBd at 1 MHz, stereo I/Q WAV:
18 284 816 samples, 73.1 MB.

**First run.** Loading, band finding and SOI detection finished in 16 s. Then the order classifier ran
for about 25 minutes and failed: `_ArrayMemoryError: Unable to allocate 8.72 GiB`, a
4 571 228 × 256 distance matrix for the 256QAM hypothesis. The process peak was 9.9 GB on a 16 GB
laptop.

**Fix.** Block-wise distance computations (§6.3).

**Result after the fix (2026-09-29).** The image was recovered **bit-exact, CRC-32 valid**: RAW
768×496×3, 1 142 789 bytes, through the feed-forward receiver at the estimated rate. Total time was
2675 s (44.6 min), with a process peak of 9.2 GB. The stage times were:

| Stage | Time |
|---|---|
| Load, band finding, SOIs | 7 s |
| Classification | 40 min (2415 s) |
| PSK loop receiver | 3.4 min |
| Frame check | 23 s |
| Feed-forward receiver + frame search | 26 s |

The run finished with `exit 0`.

**pytest (2026-09-29, after all the changes in this revision)**: `tests/unit`, 269 passed, 2 failed,
1 expected failure (31 min, run next to the 73 MB test). The two failures,
`test_qam_orders_classified[16QAM]` and `[64QAM]`, are the known ones listed in §19.3; they failed
before the image work too. `test_gui` expects the 14 tabs.

### 19.3 Earlier validation (2026-09-25 … 27, before the image work)

| Tool | What it checks | Last measured result |
|---|---|---|
| `tools/psk_validation.py` | generated PSK vectors, legacy PSK Test-Data, BER sweep | 18/18 generated (float32 copies 18/18); legacy 45/45 blind; BER within 17 % of theory where ≥ 20 errors expected (2026-09-26) |
| `tools/fsk_validate.py` | generated FSK suite, legacy FSK, family discrimination, SNR grid | 24/24 generated (+1 documented-limitation vector, 4GFSK BT 0.25, fails); legacy 3/3 blind; 0 of 98 PSK/QAM files accepted as FSK; 16/16 on the SNR grid (2026-09-26) |
| `tools/qam_validate.py` | generated QAM frames, legacy QAM captures, SER vs theory | generated 12/15; legacy classification 12/12, with expected payload 10/12, fully blind 7/12 (the earlier committed report had 10/12 blind; not yet investigated) |
| `tools/text_numeric_validate.py` | English (40 chars), numeric (33 chars) and digit-only (17 chars) messages × 10 modulations × 3 generator styles (32-bit preamble + ASCII as in `pangrams/p..py`, unframed, framed with CRC), CF32, through the worker | 76/90 (2026-09-27, before the 256QAM work). 256QAM re-run after it: 6/9 (framed 3/3, preamble-style English and numeric, unframed English); unframed numeric / digits and the preamble-style digit string fail |
| `tools/classify_validate.py` | classification of every Test-Data `.iq` and `.wav` | 212/213 correct, family 213/213 (2026-09-25; the 2026-09-26 re-run was stopped before completion) |
| `tools/iq_format_charset_validate.py` | full-charset message (A–Z, a–z, 0–9, all printable specials) × 11 modulations × CU8/CS8/CS16/CF32/CF64 × framed/unframed through the GUI path (dialog detection, duration → rate, worker) | 107/110 on commit 4f3b931 (2026-09-27); format detection 110/110; all 3 failures are 256QAM (two unframed classified 64QAM; one framed with sync found but CRC failing — every wrong character differs from the truth in the same amplitude bit). Previous run: 105/110 |
| `tools/phase4_validate.py` | 8 modulations (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK, GMSK) × 12 coding chains (uncoded, CC K=7, CC K=7 + block 16×24, CC K=5 + conv interleaver 6,4, CC 3/4 sync outside, RS(255,223), Hamming(7,4) + block 8×14, diagonal, pseudo-random LFSR, 802.11 LDPC r3/4, DVB-S structure, CCSDS structure), blind through the worker | 85/96 on commit 4f3b931 (2026-09-27, 8 cases in parallel). Failures: DVB-S structure on all 8 modulations (known, §20); 4FSK cc7 and 4FSK Hamming/block — both pass when re-run alone (≈ 90 s): under 8× parallel load they took ≈ 1100 s and hit the analyser's 600 s wall-clock safety cap, so the tool now passes a 3600 s cap (the hypothesis-count budget remains the search limit; the GUI default stays 600 s); GMSK + 802.11 LDPC fails also alone (first 47 characters correct, then garbage; first run of this combination, not investigated). Excluding DVB-S: 85/88 |

**Text and numeric messages (2026-09-27, before the 256QAM work below)** — `tools/text_numeric_validate.py`: 3 messages (40-char
English pangram, 33-char numeric `9876543210 0123456789 31415926535`, 17-char digit string) × 10 modulations
(BPSK, QPSK, 8PSK, 16PSK, 16/64/256QAM, 2/4/8FSK) × 3 generator styles (32-bit preamble + raw ASCII as in
`pangrams/p..py`; project generator unframed; project generator framed with CRC), CF32, through
`UnifiedPipelineWorker`. **76/90**. Every 256QAM case fails (9/9). The other five failures are four unframed
17-char digit strings (8PSK, 16PSK, 16QAM, 64QAM: 17–46 symbols, order misclassified and no hypothesis
verified) and unframed 16QAM numeric (classified QPSK). Before these changes, BPSK numeric failed on clean
symbols and 16/256QAM English failed the margin test. Clean-symbol decisions (`tests/unit/test_text_models.py`):
42/42 (7 linear modulations × 3 messages × with/without preamble).

**256QAM (2026-09-27)**. Diagnosis on framed 256QAM (Es/N0 39 dB, 1200 Bd, 20–400 characters, 2 seeds each):
the classifier's symbol rate was wrong on 8/8 (22.9, 14.3, 4.9, 32.0 samples/symbol for a true 8.0: the
−3 dB band collapses onto data lines of the repetitive payload); with the true rate forced, 7/8 CRC-valid.
After the bandwidth-consistent rate hypotheses, the all-orders rate stage and the sync-aided refit: 256QAM
cases of the text/numeric matrix 6/9 (was 0/9). The three 256QAM failures of the full-charset matrix were not
re-run.

**New interleavers and codes (2026-09-26):**

* Bit-level joint search (`analyze` on coded bits with LLR noise, no RF): 24/24 chains decoded blind with the
  correct code and interleaver — the 15 earlier chains plus diagonal 16×24 (down-right, K = 7), diagonal
  12×20 (down-left, K = 5), diagonal 8×14 (Hamming(7,4)), frame-sized LFSR pseudo-random interleaver (K = 7),
  802.11 LDPC n = 648 r3/4, project LDPC n = 504, DVB-S structure and CCSDS structure (I = 4).
* End to end through the GUI worker (`tools/phase4_validate.py`, QPSK / 2FSK / 16QAM × 5 new chains): 12/15.
  Diagonal, pseudo-random LFSR, 802.11 LDPC and CCSDS-structure concatenated passed on all three
  modulations. The DVB-S structure failed on all three: with one short frame, the Forney outer interleaver's
  2244 flush bytes make ~94 % of the transmitted signal constant, and the modulation classifier already fails
  (16QAM → QPSK, 2FSK → BPSK) before any FEC stage; real DVB-S randomises the stream (energy dispersal),
  which is not modelled. The DVB-S structure decodes correctly at bit level.
* Blind LDPC (unknown code): 15 frames coded with an unknown (3,6) n = 96 code, error-free bits — all 46
  parity checks recovered, 15/15 frames CRC-valid (about 60 s).
* Blind sample rate: see §4.3.1 (true rate listed for 110/117 Test-Data files).

Blind format detection on the 258 `.iq` files in Test-Data (sidecar and file name ignored): 258/258.

**pytest (2026-09-27, after the text-model / order-check changes)**: `test_text_models`, `test_psk_prototype`,
`test_qam_prototype`, `test_fsk_prototype`, `test_workers`, `test_gui` — 106 passed. `iq_format_charset_validate.py` and `phase4_validate.py` were re-run on the commit (table above); the full pytest suite has not been re-run since these changes.

**pytest**: last full run of `tests/unit` + `tests/integration`: 192 passed, 4 failed, 1 expected failure.
Failures: `test_qam_orders_classified[16QAM]` and `[64QAM]` (fail also without the Phase 4 changes),
`test_phase2_end_to_end_synthetic_pipeline` (old Phase-2 pipeline; intermittent) and one LDPC test caused by
a duplicate column in the project n = 504 code — the construction was then fixed (no duplicate columns) and
the LDPC tests pass (`tests/unit/test_phase4.py` covers diagonal, pseudo-random, LDPC, blind parity checks,
concatenated codes and blind sample-rate hypotheses).

---

## 20. Known limitations and open issues

**New with the 2026-09-28 / 29 work:**

* **Digital signals only.** Analogue demodulation (AM / FM / SSB / DSB-SC) and speech-to-text were
  implemented and then **removed on request**. An analogue capture is still located and channelised
  (§5.2), but it is analysed as a digital signal, and nothing is verified. For example, the 48 kHz
  I/Q recording with three AM stations at −13, −1 and +11 kHz is split into its stations, but their
  audio is not demodulated.
* **Large captures.** A 73 MB WAV (18.3 M samples, 4.6 M QPSK symbols) is processed and its image
  recovered bit-exact (§19.2). It takes about 45 min, 40 min of which is classification, and peaks at
  9.2 GB on a 16 GB laptop. Several full-length complex128 copies of the capture exist at once. A capture
  about twice as large would probably exceed 16 GB of RAM. Processing is not streamed.
* **Classification time.** The order classifier (§6.3) scores all seven PSK/QAM hypotheses over every
  symbol of the capture. This takes about 2–6 min for a 2 M-sample capture on the development laptop,
  and it is the main part of the 36–400 s image-test times of §19.1. It has not been limited to a
  subset of symbols, because the effect on classification accuracy has not been measured.
* **Image frame format.** Only the CCSDS-ASM file frame of §15A is recovered, uncoded or K = 7 r½, and
  unscrambled. Other file transports are not recognised: CCSDS packets / transfer frames with
  randomisers, AX.25 / KISS, SSDV, or JPEG split over many short frames. The L8–L11 search then reports
  the stream as unverified.
* **One frame per signal.** `find_image` returns the first CRC-valid frame. A signal carrying several
  files in sequence yields only the first.
* **Resolution** is the transmitter's. c.py scales the source image to `max_dim` before sending (256 px
  for the screenshot capture). Enlarging in the Image tab interpolates and adds no detail.
* **CRC-unverified images.** An image whose CRC fails but whose structure is consistent is shown and
  marked "CRC not verified". It may contain bit errors.
* **`Image Decoding/c.py`** in the repository has flattened indentation and does not run. The test
  signals were generated with a re-implementation of its signal path (§19.1).
* **GUI pauses.** Filling the tabs still takes up to about 1–2 s of GUI-thread drawing after a large
  result (§17.3), although the heavy computations run in the background.
* **Cancel granularity.** Cancel takes effect at the next checkpoint (§3.1). During the classification
  of a very large capture, that can be minutes away.
* **SOI detector merging** is single-pass (the band finder of §5.2 iterates). A very lumpy spectrum can
  still list one signal as two SOIs in the Detection tab. This does not affect the analysis, which uses
  `find_signals`.
* **Multi-signal limits:** at most 16 bands. Signals that overlap in frequency are not separated. The
  noise floor is the median PSD bin, so a spectrum more than half occupied is not split.

**Earlier limitations (still valid):**

* **256QAM** on short bursts: the classifier's symbol-rate estimate is wrong on every tested framed burst
  of 20–400 characters (1.8–4 × off), so 256QAM relies on the fallback of §3 (bandwidth-consistent standard
  rates, ASSUMED) and the sync-aided refit of §8. Text/numeric matrix: 6/9 (was 0/9). At the true rate the
  receiver still fails on about 1 in 8 framed bursts (26/30 CRC-valid over 3 lengths × 10 seeds before the
  sync-aided refit): gain and phase jump between 8-symbol groups (timing / tracking slips), and the sync
  word is not found. Unframed short numeric messages fail for the same reason. A failing 256QAM capture
  takes several minutes, because the fallback tries every order and rate hypothesis before reporting
  "unresolved".
* Very short unframed messages (≈ 17 characters, a few dozen symbols) on 8PSK and denser constellations are
  often misclassified, and no order/rate hypothesis verifies.
* The order/rate fallback (§3) makes failing short captures slow (all hypotheses are tried before
  "unresolved" is reported). Text it adopts must score ≥ −4.3 nats/char, so symbol-heavy messages are not
  recovered through it.
* Unframed text decoding is a heuristic (general / English / numeric models): upper vs lower case and similar
  near-equivalent readings can remain ambiguous. It is reported as TEXT_HEURISTIC, never as CRC-verified.
* Open classification items: `16psk_4800baud_10db.wav` is classified as 8PSK; `16psk_1200baud_10db.iq`
  does not demodulate; two legacy 64/256QAM WAVs have a wrong first character; `hello_2fsk.wav` decodes as
  8FSK (its `.iq` copy is correct).
* **SigMF**: a `.sigmf-meta` data type is used by the format auto-detection, but the Input tab does not open
  the sample-rate dialog for SigMF files and the sample rate is **not** read from the `.sigmf-meta`; selecting
  the `.sigmf-meta` file itself would be read as raw IQ.
* FEC identification is library-based: convolutional, RS GF(2⁸) with p = 0x11D / fcr = 0, Hamming, LDPC with
  known parity-check matrices (only the 802.11 n = 648 r3/4 standard matrix is included), and RS +
  convolutional concatenations (DVB-S / CCSDS structures, without their randomisers). Unknown LDPC codes are
  recovered blindly only from long, error-free, aligned captures (more codewords than n). Turbo, BCH and polar
  codes, RS with other field polynomials (e.g. CCSDS dual-basis) and scrambled streams are not identified.
* The iterative concatenated decoder's erasure step uses a coarse reliability (no soft-output Viterbi) and did
  not rescue codewords in the tests; the RS → Viterbi feedback helps only with an outer interleaver.
* Pseudo-random interleavers: only the LFSR address-generator family (primitive polynomials, standard seeds,
  or all seeds for the detected period with n ≤ 10) is searched; QPP needs given parameters; an arbitrary
  permutation is detected (rank period) but not reconstructed.
* The blind sample rate is a hypothesis (§4.3.1): only samples/symbol is measured; the proposed absolute
  rate depends on standard-rate lists and a preference order.
* GMSK + 802.11 LDPC (n = 648, r3/4) does not decode end to end (Phase 4 matrix): the first 47 payload
  characters are correct, then the frame fails; not investigated.
* The analyser's wall-clock safety cap (600 s in the GUI) can cut the search short on a heavily loaded
  machine, making results load-dependent in that case; the hypothesis-count budget is otherwise the limit.
* DVB-S-structure signals built from one short frame are dominated by the zero flush of the outer Forney
  interleaver and fail modulation classification end to end (§19).
* Block and diagonal interleavers larger than √N bits are only found through burst-length hypotheses
  (whole number of blocks, data start within the trimmed span ± refinement); convolutional interleavers only
  for B ≤ 16, M ≤ 16 and a burst starting at the interleaver start.
* Framing discovery knows the listed sync words and CRCs; frames without a known sync word are not
  delimited (the text model may still produce readable text, reported as unverified).
* The analyser's `receiver_frame_ok` argument is accepted but not used.
* No real-time / streaming processing. Whole files are loaded into memory, and several full-length
  complex128 copies exist during processing (§19.2).
* The FFT detector reports frequency extent only (no time segmentation). Signals separated in frequency
  are each demodulated (§5.2). Signals separated only in time, within one band, are not: the receivers
  use their own burst detection, which takes the main burst.
* Sync tab labels do not describe the displayed values exactly (see §17).

---

## 21. Code present but not used by the application

Not reachable from `app/main.py` (kept from earlier phases or for tests/tools):

* `gui/workers/phase2_worker.py`, `gui/workers/pipeline_worker.py` (older pipelines),
  `gui/tabs/placeholder_tab.py`, `gui/widgets/eye_diagram.py`.
* `modulation/ml_classifier.py`, `modulation/dsp_features.py`.
* `estimation/bandwidth.py`, `center_frequency.py`, `sample_rate.py`, `snr.py`, `symbol_rate.py`.
* `synthetic/generator.py`, `fsk_generator.py`, `channel_coding.py` (validation only).

Imported by the application but not executed in the unified pipeline:

* `synchronization/agc.py`, `costas_loop.py`, `timing_recovery.py` — only their result dataclasses
  (`AGCResult`, `CostasResult`, `TimingRecoveryResult`) are used as containers for the Sync tab.
* `ingestion/iq_reader.py`, `wav_reader.py`, `sigmf_reader.py` — used by `FormatDetector.load_signal`
  (legacy loader) and by the Input tab to read a WAV header for display; the pipeline itself loads files
  with `ingestion/capture.py`.

Test and tool code for the image path, not used by the application:

* `synthetic/image_framing.py`, `tx_image.py`, `rx_image.py`, `gen_test_image.py`,
  `run_image_pipeline.py`, `verify_roundtrip.py`.

Removed from the repository (2026-09-28): `demodulation/analog.py` (AM / FM / SSB / DSB-SC),
`demodulation/speech.py` (Whisper speech-to-text) and `models/whisper`. Nothing refers to them any more.
