"""
Phase 1 (PSK) validation runner and report generator.

    python tools/psk_validation.py                 # everything, writes reports/phase1_psk_report.md
    python tools/psk_validation.py --regen         # regenerate Test-Data/synthetic_psk first
    python tools/psk_validation.py --no-sweep      # skip the BER-vs-Es/N0 sweep
    python tools/psk_validation.py --no-baseline   # skip the legacy-pipeline comparison

Sections:
  A. Generated known-payload vectors (primary ground truth), demodulated blind.
     PASS = classification correct AND decoded payload byte-exact AND CRC valid.
  B. Legacy Test-Data PSK captures (no sync word): classification, EVM, BER vs
     the ground-truth bits; ambiguity resolved (1) with the expected payload and
     (2) fully blind (text-likelihood heuristic).
  C. Baseline: the existing GUI pipeline (gui/workers/phase2_worker.py) on the same files.
  D. BER vs Es/N0 against theory (long random-text payloads, full blind receiver).

The demodulator only ever sees samples + sample rate. Truth is read here, after demodulation.
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.special import erfc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from demodulation.psk_receiver import PSKReceiverConfig, receive_psk  # noqa: E402
from synthetic.framing import SYNC_BITS, bytes_to_bits  # noqa: E402
from synthetic.generator import Impairments, SignalSpec, generate, psk_suite, save  # noqa: E402
from synthetic.iq_io import read_iq, read_truth  # noqa: E402

GEN_DIR = os.path.join(ROOT, "Test-Data", "synthetic_psk")
LEGACY_GLOBS = ["hello_bpsk.iq", "hello_qpsk.iq", "hello_8psk.iq",
                "bpsk_*baud_*db.iq", "qpsk_*baud_*db.iq", "8psk_*baud_*db.iq"]


def q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def theory_ber(mod: str, esn0_db: float) -> float:
    """Gray-coded coherent BER (8PSK: nearest-neighbour approximation)."""
    es = 10 ** (esn0_db / 10)
    if mod == "BPSK":
        return q(np.sqrt(2 * es))
    if mod == "QPSK":
        return q(np.sqrt(es))
    if mod == "8PSK":
        return (2 / 3) * q(np.sqrt(2 * es) * np.sin(np.pi / 8))
    raise ValueError(mod)


def ascii_safe(s: str, n: int = 40) -> str:
    s = s.encode("ascii", "backslashreplace").decode()
    s = s.replace("|", "\\|")
    return s if len(s) <= n else s[:n] + "..."


# ---------------------------------------------------------------------------
# A. Generated vectors
# ---------------------------------------------------------------------------

def frame_ber(r, truth) -> tuple:
    """Bit errors over the whole frame (sync..pad), aligned at the detected sync word."""
    ref = np.array(list(truth["frame_bits"]), dtype=np.uint8)
    d = r.decode
    if d.sync_symbol_index is None:
        return None, len(ref)
    start = d.payload_bit_start - 16 - len(SYNC_BITS)      # first sync bit
    got = d.hard_bits[start:start + len(ref)]
    if len(got) < len(ref):
        return len(ref), len(ref)
    return int(np.count_nonzero(got != ref)), len(ref)


def llr_agreement(r, truth) -> float:
    ref = np.array(list(truth["frame_bits"]), dtype=np.uint8)
    d = r.decode
    if d.payload_bit_start is None:
        return float("nan")
    start = d.payload_bit_start - 16 - len(SYNC_BITS)
    llr = d.llrs[start:start + len(ref)]
    if len(llr) < len(ref):
        return float("nan")
    return float(np.mean((llr < 0).astype(np.uint8) == ref))


def run_generated(regen: bool):
    if regen or not glob.glob(os.path.join(GEN_DIR, "*.iq")):
        for name, spec in psk_suite():
            save(spec, os.path.join(GEN_DIR, name))
    rows = []
    for name, _ in psk_suite():
        path = os.path.join(GEN_DIR, name + ".iq")
        truth = read_truth(path)
        f = read_iq(path)
        t0 = time.time()
        r = receive_psk(f.samples, f.sample_rate)
        dt = time.time() - t0
        # Same capture as float32: must give the identical payload.
        f32 = read_iq(os.path.join(GEN_DIR, name + "_f32.iq"))
        r32 = receive_psk(f32.samples, f32.sample_rate)
        errs, nbits = frame_ber(r, truth)
        payload_ok = r.decode.payload == truth["payload_text"].encode()
        crc_ok = bool(r.decode.frame and r.decode.frame.crc_ok)
        cls_ok = r.modulation == truth["modulation"]
        imp = truth["impairments"]
        rows.append(dict(
            file=name, intended=truth["modulation"], classified=r.modulation,
            confidence=r.classification.confidence, evm=100 * r.evm, esn0_true=imp["esn0_db"],
            esn0_est=r.esn0_est_db, sps_true=truth["samples_per_symbol"], sps_est=r.parameters["sps"].value,
            cfo_true=imp["cfo_hz"], cfo_est=r.parameters["cfo"].value, beta_true=truth["rrc_beta"],
            beta_est=r.parameters["rrc_beta"].value, method=r.decode.method,
            differential=truth["differential"], frame_bit_errors=errs, frame_bits=nbits,
            llr_sign_agreement=llr_agreement(r, truth), crc=crc_ok, payload_ok=payload_ok,
            f32_payload_ok=r32.decode.payload == truth["payload_text"].encode(),
            decoded=r.payload_text, seconds=dt,
            passed=bool(cls_ok and payload_ok and crc_ok)))
        print(f"  A {name:24s} {truth['modulation']:5s}->{r.modulation:5s} EVM {100 * r.evm:5.1f}%  "
              f"{r.decode.method:12s} CRC {crc_ok}  {'PASS' if rows[-1]['passed'] else 'FAIL'}")
    return rows


# ---------------------------------------------------------------------------
# B. Legacy Test-Data
# ---------------------------------------------------------------------------

def legacy_files():
    out = []
    for g in LEGACY_GLOBS:
        out += sorted(p for p in glob.glob(os.path.join(ROOT, "Test-Data", g)) if not p.endswith("_f32.iq"))
    return out


def run_legacy():
    rows = []
    for path in legacy_files():
        truth = read_truth(path)
        f = read_iq(path)
        ref_bits = np.array(truth["bits"], dtype=np.uint8)
        msg = truth["payload_text"]
        res = {}
        for mode, payload in (("known", msg), ("blind", None)):
            r = receive_psk(f.samples, f.sample_rate, PSKReceiverConfig(expected_payload=payload))
            d = r.decode
            start = d.payload_bit_start or 0
            got = d.hard_bits[start:start + len(ref_bits)]
            n = min(len(got), len(ref_bits))
            ber_errs = int(np.count_nonzero(got[:n] != ref_bits[:n])) + (len(ref_bits) - n)
            res[mode] = dict(r=r, errs=ber_errs, ok=d.payload[:len(msg)] == msg.encode() and
                             r.modulation == truth["modulation"])
        k = res["known"]["r"]
        rows.append(dict(
            file=os.path.basename(path), intended=truth["modulation"], classified=k.modulation,
            confidence=k.classification.confidence, evm=100 * k.evm, snr_file_db=truth["config"]["snr_db"],
            esn0_est=k.esn0_est_db, bits=len(ref_bits),
            errs_known=res["known"]["errs"], errs_blind=res["blind"]["errs"],
            decoded_known=k.payload_text, decoded_blind=res["blind"]["r"].payload_text,
            method_blind=res["blind"]["r"].decode.method,
            pass_known=bool(res["known"]["ok"]), pass_blind=bool(res["blind"]["ok"])))
        print(f"  B {os.path.basename(path):24s} {truth['modulation']:5s}->{k.modulation:5s} EVM {100 * k.evm:5.1f}%  "
              f"known:{'PASS' if rows[-1]['pass_known'] else 'FAIL'} blind:{'PASS' if rows[-1]['pass_blind'] else 'FAIL'}")
    return rows


# ---------------------------------------------------------------------------
# C. Baseline (existing GUI pipeline)
# ---------------------------------------------------------------------------

def run_baseline(paths_fs):
    try:
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.CRITICAL)
        from gui.workers.phase2_worker import Phase2PipelineWorker
    except Exception as e:  # PyQt or model files unavailable
        print(f"  baseline skipped: {e}")
        return None
    rows = {}
    for path, fs, expected in paths_fs:
        try:
            w = Phase2PipelineWorker({"file_path": path, "dtype": "int16", "sample_rate": fs})
            res = w.work()
            d = res["demod_result"]
            b = d.bits
            text = bytes(int("".join(map(str, b[i:i + 8])), 2) for i in range(0, len(b) - 7, 8))
            rows[os.path.basename(path)] = dict(classified=str(res["parameters"]["modulation"].value),
                                                evm=float(d.evm_pct),
                                                payload_found=expected.encode() in text,
                                                text=text[:24].decode("latin-1"))
        except Exception as e:
            rows[os.path.basename(path)] = dict(classified=f"ERROR {type(e).__name__}", evm=float("nan"),
                                                payload_found=False, text="")
    return rows


# ---------------------------------------------------------------------------
# D. BER sweep
# ---------------------------------------------------------------------------

SWEEP = {"BPSK": [2, 4, 6, 8], "QPSK": [5, 7, 9, 11], "8PSK": [10, 12, 14, 16]}


def run_sweep(n_bytes: int = 1500):
    rows = []
    rng = np.random.default_rng(7)
    for mod, points in SWEEP.items():
        for es in points:
            text = "".join(chr(c) for c in rng.integers(32, 127, n_bytes))
            spec = SignalSpec(mod, payload_text=text, impairments=Impairments(
                esn0_db=es, cfo_hz=23.0, phase_deg=float(rng.uniform(0, 360)), timing_offset=float(rng.uniform()),
                clock_ppm=50.0, seed=int(rng.integers(1 << 30))))
            x, truth = generate(spec)
            r = receive_psk(x, spec.sample_rate)
            ref = np.array(list(truth["frame_bits"]), dtype=np.uint8)
            d = r.decode
            if d.payload_bit_start is None:
                errs, n = None, len(ref)
            else:
                s0 = d.payload_bit_start - 16 - len(SYNC_BITS)
                got = d.hard_bits[s0:s0 + len(ref)]
                n = len(ref)
                errs = int(np.count_nonzero(got != ref[:len(got)])) + (n - len(got))
            ber = None if errs is None else errs / n
            rows.append(dict(mod=mod, esn0=es, classified=r.modulation, bits=n, errors=errs, ber=ber,
                             theory=theory_ber(mod, es), evm=100 * r.evm, method=d.method))
            print(f"  D {mod:5s} Es/N0 {es:4.1f} dB  cls {r.modulation:5s}  BER {ber if ber is not None else 'n/a'}"
                  f"  theory {theory_ber(mod, es):.2e}")
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(gen, legacy, baseline_gen, baseline_leg, sweep, path):
    L = []
    npass = sum(r["passed"] for r in gen)
    L.append("# Phase 1 — PSK Family Validation Report\n")
    L.append(f"Generated by `tools/psk_validation.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
             "All demodulation is blind: the receiver sees only the samples and the sample rate.\n")
    L.append("## Summary\n")
    L.append(f"* **Generated known-payload vectors:** {npass}/{len(gen)} PASS "
             "(classification correct, payload byte-exact, CRC-16 valid). "
             f"float32 copies decode identically: {sum(r['f32_payload_ok'] for r in gen)}/{len(gen)}.")
    if legacy:
        cls_ok = sum(r["intended"] == r["classified"] for r in legacy)
        L.append(f"* **Legacy Test-Data PSK captures:** classification {cls_ok}/{len(legacy)}; payload exact with "
                 f"expected-payload ambiguity resolution {sum(r['pass_known'] for r in legacy)}/{len(legacy)}; "
                 f"fully blind {sum(r['pass_blind'] for r in legacy)}/{len(legacy)}.")
    if sweep:
        ok = [r for r in sweep if r["ber"] is not None]
        solid = [r for r in ok if r["theory"] * r["bits"] >= 20]     # >= 20 expected errors
        worst = max(abs(r["ber"] / r["theory"] - 1) for r in solid) if solid else float("nan")
        L.append(f"* **BER vs Es/N0:** {len(ok)}/{len(sweep)} sweep points synchronised (blind, down to 2 dB Es/N0 "
                 f"for BPSK). Where >= 20 errors are expected, measured BER is within {100 * worst:.0f}% of "
                 "coherent Gray-coded theory.")
    L.append("")
    L.append(STATIC_FORMAT)

    L.append("## A. Generated known-payload vectors (`Test-Data/synthetic_psk/`)\n")
    L.append("Impairments per scenario: `clean` Es/N0 30 dB; `impaired` CFO 37 Hz, phase 73°, timing 0.37 T, "
             "clock +150 ppm, DC offset, gain 0.3; `fracsps_bigcfo` fs = 10 kHz (8.333 sps), CFO −95 Hz, clock "
             "−120 ppm; `beta025_4800bd` β = 0.25 at 4800 Bd, CFO 150 Hz; `differential` differentially encoded; "
             "`short_burst` 11-byte payload. Es/N0 is at the matched-filter output.\n")
    L.append("| File | Intended | Classified (conf.) | Es/N0 true / est. | EVM | sps true / est. | CFO true / est. (Hz) "
             "| Ambiguity | Frame BER | LLR sign agree | CRC | Decoded (first 40 chars) | Result |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in gen:
        ber = "n/a" if r["frame_bit_errors"] is None else f"{r['frame_bit_errors']}/{r['frame_bits']}"
        L.append(f"| {r['file']} | {r['intended']} | {r['classified']} ({r['confidence']:.2f}) | "
                 f"{r['esn0_true']:.0f} / {r['esn0_est']:.1f} dB | {r['evm']:.1f}% | {r['sps_true']:.3f} / "
                 f"{r['sps_est']:.3f} | {r['cfo_true']:.0f} / {r['cfo_est']:.1f} | {r['method']} | {ber} | "
                 f"{r['llr_sign_agreement']:.4f} | {'VALID' if r['crc'] else 'FAIL'} | `{ascii_safe(r['decoded'])}` | "
                 f"**{'PASS' if r['passed'] else 'FAIL'}** |")
    L.append("")
    if baseline_gen:
        L.append("### Baseline: existing GUI pipeline on the same generated vectors\n")
        L.append("| File | Existing classifier | Existing EVM | Payload found |")
        L.append("|---|---|---|---|")
        for r in gen:
            b = baseline_gen.get(r["file"] + ".iq", {})
            L.append(f"| {r['file']} | {b.get('classified', '-')} | {b.get('evm', float('nan')):.1f}% | "
                     f"{'yes' if b.get('payload_found') else 'no'} |")
        L.append("")

    if legacy:
        L.append("## B. Legacy Test-Data PSK captures\n")
        L.append("These files carry `\"HELLO WORLD\"` (88 bits; 8PSK padded to 90) with **no sync word**, 30–88 "
                 "symbols, no impairments except AWGN. `SNR (file)` is the legacy per-sample SNR "
                 "(≈ Es/N0 − 9 dB). Without a sync word the M rotations × M carrier aliases can only be resolved "
                 "from content: *known* = expected payload supplied, *blind* = generic English-text likelihood.\n")
        L.append("| File | Intended | Classified (conf.) | SNR (file) | EVM | BER known | BER blind | Decoded (known) "
                 "| Known | Blind |" + (" Existing pipeline (class / EVM / payload) |" if baseline_leg else ""))
        L.append("|---|---|---|---|---|---|---|---|---|---|" + ("---|" if baseline_leg else ""))
        for r in legacy:
            extra = ""
            if baseline_leg:
                b = baseline_leg.get(r["file"], {})
                extra = (f" {b.get('classified', '-')} / {b.get('evm', float('nan')):.1f}% / "
                         f"{'yes' if b.get('payload_found') else 'no'} |")
            L.append(f"| {r['file']} | {r['intended']} | {r['classified']} ({r['confidence']:.2f}) | "
                     f"{r['snr_file_db']:.0f} dB | {r['evm']:.1f}% | {r['errs_known']}/{r['bits']} | "
                     f"{r['errs_blind']}/{r['bits']} | `{ascii_safe(r['decoded_known'], 16)}` | "
                     f"{'PASS' if r['pass_known'] else 'FAIL'} | {'PASS' if r['pass_blind'] else 'FAIL'} |" + extra)
        L.append("")

    if sweep:
        L.append("## D. BER vs Es/N0 (full blind receiver vs. theory)\n")
        L.append("1500-byte random printable-text payloads, CFO 23 Hz, random phase and timing, clock +50 ppm. "
                 "BER counted over the whole frame from the detected sync word. Theory: coherent Gray-coded "
                 "(8PSK nearest-neighbour approximation).\n")
        L.append("Rows with fewer than ~20 expected errors are statistically weak (Poisson spread of a few counts).\n")
        L.append("| Modulation | Es/N0 | Classified | Bits | Errors | Expected errors | Measured BER | Theory BER | Ratio |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in sweep:
            expected = r["theory"] * r["bits"]
            if r["ber"] is None:
                L.append(f"| {r['mod']} | {r['esn0']} dB | {r['classified']} | {r['bits']} | - | {expected:.1f} | "
                         f"no sync | {r['theory']:.2e} | - |")
            else:
                ratio = r["ber"] / r["theory"] if r["theory"] > 0 else float("nan")
                L.append(f"| {r['mod']} | {r['esn0']} dB | {r['classified']} | {r['bits']} | {r['errors']} | "
                         f"{expected:.1f} | {r['ber']:.2e} | {r['theory']:.2e} | {ratio:.2f} |")
        L.append("")

    L.append(STATIC_NOTES)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


STATIC_FORMAT = """## IQ format of the existing Test-Data (documented from `sg.py`, the ground-truth JSONs and the samples)

| Property | Value |
|---|---|
| Files | `<name>.iq` = interleaved I/Q **int16** little-endian, peak-scaled to 0.95 FS; `<name>_f32.iq` = interleaved I/Q **float32** LE, unscaled (identical content, correlation 1.0000); `<name>.wav` = real passband, 48 kHz, fc = 6 kHz |
| Header | none |
| Sample rate | `baseband_fs_hz` in `*_ground_truth.json` = 8 x baud (9.6 / 19.2 / 38.4 kHz); `sample_rate` in `hello_*.json` |
| Centre frequency | 0 (complex baseband) |
| Pulse shaping | RRC, beta = 0.35, span 10 symbols, unit-energy taps, `np.convolve(..., 'same')` so symbol 0 is at sample 0 |
| Mapping | Gray: angle = phi0 + 2*pi*k/M, label = k ^ (k>>1); phi0 = pi (BPSK: bit 0 -> -1), pi/4 (QPSK), 0 (8PSK); bits MSB-first |
| Payload | `"HELLO WORLD"` ASCII, 88 bits (8PSK zero-padded to 90), **no preamble / sync word** |
| Impairments | AWGN only, at a *per-sample* SNR (Es/N0 = SNR + 9 dB); no CFO, phase, timing or clock offset |

New vectors (`Test-Data/synthetic_psk/`) use the same `.iq` / `_f32.iq` layout plus a `<name>.json` sidecar whose
`format` block (datatype, sample rate) is all the demodulator reads; the `truth` block is only read by the validator.
Frame: 96-bit PN sync word | 16-bit length | payload | CRC-16/CCITT | pad; Es/N0 is specified at the matched-filter output.
"""

STATIC_NOTES = """## Root causes found and fixed along the way

| Symptom | Root cause | Fix |
|---|---|---|
| Legacy QPSK decodes as garbage (`{ussp...`) | Existing `demodulator.py` QPSK bit map is natural-binary over a different point order, not the generator's Gray map | New Gray-labelled constellation module shared by generator and demodulator (`dsp/constellations.py`) |
| Symbol rate locked to 10.67 or 16 sps instead of 8 | Repeated ASCII creates data-induced cyclic lines stronger than the true rate line | -3 dB bandwidth (beta-independent: RC PSD is -3 dB at Rs/2) + cyclic lines as *hypotheses*, scored by strobe compactness |
| Clean BPSK EVM 23 %, legacy BPSK 21 % at 30 dB | "DC removal" subtracted the genuine mean of ASCII-biased symbols | DC taken only from noise-only samples; otherwise estimated decision-directed in the symbol domain and kept only if EVM improves |
| Differential 8PSK: CFO off by exactly Rs/8; legacy QPSK/8PSK: by Rs/4, Rs/8 | M-th power frequency estimate is ambiguous modulo Rs/M; unscrambled ASCII makes the spectrum asymmetric, biasing the coarse estimate by one alias | Every Rs/M alias is a hypothesis resolved by the sync word (or known payload / text likelihood for legacy captures) |
| DBPSK false sync, CRC fail | A differentially encoded m-sequence is nearly a shifted m-sequence (shift-and-add), so coherent sync "matched" | Coherent and differential hypotheses ranked together: CRC-valid first, then sync-bit errors |
| Roll-off estimated as 1.0 on coloured data, residual ISI | Raised-cosine PSD fit assumes white symbols | Final beta chosen decision-directed (min EVM) jointly with a timing-phase scan; PSD fit kept as evidence |
| 30-symbol legacy captures: rate off by ~0.75 % | Cyclic-line resolution on 240 samples | Decision-directed rate scan for bursts < 200 symbols |
| BPSK below 4 dB Es/N0: no sync | Double-rate hypothesis slipped inside the tolerance; line missed between search-grid points | Keep the spectral estimate unless clearly beaten; >= 4x zero-padded FFT line search; channel filter before cyclic analysis |

## Limitations (honest scope)

* Hypothesis set is BPSK / QPSK(=4QAM) / 8PSK. `hello_16psk.iq` exists but 16PSK is not in scope yet (it is labelled 8PSK).
* One burst per capture; no multi-signal channelisation (L2 finds a single active region).
* Legacy captures have no sync word: carrier-phase ambiguity is resolved either with the expected payload or with a
  generic English-text likelihood. That heuristic works on all 45 files but is content-dependent and flagged as weak evidence.
* Classification confidence = ALRT posterior (effective sample size capped at 100) x fraction of agreeing checks;
  it is a transparent composite, not a calibrated probability.
* Differential detection costs ~3 dB versus coherent (expected); EVM for differential vectors is measured after
  coherent carrier recovery.

## Reproduce

```bash
python -m synthetic.generator --out Test-Data/synthetic_psk      # regenerate the 18 PSK vectors
python -m pytest tests/unit/test_psk_prototype.py -q               # 25 unit/integration tests
python tools/psk_validation.py                                     # this report (+ .json); exit code 0 = all pass
python tools/demod_psk.py --file Test-Data/synthetic_psk/8psk_impaired.iq
python tools/demod_psk.py --file Test-Data/hello_qpsk.iq --payload "HELLO WORLD"
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-legacy", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "phase1_psk_report.md"))
    args = ap.parse_args()

    print("A. generated vectors")
    gen = run_generated(args.regen)
    legacy = [] if args.no_legacy else (print("B. legacy Test-Data") or run_legacy())
    base_gen = base_leg = None
    if not args.no_baseline:
        print("C. baseline (existing pipeline)")
        base_gen = run_baseline([(os.path.join(GEN_DIR, r["file"] + ".iq"),
                                  read_iq(os.path.join(GEN_DIR, r["file"] + ".iq")).sample_rate,
                                  read_truth(os.path.join(GEN_DIR, r["file"] + ".iq"))["payload_text"]) for r in gen])
        if legacy:
            base_leg = run_baseline([(p, read_iq(p).sample_rate, "HELLO WORLD") for p in legacy_files()])
    sweep = [] if args.no_sweep else (print("D. BER sweep") or run_sweep())

    write_report(gen, legacy, base_gen, base_leg, sweep, args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({"generated": gen, "legacy": legacy, "baseline_generated": base_gen,
                   "baseline_legacy": base_leg, "sweep": sweep}, fh, indent=1, default=str)
    print(f"\nreport: {args.out}")
    all_ok = all(r["passed"] for r in gen) and all(r["pass_known"] for r in legacy)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
