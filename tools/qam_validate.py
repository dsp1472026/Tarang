"""
Phase-3 QAM validation.

    python tools/qam_validate.py                # generated suite + legacy captures + SER sweep
    python tools/qam_validate.py --no-sweep

A. Generated known-payload frames (synthetic.generator.qam_suite): blind receive,
   pass = classification correct, payload byte-exact, CRC-16 valid.
B. Legacy Test-Data QAM captures that carry text (.iq and .wav): payload exact
   with expected-payload ambiguity resolution, and fully blind.
C. Symbol error rate vs theory (Es/N0 sweep, forced order, clean timing).
Writes reports/qam_report.md (+ .json).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demodulation.qam_receiver import QAMReceiverConfig, receive_qam  # noqa: E402
from ingestion.capture import load_capture  # noqa: E402
from modulation.unified_classifier import classify_capture  # noqa: E402
from synthetic.generator import Impairments, SignalSpec, generate, qam_suite  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_TEXT = "BRIGHT VIXENS JUMP; DOZY FOWL QUACK"
LEGACY = [("all_modulations_output/16QAM", "16QAM"), ("all_modulations_output/64QAM", "64QAM"),
          ("all_modulations_output/256QAM", "256QAM"),
          ("qam_pangram_all_files/16qam_bright_vixens", "16QAM"),
          ("qam_pangram_all_files/64qam_bright_vixens", "64QAM"),
          ("qam_pangram_all_files/256qam_bright_vixens", "256QAM")]
SWEEP = {"16QAM": [12, 14, 16, 18], "64QAM": [18, 20, 22, 24], "256QAM": [24, 26, 28, 30]}


def ser_theory(order: int, esn0_db: float) -> float:
    """Square M-QAM symbol error rate in AWGN."""
    m = np.sqrt(order)
    es = 10 ** (esn0_db / 10)
    p = (1 - 1 / m) * erfc(np.sqrt(3 * es / (2 * (order - 1))))
    return float(1 - (1 - p) ** 2)


def run_generated():
    rows = []
    for name, spec in qam_suite():
        x, truth = generate(spec)
        t0 = time.time()
        r = receive_qam(x, truth["sample_rate_hz"])
        crc = bool(r.decode.frame and r.decode.frame.crc_ok)
        ok = crc and r.modulation == truth["modulation"] and r.payload_text == truth["payload_text"]
        rows.append(dict(name=name, truth=truth["modulation"], classified=r.modulation, crc=crc, passed=ok,
                         evm_pct=round(100 * r.evm, 2), method=r.decode.method, seconds=round(time.time() - t0, 1),
                         sps=round(float(r.parameters["sps"].value), 4), sps_true=truth["samples_per_symbol"]))
        print(f"  A {name:26s} {'PASS' if ok else 'FAIL'} {r.modulation:7s} EVM {100 * r.evm:5.1f}%  "
              f"{r.decode.method}  {rows[-1]['seconds']}s", flush=True)
    return rows


def run_legacy():
    rows = []
    for stem, mod in LEGACY:
        for ext in (".iq", ".wav"):
            path = os.path.join(ROOT, "Test-Data", stem + ext)
            if not os.path.exists(path):
                continue
            cap = load_capture(path)
            t0 = time.time()
            cls = classify_capture(cap.samples, cap.sample_rate)
            out = {}
            for tag, exp in (("known", LEGACY_TEXT), ("blind", None)):
                r = receive_qam(cap.samples, cap.sample_rate, QAMReceiverConfig(modulation=cls.modulation,
                                                                               expected_payload=exp),
                                front=cls.front, classification=cls.linear)
                txt = r.payload_text
                out[tag] = dict(text=txt, ok=(txt == LEGACY_TEXT) if tag == "known" else (LEGACY_TEXT in txt),
                                char_errors=sum(a != b for a, b in zip(txt, LEGACY_TEXT)) + abs(len(txt) - len(LEGACY_TEXT)),
                                method=r.decode.method, evm_pct=round(100 * r.evm, 2),
                                mapping=next((e.description.split("[")[1].split("]")[0] for e in r.decode.evidence
                                              if "mapping [" in e.description), "Gray (standard)"))
            rows.append(dict(file=stem + ext, truth=mod, classified=cls.modulation, cls_ok=cls.modulation == mod,
                             known=out["known"], blind=out["blind"], seconds=round(time.time() - t0, 1)))
            print(f"  B {stem + ext:48s} {cls.modulation:7s} known:{'PASS' if out['known']['ok'] else 'FAIL'} "
                  f"blind:{'PASS' if out['blind']['ok'] else 'FAIL'} ({out['blind']['text'][:24]!r})", flush=True)
    return rows


def run_sweep(n_bytes: int = 400):
    rows = []
    rng = np.random.default_rng(7)
    for mod, grid in SWEEP.items():
        order = int(mod[:-3])
        text = "".join(chr(c) for c in rng.integers(32, 127, n_bytes))
        for es in grid:
            x, truth = generate(SignalSpec(mod, payload_text=text, impairments=Impairments(esn0_db=float(es),
                                                                                           seed=int(es))))
            r = receive_qam(x, truth["sample_rate_hz"], QAMReceiverConfig(modulation=mod, sps=8.0, rrc_beta=0.35))
            d = r.decode
            if not d.frame or d.payload_bit_start is None:
                rows.append(dict(mod=mod, esn0=es, ser=None, theory=ser_theory(order, es), note=d.method))
                continue
            bps = int(np.log2(order))
            tx = np.frombuffer(truth["frame_bits"].encode(), np.uint8) - 48
            start = d.payload_bit_start - 16 - 96
            rx = d.hard_bits[start:start + len(tx)]
            n = min(len(tx), len(rx)) // bps * bps
            sym_err = np.any((tx[:n] != rx[:n]).reshape(-1, bps), axis=1)
            rows.append(dict(mod=mod, esn0=es, ser=float(np.mean(sym_err)), theory=ser_theory(order, es),
                             symbols=int(n // bps)))
            print(f"  C {mod} Es/N0 {es} dB: SER {rows[-1]['ser']:.2e} (theory {rows[-1]['theory']:.2e})", flush=True)
    return rows


def write_report(gen, leg, sweep, path):
    L = ["# Phase 3 - QAM (16 / 64 / 256QAM) validation", ""]
    if gen:
        L.append(f"* **Generated known-payload frames:** {sum(r['passed'] for r in gen)}/{len(gen)} PASS "
                 "(blind classification correct, payload byte-exact, CRC-16 valid).")
    if leg:
        L.append(f"* **Legacy Test-Data QAM captures (.iq + .wav):** classification "
                 f"{sum(r['cls_ok'] for r in leg)}/{len(leg)}; payload exact with expected payload "
                 f"{sum(r['known']['ok'] for r in leg)}/{len(leg)}; fully blind {sum(r['blind']['ok'] for r in leg)}/{len(leg)}.")
    L += ["", "## Receiver", "",
          "| Stage | Method |", "|---|---|",
          "| Front end | shared with PSK: AGC, burst, coarse CFO, RRC matched filter, Oerder-Meyr + Gardner timing |",
          "| Rate / CFO | if the strobes fit no PSK/QAM lattice: coarse-CFO x symbol-rate hypotheses (integer sub-rates, "
          "coarse-to-fine amplitude-ring scan, CFO micro-scan), best lattice fit wins |",
          "| Order | ALRT over BPSK..256QAM after sigma_A^2 (radial vs tangential spread) splits PSK/QAM; EVM per order "
          "(12.5/8/3.5 % limits), C40 and amplitude kurtosis reported as evidence. No PAPR. |",
          "| Timing | roll-off x timing (x rate for < 200 symbols) scan scored by the lattice residual |",
          "| Carrier | 4th-power line + joint phase / residual-frequency / scale fit (decision-directed complex LS "
          "from seeds), then decision-directed carrier + gain loops on confident symbols |",
          "| Edges | symbols next to a capture edge that cuts the pulses: truncated-pulse least squares |",
          "| Ambiguity | sync word (4 rotations x 4 Rs/4 aliases); captures without one: label-mapping hypotheses "
          "(Gray/natural x ascending/descending x I/Q order) x rotations x aliases, by expected payload or text |",
          "| Bits | Gray per axis, max-log LLRs (ln P0/P1) |", ""]
    if gen:
        L += ["## A. Generated frames", "", "| Vector | Truth | Classified | EVM % | Ambiguity | CRC | Result | s |",
              "|---|---|---|---|---|---|---|---|"]
        for r in gen:
            L.append(f"| {r['name']} | {r['truth']} | {r['classified']} | {r['evm_pct']} | {r['method']} | "
                     f"{'valid' if r['crc'] else 'fail'} | {'PASS' if r['passed'] else 'FAIL'} | {r['seconds']} |")
        L.append("")
    if leg:
        L += ["## B. Legacy captures (no sync word; payload 'BRIGHT VIXENS JUMP; DOZY FOWL QUACK')", "",
              "| File | Truth | Classified | Mapping found | Known payload | Blind | Blind text |", "|---|---|---|---|---|---|---|"]
        for r in leg:
            L.append(f"| `{r['file']}` | {r['truth']} | {r['classified']} | {r['blind']['mapping']} | "
                     f"{'PASS' if r['known']['ok'] else 'FAIL (' + str(r['known']['char_errors']) + ' chars)'} | "
                     f"{'PASS' if r['blind']['ok'] else 'FAIL'} | `{r['blind']['text'][:40]!r}` |")
        L.append("")
    if sweep:
        L += ["## C. Symbol error rate vs theory (known timing, forced order)", "", "| Mod | Es/N0 dB | SER | Theory |",
              "|---|---|---|---|"]
        for r in sweep:
            ser = "-" if r["ser"] is None else f"{r['ser']:.2e}"
            L.append(f"| {r['mod']} | {r['esn0']} | {ser} | {r['theory']:.2e} |")
        L.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    with open(os.path.splitext(path)[0] + ".json", "w") as fh:
        json.dump(dict(generated=gen, legacy=leg, sweep=sweep), fh, indent=1, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-generated", action="store_true")
    ap.add_argument("--no-legacy", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "qam_report.md"))
    args = ap.parse_args()
    gen = [] if args.no_generated else (print("A. generated frames") or run_generated())
    leg = [] if args.no_legacy else (print("B. legacy captures") or run_legacy())
    sweep = [] if args.no_sweep else (print("C. SER sweep") or run_sweep())
    write_report(gen, leg, sweep, args.out)
    print(f"\nreport: {args.out}")
    if gen:
        print(f"generated {sum(r['passed'] for r in gen)}/{len(gen)}")
    if leg:
        print(f"legacy known {sum(r['known']['ok'] for r in leg)}/{len(leg)}, blind {sum(r['blind']['ok'] for r in leg)}/{len(leg)}")


if __name__ == "__main__":
    main()
