"""
Phase 2 (FSK) validation runner and report generator.

    python tools/fsk_validate.py                  # everything -> reports/fsk_report.md (+ .json, reports/fsk/*.txt)
    python tools/fsk_validate.py --regen          # regenerate Test-Data/fsk first
    python tools/fsk_validate.py --quick          # skip the BER sweep and the rejection test

Sections:
  A. Generated known-payload vectors (Test-Data/fsk), demodulated blind.
     PASS = variant correct AND payload byte-exact AND CRC valid.
     Vectors named limit_* are documented limitations, reported but not pass criteria.
  B. Legacy Test-Data FSK: all_modulations_output/{BFSK,4FSK,8FSK} (payload known, no
     sync word) with and without the expected payload; sg.py hello_2fsk/hello_4fsk
     (random symbols, no payload) scored by symbol error rate against the reproduced
     RNG sequence.
  C. Family discrimination: every PSK/QAM file must not be accepted as FSK.
  D. BER vs Es/N0 per demodulator vs theory, and the spec's per-sample SNR grid.
"""

import argparse
import glob
import json
import os
import sys
import time
from math import comb

import numpy as np
from scipy.special import erfc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from demodulation.fsk_receiver import FSKReceiverConfig, receive_fsk  # noqa: E402
from dsp.conditioning import detect_burst, estimate_dc, normalize_power  # noqa: E402
from estimation.fsk_estimator import occupied_band  # noqa: E402
from modulation.fsk_classifier import classify_family  # noqa: E402
from synthetic.framing import SYNC_BITS  # noqa: E402
from synthetic.fsk_generator import FSKImpairments, FSKSpec, fsk_suite, generate_fsk, save_fsk  # noqa: E402
from synthetic.iq_io import read_iq, read_truth  # noqa: E402

GEN_DIR = os.path.join(ROOT, "Test-Data", "fsk")
REPORT_DIR = os.path.join(ROOT, "reports")


def q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def ber_noncoherent_orthogonal(M: int, esn0_db: float) -> float:
    es = 10 ** (esn0_db / 10)
    ps = sum((-1) ** (n + 1) * comb(M - 1, n) / (n + 1) * np.exp(-n / (n + 1) * es) for n in range(1, M))
    return ps * (M / 2) / (M - 1)


def ber_coherent_msk(esn0_db: float) -> float:
    return q(np.sqrt(2 * 10 ** (esn0_db / 10)))


def ascii_safe(s: str, n: int = 36) -> str:
    s = s.encode("ascii", "backslashreplace").decode().replace("|", "\\|")
    return s if len(s) <= n else s[:n] + "..."


def frame_errors(decode, frame_bits: str):
    """Bit errors over the whole frame, aligned at the detected sync word (None if no sync)."""
    if decode.resolution != "sync-word" or decode.payload_bit_start is None:
        return None
    ref = np.array(list(frame_bits), dtype=np.uint8)
    s0 = decode.payload_bit_start - 16 - len(SYNC_BITS)
    got = decode.hard_bits[s0:s0 + len(ref)]
    return int(np.count_nonzero(got != ref[:len(got)])) + (len(ref) - len(got)), len(ref)


def llr_agreement(decode, frame_bits: str) -> float:
    if decode.resolution != "sync-word":
        return float("nan")
    ref = np.array(list(frame_bits), dtype=np.uint8)
    s0 = decode.payload_bit_start - 16 - len(SYNC_BITS)
    llr = decode.llrs[s0:s0 + len(ref)]
    return float(np.mean((llr[:len(ref)] < 0).astype(np.uint8) == ref[:len(llr)])) if len(llr) else float("nan")


# ---------------------------------------------------------------------------
# Per-file diagnostic report (spec 2.7 format)
# ---------------------------------------------------------------------------

def diagnostic_text(name: str, truth: dict, r, passed: bool) -> str:
    p = r.parameters
    m = r.model
    prim = r.primary
    bits = "".join(map(str, prim.hard_bits[:24].tolist()))
    soft = ", ".join(f"{v:+.2f}" for v in prim.llrs[:6])
    fe = frame_errors(prim, truth.get("frame_bits", "")) if truth.get("frame_bits") else None
    L = ["=" * 72, "FSK DEMODULATION REPORT", "=" * 72, f"File: {name}"]
    if truth.get("generator"):
        L.append(f"Intended: {truth['modulation']}, M={truth['order']}, h={truth['h']}, pulse={truth['pulse']}"
                 f"{'' if truth['bt'] is None else ' BT=' + str(truth['bt'])}, Rs={truth['nominal_symbol_rate_hz']:.0f}, "
                 f"payload={truth['payload_text']!r}")
    L += ["-" * 72,
          f"Detection      : {r.family.family} (envelope excess CV {r.family.excess_cv:.3f}, p_fsk {r.family.p_fsk:.2f})",
          f"Classification : {r.variant}",
          f"  Evidence     : " + "\n                 ".join(e.description for e in r.evidence
                                                          if e.source in ("L3_TONES", "PSD_PEAKS", "FAMILY_ENVELOPE",
                                                                          "FAMILY_DEMOD", "L3_BACKTRACK")),
          "Parameters     :"]
    for key in ("order", "symbol_rate", "sps", "h", "tone_spacing", "deviation", "cfo", "pulse", "continuity", "snr"):
        if key in p:
            v = p[key].value
            vs = f"{v:.6g}" if isinstance(v, (int, float)) else str(v)
            L.append(f"  {p[key].name:18s}: {vs} {p[key].unit}  [{p[key].method}]")
    L += ["Sync           :",
          f"  Boundary     : sample {m.starts[0]:.2f} of the burst (sps {m.sps:.3f})",
          "Demodulation   :"]
    for meth, d in r.decodes.items():
        fe_m = frame_errors(d, truth.get("frame_bits", "")) if truth.get("frame_bits") else None
        L.append(f"  {meth:15s}: {d.resolution:14s} mapping '{d.hyp.label()}'"
                 + (f", CRC {'VALID' if d.crc_ok else 'FAIL'}" if d.frame else "")
                 + (f", frame BER {fe_m[0]}/{fe_m[1]}" if fe_m else ""))
    L += [f"  Primary      : {prim.method}",
          f"  Hard bits    : {bits}...",
          f"  Soft bits    : {soft}, ... (LLR = ln P0/P1)",
          f"  BER          : {'n/a' if fe is None else f'{100 * fe[0] / fe[1]:.2f}% ({fe[0]}/{fe[1]})'}",
          f"  FSK error    : {r.fsk_error_pct:.1f}% (RMS IF error / peak deviation)",
          f"  Decoded      : {prim.text!r}",
          f"  Status       : {'PASS' if passed else 'FAIL'}",
          "-" * 72]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# A. Generated vectors
# ---------------------------------------------------------------------------

def run_generated(regen: bool):
    if regen or not glob.glob(os.path.join(GEN_DIR, "*.iq")):
        for name, spec in fsk_suite():
            save_fsk(spec, os.path.join(GEN_DIR, name))
    os.makedirs(os.path.join(REPORT_DIR, "fsk"), exist_ok=True)
    rows = []
    for name, _ in fsk_suite():
        path = os.path.join(GEN_DIR, name + ".iq")
        t = read_truth(path)
        f = read_iq(path)
        t0 = time.time()
        r = receive_fsk(f.samples, f.sample_rate)
        dt = time.time() - t0
        prim = r.primary
        fe = frame_errors(prim, t["frame_bits"])
        passed = bool(prim.crc_ok and prim.payload == t["payload_text"].encode() and r.variant == t["modulation"])
        imp = t["impairments"]
        rows.append(dict(
            file=name, limitation=name.startswith("limit_"), intended=t["modulation"], classified=r.variant,
            M=(t["order"], r.model.order), h=(t["h"], r.model.h),
            pulse=(t["pulse"] if t["bt"] is None else f"G{t['bt']}",
                   r.model.pulse if r.model.bt is None else f"G{r.model.bt}"),
            cont=(t["continuous_phase"], r.model.continuous),
            sps=(t["samples_per_symbol"], r.model.sps),
            cfo=(imp["cfo_hz"], r.parameters["cfo"].value),
            snr=imp["snr_db"], esn0=(imp["esn0_db"], r.esn0_est_db), family=r.family.family,
            methods={k: ("CRC" if d.crc_ok else d.resolution) for k, d in r.decodes.items()},
            primary=prim.method, mapping=prim.hyp.label(),
            frame_errors=None if fe is None else fe[0], frame_bits=len(t["frame_bits"]),
            llr_agree=llr_agreement(prim, t["frame_bits"]), fsk_err=r.fsk_error_pct,
            decoded=prim.text, seconds=dt, passed=passed))
        with open(os.path.join(REPORT_DIR, "fsk", name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(diagnostic_text(name + ".iq", t, r, passed))
        print(f"  A {name:32s} {t['modulation']:6s}->{r.variant:6s} {prim.method:13s} "
              f"{'PASS' if passed else 'FAIL'} {dt:.1f}s")
    return rows


# ---------------------------------------------------------------------------
# B. Legacy
# ---------------------------------------------------------------------------

def sg_truth_positions(n_states: int, n_syms: int = 800) -> np.ndarray:
    """Reproduce sg.py's FSK symbol draw: np.random.seed(42); choice(linspace(-1,1,n), n_syms)."""
    np.random.seed(42)
    lv = np.linspace(-1, 1, n_states)
    return np.searchsorted(lv, np.random.choice(lv, n_syms))


def run_legacy():
    rows = []
    for n in ("BFSK", "4FSK", "8FSK"):
        path = os.path.join(ROOT, "Test-Data", "all_modulations_output", n + ".iq")
        if not os.path.exists(path):
            continue
        t = read_truth(path)
        f = read_iq(path)
        res = {}
        for mode, pl in (("known", t["payload_text"]), ("blind", None)):
            r = receive_fsk(f.samples, f.sample_rate, FSKReceiverConfig(expected_payload=pl))
            res[mode] = r
        k = res["known"]
        rows.append(dict(file=f"all_modulations_output/{n}.iq", kind="payload", intended=n, classified=k.variant,
                         M=k.model.order, h=k.model.h, sps=k.model.sps, pulse=k.model.pulse,
                         mapping=k.primary.hyp.label(), family=k.family.family,
                         known_ok=t["payload_text"].encode() in k.primary.payload,
                         blind_ok=t["payload_text"].encode() in res["blind"].primary.payload,
                         blind_method=res["blind"].primary.resolution, decoded=res["blind"].primary.text))
        print(f"  B {n}: {k.variant} known {rows[-1]['known_ok']} blind {rows[-1]['blind_ok']}")
    for name, M, h_nom in (("hello_2fsk", 2, 2.0), ("hello_4fsk", 4, 2 / 3)):
        path = os.path.join(ROOT, "Test-Data", name + ".iq")
        if not os.path.exists(path):
            continue
        f = read_iq(path)
        r = receive_fsk(f.samples, f.sample_rate)
        tr = sg_truth_positions(M)
        pos = r.positions
        best = (1.0, 0)
        for o in range(-3, 4):
            a, b = max(0, -o), len(pos) - max(0, o)
            ref = tr[max(0, o):max(0, o) + (b - a)]
            n_cmp = min(len(ref), b - a)
            if n_cmp > 0:
                ser = float(np.mean(pos[a:a + n_cmp] != ref[:n_cmp]))
                best = min(best, (ser, o))
        rows.append(dict(file=name + ".iq", kind="symbols", intended=f"{M}FSK (h {h_nom:.3f} nominal, "
                         f"{h_nom * 7 / 8:.3f} effective)", classified=r.variant, M=r.model.order, h=r.model.h,
                         sps=r.model.sps, pulse=r.model.pulse, family=r.family.family, ser=best[0],
                         n_symbols=len(pos)))
        print(f"  B {name}: {r.variant} SER {best[0]:.4f}")
    return rows


# ---------------------------------------------------------------------------
# C. Family rejection
# ---------------------------------------------------------------------------

REJECT_GLOBS = ["Test-Data/synthetic_psk/*.iq", "Test-Data/hello_bpsk.iq", "Test-Data/hello_qpsk.iq",
                "Test-Data/hello_8psk.iq", "Test-Data/hello_16psk.iq", "Test-Data/*psk_*baud_*db.iq",
                "Test-Data/*qam*.iq", "Test-Data/all_modulations_output/*[KM].iq",
                "Test-Data/qam_pangram_all_files/*.iq", "Test-Data/pack_my_box_bpsk.iq",
                "Test-Data/pangram_bpsk.iq", "Test-Data/bright_vixens_qpsk.iq"]


def run_rejection():
    files = []
    for g in REJECT_GLOBS:
        files += [p for p in glob.glob(os.path.join(ROOT, g)) if not p.endswith("_f32.iq") and "fsk" not in p.lower()]
    files = sorted(set(files))
    rows = []
    for p in files:
        x, _ = normalize_power(read_iq(p).samples.astype(complex))
        b = occupied_band(x)
        bu = detect_burst(x, b.center, b.bw99)
        dc, _ = estimate_dc(x, bu)
        x = x - dc
        xb = x[bu.start:bu.stop]
        npw = None if bu.whole_file else float(np.mean(np.abs(np.r_[x[:bu.start], x[bu.stop:]]) ** 2))
        fam = classify_family(xb, occupied_band(xb), npw)
        demod_crc = None
        if fam.family == "UNCERTAIN":           # let the FSK demodulator try: it must not find a valid frame
            f = read_iq(p)
            try:
                demod_crc = bool(receive_fsk(f.samples, f.sample_rate).primary.crc_ok)
            except ValueError:
                demod_crc = False             # no usable FSK hypothesis at all
        accepted = fam.family == "FSK" or bool(demod_crc)
        rows.append(dict(file=os.path.relpath(p, ROOT).replace("\\", "/"), family=fam.family,
                         excess_cv=fam.excess_cv, fsk_crc=demod_crc, accepted_as_fsk=accepted))
    print(f"  C {len(rows)} PSK/QAM files, accepted as FSK: {sum(r['accepted_as_fsk'] for r in rows)}")
    return rows


# ---------------------------------------------------------------------------
# D. BER sweep
# ---------------------------------------------------------------------------

SWEEP = [
    ("2FSK h=1", dict(order=2, h=1.0), [6, 8, 10, 12]),
    ("4FSK h=1", dict(order=4, h=1.0), [6, 8, 10, 12]),
    ("MSK", dict(order=2, h=0.5), [4, 6, 8, 10]),
    ("GMSK BT=0.3", dict(order=2, h=0.5, pulse="gaussian", bt=0.3), [4, 6, 8, 10]),
]
SWEEP_SPS = 8


def run_sweep(n_bytes: int = 300):
    rows = []
    rng = np.random.default_rng(11)
    for label, kw, points in SWEEP:
        for es in points:
            text = "".join(chr(c) for c in rng.integers(32, 127, n_bytes))
            spec = FSKSpec(**kw, symbol_rate=10_000.0, sample_rate=10_000.0 * SWEEP_SPS, payload_text=text,
                           impairments=FSKImpairments(snr_db=es - 10 * np.log10(SWEEP_SPS), cfo_hz=300.0,
                                                      phase_deg=float(rng.uniform(0, 360)),
                                                      timing_offset=float(rng.uniform()),
                                                      seed=int(rng.integers(1 << 30))))
            x, t = generate_fsk(spec)
            r = receive_fsk(x, spec.sample_rate)
            res = {}
            for meth, d in r.decodes.items():
                fe = frame_errors(d, t["frame_bits"])
                res[meth] = None if fe is None else fe[0] / fe[1]
            M = kw["order"]
            theory = ber_coherent_msk(es) if kw["h"] == 0.5 else ber_noncoherent_orthogonal(M, es)
            rows.append(dict(label=label, esn0=es, snr_sample=es - 10 * np.log10(SWEEP_SPS), bits=len(t["frame_bits"]),
                             classified=r.variant, ber=res, theory=theory,
                             theory_kind="coherent MSK/BPSK bound" if kw["h"] == 0.5 else f"non-coherent orthogonal {M}-FSK"))
            print(f"  D {label:12s} Es/N0 {es:4.1f} dB -> {r.variant:6s} "
                  + " ".join(f"{k}={'n/a' if v is None else f'{v:.2e}'}" for k, v in res.items())
                  + f" theory {theory:.2e}")
    return rows


def run_snr_grid():
    """Spec 2.9: per-sample SNR 0/5/10/20 dB at the spec's 20 sps (Es/N0 = SNR + 13 dB)."""
    rows = []
    rng = np.random.default_rng(5)
    for label, kw in (("2FSK h=1", dict(order=2, h=1.0)), ("4FSK h=0.5", dict(order=4, h=0.5)),
                      ("MSK", dict(order=2, h=0.5)), ("GMSK BT=0.3", dict(order=2, h=0.5, pulse="gaussian", bt=0.3))):
        for snr in (0, 5, 10, 20):
            spec = FSKSpec(**kw, impairments=FSKImpairments(snr_db=snr, cfo_hz=500.0, timing_offset=0.4,
                                                            seed=int(rng.integers(1 << 30))))
            x, t = generate_fsk(spec)
            r = receive_fsk(x, spec.sample_rate)
            fe = frame_errors(r.primary, t["frame_bits"])
            rows.append(dict(label=label, snr=snr, esn0=t["impairments"]["esn0_db"], classified=r.variant,
                             errors=None if fe is None else fe[0], bits=len(t["frame_bits"]),
                             payload_ok=r.primary.crc_ok and r.primary.payload == t["payload_text"].encode()))
            print(f"  D' {label:12s} SNR {snr:2d} dB -> {r.variant} errors {rows[-1]['errors']} ok {rows[-1]['payload_ok']}")
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(gen, legacy, rej, sweep, grid, path):
    L = ["# Phase 2 — FSK Family Validation Report\n",
         f"Generated by `tools/fsk_validate.py` on {time.strftime('%Y-%m-%d %H:%M')}. All demodulation is blind "
         "(samples + sample rate only). SNR convention: `SNR` = per-sample SNR as in the legacy `sg.py` Test-Data; "
         "Es/N0 = SNR + 10·log10(sps) (= SNR + 13 dB at the spec's 20 sps).\n", "## Summary\n"]
    core = [r for r in gen if not r["limitation"]]
    lim = [r for r in gen if r["limitation"]]
    L.append(f"* **Generated FSK vectors:** {sum(r['passed'] for r in core)}/{len(core)} PASS (variant correct, payload "
             f"byte-exact, CRC-16 valid)" + (f"; {len(lim)} documented-limitation vector(s) reported separately." if lim else "."))
    if legacy:
        pay = [r for r in legacy if r["kind"] == "payload"]
        sym = [r for r in legacy if r["kind"] == "symbols"]
        L.append(f"* **Legacy FSK (payload files):** expected payload {sum(r['known_ok'] for r in pay)}/{len(pay)}, "
                 f"fully blind {sum(r['blind_ok'] for r in pay)}/{len(pay)}. **Legacy sg.py random-symbol files:** "
                 + ", ".join(f"{r['file']} SER {r['ser']:.4f}" for r in sym) + ".")
    if rej:
        L.append(f"* **Family discrimination:** {len(rej)} PSK/QAM files, accepted as FSK: "
                 f"{sum(r['accepted_as_fsk'] for r in rej)}.")
    if grid:
        L.append(f"* **Spec SNR grid (per-sample 0/5/10/20 dB):** {sum(r['payload_ok'] for r in grid)}/{len(grid)} "
                 "payloads exact.")
    L.append("")

    L.append("## A. Generated vectors (`Test-Data/fsk/`)\n")
    L.append("Core matrix (spec 2.1) and robustness cases (spec 2.9). Common impairments unless noted: CFO 730 Hz, "
             "phase 40°, timing 0.31 T, clock +80 ppm, per-sample SNR 10 dB. Rs = 10 kBd, fs = 200 kHz (20 sps) unless "
             "noted. Per-file diagnostic reports: `reports/fsk/<name>.txt`.\n")
    L.append("| File | Intended | Classified | M | h true/est | Pulse true/est | CP true/est | sps true/est | "
             "CFO true/est (Hz) | SNR | Es/N0 true/est | Family | Demods (disc/corr/coh) | Primary | Frame BER | "
             "LLR agree | Decoded | Result |")
    L.append("|" + "---|" * 18)
    for r in core + lim:
        md = r["methods"]
        dem = "/".join(md.get(k, "-") for k in ("discriminator", "correlator", "coherent-bcjr"))
        ber = "n/a" if r["frame_errors"] is None else f"{r['frame_errors']}/{r['frame_bits']}"
        res = ("**PASS**" if r["passed"] else "**FAIL**") + (" (limitation)" if r["limitation"] else "")
        L.append(f"| {r['file']} | {r['intended']} | {r['classified']} | {r['M'][0]}/{r['M'][1]} | "
                 f"{r['h'][0]}/{r['h'][1]:.3f} | {r['pulse'][0]}/{r['pulse'][1]} | "
                 f"{'Y' if r['cont'][0] else 'N'}/{'Y' if r['cont'][1] else 'N'} | {r['sps'][0]:.3f}/{r['sps'][1]:.3f} | "
                 f"{r['cfo'][0]:.0f}/{r['cfo'][1]:.0f} | {r['snr']:.0f} dB | {r['esn0'][0]:.0f}/{r['esn0'][1]:.1f} | "
                 f"{r['family']} | {dem} | {r['primary']} | {ber} | {r['llr_agree']:.3f} | "
                 f"`{ascii_safe(r['decoded'], 20)}` | {res} |")
    L.append("")

    if legacy:
        L.append("## B. Legacy Test-Data FSK\n")
        L.append("`all_modulations_output/*FSK`: rectangular CPFSK, h = 0.5, 8 sps, **natural-binary** tone mapping, "
                 "payload \"BRIGHT VIXENS JUMP; DOZY FOWL QUACK\", no sync word, noiseless, the first symbol is clipped "
                 "by the generator. Without a sync word the mapping (Gray/natural x inversion x differential) and the "
                 "byte alignment come from content: *known* = expected payload, *blind* = English-text likelihood.\n")
        L.append("| File | Classified | M | h | sps | Tone mapping found | Known payload | Blind | Blind decode |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in [r for r in legacy if r["kind"] == "payload"]:
            L.append(f"| {r['file']} | {r['classified']} | {r['M']} | {r['h']:.3f} | {r['sps']:.3f} | {r['mapping']} | "
                     f"{'PASS' if r['known_ok'] else 'FAIL'} | {'PASS' if r['blind_ok'] else 'FAIL'} "
                     f"({r['blind_method']}) | `{ascii_safe(r['decoded'], 36)}` |")
        L.append("\n`sg.py` `hello_2fsk` / `hello_4fsk`: 800 random symbols (no payload), tones ±1200 Hz and "
                 "±400/±1200 Hz at 1200 Bd, per-sample SNR 20 dB. `sg.py` starts each symbol from the previous "
                 "symbol's *last sample*, so the phase advances only 7/8 of the nominal amount per symbol: the "
                 "effective h is 7/8 of nominal, which is what the estimator measures.\n")
        L.append("| File | Intended | Classified | M | h est | sps | Symbols | SER vs reproduced RNG |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in [r for r in legacy if r["kind"] == "symbols"]:
            L.append(f"| {r['file']} | {r['intended']} | {r['classified']} | {r['M']} | {r['h']:.3f} | {r['sps']:.3f} | "
                     f"{r['n_symbols']} | {r['ser']:.4f} |")
        L.append("")

    if rej:
        L.append("## C. Family discrimination (FSK vs PSK/QAM)\n")
        cnt = {k: sum(r["family"] == k for r in rej) for k in ("FSK", "UNCERTAIN", "LINEAR")}
        L.append(f"Envelope test on {len(rej)} PSK/QAM files: LINEAR {cnt['LINEAR']}, UNCERTAIN {cnt['UNCERTAIN']}, "
                 f"FSK {cnt['FSK']}. UNCERTAIN files were passed to the FSK demodulator, which must not produce a "
                 f"CRC-valid frame: {sum(1 for r in rej if r['fsk_crc'])} did. "
                 f"**Accepted as FSK: {sum(r['accepted_as_fsk'] for r in rej)}.**\n")
        unc = [r for r in rej if r["family"] != "LINEAR"]
        if unc:
            L.append("| File | Envelope test | Excess CV | FSK demod CRC |")
            L.append("|---|---|---|---|")
            for r in unc:
                L.append(f"| {r['file']} | {r['family']} | {r['excess_cv']:.3f} | {r['fsk_crc']} |")
        L.append("")

    if sweep:
        L.append("## D. BER vs Es/N0 per demodulator\n")
        L.append(f"300-byte random printable-text payloads (~2.5 kbit frames), {SWEEP_SPS} sps, CFO 300 Hz, random phase and "
                 "timing; BER over the whole frame from the detected sync word ('n/a' = that demodulator did not find the "
                 "sync word). Theory: non-coherent orthogonal M-FSK for h = 1; coherent MSK (= BPSK) bound for h = 0.5 "
                 "(GMSK BT 0.3 is expected ~0.5-1 dB worse).\n")
        L.append("| Signal | Es/N0 | per-sample SNR | Classified | Discriminator | Correlator | Coherent BCJR | Theory |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in sweep:
            b = r["ber"]
            f_ = lambda k: "-" if k not in b else ("n/a" if b[k] is None else f"{b[k]:.2e}")
            L.append(f"| {r['label']} | {r['esn0']} dB | {r['snr_sample']:.1f} dB | {r['classified']} | "
                     f"{f_('discriminator')} | {f_('correlator')} | {f_('coherent-bcjr')} | {r['theory']:.2e} "
                     f"({r['theory_kind']}) |")
        L.append("")
    if grid:
        L.append("### Spec 2.9 SNR grid (per-sample SNR at 20 sps)\n")
        L.append("| Signal | SNR 0 dB (Es/N0 13) | SNR 5 dB (18) | SNR 10 dB (23) | SNR 20 dB (33) |")
        L.append("|---|---|---|---|---|")
        for label in dict.fromkeys(r["label"] for r in grid):
            cells = [r for r in grid if r["label"] == label]
            L.append(f"| {label} | " + " | ".join(
                ("PASS" if c["payload_ok"] else "FAIL") + f" ({c['errors']}/{c['bits']} bit errors)" for c in cells) + " |")
        L.append("")

    L.append(STATIC_NOTES)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


STATIC_NOTES = """## Design

| Layer | Method (module) |
|---|---|
| L1 | Block AGC; DC from noise-only guard samples; blind IQ-imbalance correction by I/Q whitening, **accepted only if it flattens the envelope** (`demodulation/fsk_receiver.py`, `estimation/fsk_estimator.iq_balance`) |
| L2 | 97 % occupied band from the PSD (noise floor from the lower half of bins), channel filter, energy burst detection |
| L3 rate | Candidates: cyclic lines of the smoothed-IF difference and IF square law at several smoothing scales (scored jointly), plus the **x^P power-law line pair** (exact Rs for P·h integer: MSK/GMSK) and x2 / /2 relatives; local decision-directed refinement; **two-tier backtracking** (cheap sync screen, then full fits) validated by the CRC (`estimation/fsk_estimator.estimate_rate`) |
| L3 continuity | Mean unfiltered-IF magnitude per symbol-phase bin: phase jumps pile up in one bin (timing-free); for discontinuous FSK the spike histogram also gives the exact rate |
| L3/L4 tones | Median IF strobes -> equally spaced M-level fits (multi-start) -> **joint (M, pulse, BT, h, timing) re-modulation match** of the unfiltered signal against CPM models; Gaussian hypotheses use **ISI-MLSE (Viterbi) decisions**; M chosen by an **MDL rule** (ln M nats/symbol vs likelihood gain) |
| L4 family | Band-isolated envelope CV beyond the noise contribution (noise from guard or PSD); inconclusive cases are settled by demodulation evidence (sync/CRC) (`modulation/fsk_classifier.py`) |
| L5 | Timing from the refinement above; coherent phase reference by decision-directed comparison with the re-modulated CPM model |
| L6 | Discriminator (Gaussian LL on IF strobes), non-coherent tone-correlator bank (LL = 2|y|/N0), **coherent max-log BCJR over the CPM trellis** (phase state x L-1 previous symbols, rational h); each resolved against the sync word over Gray/natural x inversion x differential hypotheses; hard bits + max-log LLRs (`demodulation/fsk_demod.py`) |

## Root causes found along the way

| Symptom | Cause | Fix |
|---|---|---|
| Symbol rate wrong on nearly everything | 99 % occupied band swept in the noise floor (rect CPFSK side lobes), so the channel filter did nothing and the IF was noise | 97 % band with a noise-referenced floor |
| Integrate-and-dump strobe quality could not rank rates | Averages of k binary symbols form k+1 equally spaced levels, which a larger M grid fits perfectly | Rate from spectral lines; re-modulation used only locally |
| Re-modulation preferred sub-multiple rates | A normalised correlation over N samples has a ~1/N overfitting floor, so per-symbol windows favour small sps | Fixed-length windows (samples) for continuous CPM |
| GMSK / GFSK rate lines weak | Narrow Gaussian filtering leaves little IF energy near Rs/2 | x^2 line pair (exact Rs for h = 1/2); backtracking over harmonic relatives |
| Rect pulses labelled Gaussian BT 1.0 | Model compared with the channel-filtered signal, whose transitions look Gaussian | Model tests on the unfiltered burst |
| GMSK labelled 4FSK h 0.25; GFSK labelled 8GFSK | Gaussian ISI makes strobes multi-level; bigger grids fit ISI and noise | Joint (M, pulse) test, ISI-MLSE decisions, MDL penalty for M |
| Phase-discontinuous FSK modelled as 8-level | Continuous-phase model across phase jumps; filtered spikes | Timing-free continuity test on the unfiltered IF; per-symbol windows |
| Legacy BFSK labelled 8FSK h 0.08 | Blind IQ whitening "corrected" genuine x^2 structure of h = 1/2 CPM with biased data | Accept IQ correction only if it flattens the envelope |
| Legacy captures: first byte / whole payload wrong | Generator clips the first symbol (7 of 8 samples) and byte alignment then shifts | Keep >= 75 % symbols; overlap payload search; text test over 8 bit offsets |
| Legacy 8FSK h 0.41 | Top tone used once in 94 symbols; percentile-only initialisation | Multi-start level fit |

## Limitations (honest scope)

* `limit_gfsk_bt0.25_4ary` (4-ary GFSK, BT 0.25, h 0.5) is **not** recovered: the eye is nearly closed and an 8-level
  half-spacing grid explains the ISI states better than the true 4 levels even at the exact rate and timing. A full
  joint (M, pulse) CPM-trellis search (hundreds of states) would be needed. 4-GFSK at BT 0.5 is recovered.
* **Low-SNR MSK/GMSK is well short of theory.** In the Es/N0 sweep (8 sps, 2.5 kbit frames) 2FSK/4FSK track non-coherent theory (coherent BCJR ~2-3 dB better), but MSK at 8-10 dB has BER 2e-3..6e-2 versus the coherent bound 2e-4..4e-6, and GMSK loses sync or has BER ~1e-2..1e-1 below 10 dB. Causes: the coherent BCJR's carrier reference is decision-directed from noisy non-coherent decisions (a trellis with per-survivor phase tracking is the fix), and pulse/BT/rate estimation degrades below ~0 dB per-sample SNR. At the spec's operating point (per-sample SNR >= 0 dB at 20 sps) all payloads are exact.
* Gaussian-pulse hypotheses are tested for M <= 4 (8-ary GFSK not in scope).
* GFSK with a weak rate line relies on backtracking; such files take ~1 min (most files take 2-8 s).
* The coherent BCJR needs a rational h (q <= 16 within 3 %) and continuous phase; otherwise the non-coherent
  demodulators are used. Its carrier reference is decision-directed from the non-coherent decisions.
* Unframed captures need an expected payload or English-like content to resolve mapping and byte alignment.
* One burst per capture; RRC-shaped FSK (spec 2.1 option) is not generated (non-standard for CPFSK).

## Reproduce

```bash
python -m synthetic.fsk_generator --out Test-Data/fsk          # regenerate the FSK vectors
python -m pytest tests/unit/test_fsk_prototype.py -q            # unit/integration tests
python tools/fsk_validate.py                                    # this report (+ .json, reports/fsk/*.txt)
python tools/demod_fsk.py --file Test-Data/fsk/hello_gmsk.iq    # one file, full evidence
```

## Integration notes (L1-L6 now, L7-L11 later)

* `receive_fsk(samples, sample_rate)` mirrors `receive_psk`: both return parameters as `core.evidence.Parameter`
  (value, unit, KNOWN/ESTIMATED/UNKNOWN, method, evidence), an evidence list, hard bits and LLRs. The Phase 4 CLI
  will run the family test (`classify_family`) and dispatch to the family receiver; the QAM receiver (Phase 3) will
  follow the same contract.
* Hard bits + LLRs (ln P0/P1) per demodulator are the L7 input (bitstream forensics); the sync/length/CRC framing
  used here is test-harness framing only and will be replaced by the L10 frame-analysis stage.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--quick", action="store_true", help="skip the rejection test and the BER sweeps")
    ap.add_argument("--out", default=os.path.join(REPORT_DIR, "fsk_report.md"))
    args = ap.parse_args()
    print("A. generated vectors"); gen = run_generated(args.regen)
    print("B. legacy"); legacy = run_legacy()
    rej = sweep = grid = []
    if not args.quick:
        print("C. family rejection"); rej = run_rejection()
        print("D. BER sweep"); sweep = run_sweep()
        print("D'. SNR grid"); grid = run_snr_grid()
    write_report(gen, legacy, rej, sweep, grid, args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(dict(generated=gen, legacy=legacy, rejection=rej, sweep=sweep, snr_grid=grid), fh, indent=1,
                  default=str)
    print(f"report: {args.out}")
    core_ok = all(r["passed"] for r in gen if not r["limitation"])
    sys.exit(0 if core_ok else 1)


if __name__ == "__main__":
    main()
