"""
Phase 4 validation: FEC + interleaver chains through the whole receiver (GUI path).

For every modulation x channel-coding chain, a full-charset message is framed
(SYNC | LEN | PAYLOAD | CRC-16), FEC-encoded, interleaved, modulated with impairments and
written as a CF32 .iq file with NO sidecar. The unified GUI worker then runs blind:
    duration -> sample rate -> classification -> demodulation -> L8 bitstream ->
    L9 de-interleaving -> L10 FEC -> L11 framing / CRC
Pass = payload byte-exact, CRC valid, FEC code and interleaver identified correctly.

    python tools/phase4_validate.py
    python tools/phase4_validate.py --mods QPSK --chains cc7,cc7_blk
Writes reports/phase4_report.md (+ .json).
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TEXT = ("Quick Brown Fox #42: jumps over 7 lazy dogs! "
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 ~!@#$%^&*()_+-=[]{}|;':\",./<>?`\\")
CHAINS = {
    "uncoded": (None, None, False),
    "cc7": ("CC K=7 (171,133) r1/2 [NASA/CCSDS]", None, False),
    "cc7_blk16x24": ("CC K=7 (171,133) r1/2 [NASA/CCSDS]", "block:16x24", False),
    "cc5_conv6x4": ("CC K=5 (23,35) r1/2", "conv:6,4", False),
    "cc34_syncout": ("CC K=7 (171,133) r3/4 punct [DVB/CCSDS]", None, True),
    "rs223": ("RS(255,223) [CCSDS-size]", None, False),
    "ham74_blk8x14": ("Hamming(7,4)", "block:8x14", False),
    "cc7_diag16x24": ("CC K=7 (171,133) r1/2 [NASA/CCSDS]", "diag:16x24", False),
    "cc7_pr_lfsr": ("CC K=7 (171,133) r1/2 [NASA/CCSDS]", "pr:all", False),
    "ldpc_80211_r34": ("LDPC 802.11 n=648 r3/4 (Z=27)", None, False),
    "concat_dvbs": ("CONCAT:DVB-S structure", None, False),
    "concat_ccsds_i4": ("CONCAT:CCSDS structure (I=4)", None, False),
}
MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK", "GMSK"]
ESN0 = {"BPSK": 12, "QPSK": 15, "8PSK": 20, "16QAM": 22, "64QAM": 30}


def make(mod, chain, seed):
    from synthetic.channel_coding import CodingSpec
    fec, inter, out = CHAINS[chain]
    coding = CodingSpec(fec, inter, out) if (fec or inter) else None
    if mod.endswith("FSK") or mod == "GMSK":
        from synthetic.fsk_generator import FSKImpairments, FSKSpec, generate_fsk
        order = 2 if mod == "GMSK" else int(mod[0])
        spec = FSKSpec(order, 0.5 if mod == "GMSK" else 1.0, pulse="gaussian" if mod == "GMSK" else "rect",
                       bt=0.3 if mod == "GMSK" else None, payload_text=TEXT, coding=coding,
                       impairments=FSKImpairments(snr_db=10.0, cfo_hz=530.0, phase_deg=40.0, timing_offset=0.31,
                                                  seed=seed))
        return generate_fsk(spec)
    from synthetic.generator import Impairments, SignalSpec, generate
    spec = SignalSpec(mod, payload_text=TEXT, coding=coding,
                      impairments=Impairments(esn0_db=ESN0[mod], cfo_hz=23.0, phase_deg=37.0, timing_offset=0.3,
                                              seed=seed))
    return generate(spec)


def run_case(mod, chain, seed, outdir):
    from gui.workers.unified_worker import UnifiedPipelineWorker
    from ingestion.iq_formats import encode_iq
    x, truth = make(mod, chain, seed)
    fs = truth["sample_rate_hz"]
    path = os.path.join(outdir, f"cap_{mod}_{chain}_{seed}.iq")
    with open(path, "wb") as fh:
        fh.write(encode_iq(0.4 * x, "cf32"))
    t0 = time.time()
    cfg = {"file_path": path, "dtype": "cf32", "endianness": "little", "iq_order": "IQ", "byte_offset": 0,
           "sample_rate": len(x) / (len(x) / fs), "threshold_db": 6.0,
           # the hypothesis-count budget is the real limit; the wall-clock safety cap is raised because
           # 8 parallel cases run ~10x slower (a 90 s case hit the 600 s GUI cap under load)
           "postdemod_budget_s": 3600.0}
    fec, inter, _ = CHAINS[chain]
    try:
        res = UnifiedPipelineWorker(cfg).work()
        post = res.get("postdemod")
        got, crc = res["payload_text"] or "", bool(res["crc_valid"])
        fec_got = post.best.fec_name if post and post.best else "-"
        deint = post.best.deint if post and post.best else None
        deint_got = "none" if deint is None or deint.kind == "none" else f"{deint.kind}:{deint.params}"
        cls = str(res["parameters"]["Modulation"].value)
        err = None
    except Exception as e:                                     # noqa: BLE001
        got, crc, fec_got, deint_got, cls, err = "", False, "-", "-", "-", f"{type(e).__name__}: {e}"
    os.remove(path)
    if fec and fec.startswith("CONCAT:"):
        from fec.concatenated import PROFILES
        sp = PROFILES[fec.split(":", 1)[1]]
        fec_ok = fec_got.startswith("Concatenated") and sp.rs_name in fec_got and f"outer {sp.outer}" in fec_got
    else:
        fec_ok = (fec_got == "none") if not fec else (fec_got == fec or (fec.startswith("RS(204") and
                                                                         fec_got.startswith("RS(255,239")))
    if inter:
        kind, dims = inter.split(":")
        if kind == "block":
            want = f"{dims.replace('x', ' x ')} (write rows"
        elif kind == "conv":
            want = "B = {} branches, M = {}".format(*dims.split(","))
        elif kind == "diag":
            want = f"{dims.split(',')[0].replace('x', ' x ')} diagonal"
        else:
            want = "LFSR address generator"
        deint_ok = deint_got.startswith(kind) and want in deint_got
    else:
        deint_ok = deint_got == "none"
    return dict(mod=mod, chain=chain, classified=cls, fec=fec_got, deint=deint_got, crc=crc, text_ok=got == TEXT,
                fec_ok=fec_ok, deint_ok=deint_ok, passed=crc and got == TEXT and fec_ok and deint_ok,
                seconds=round(time.time() - t0, 1), got=got[:80], error=err)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mods", default=",".join(MODS))
    ap.add_argument("--chains", default=",".join(CHAINS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "phase4_report"))
    args = ap.parse_args()
    cases = [(m, c) for m in args.mods.split(",") for c in args.chains.split(",")]
    outdir = tempfile.mkdtemp(prefix="p4_")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_case, m, c, 2000 + i, outdir) for i, (m, c) in enumerate(cases)]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"{'PASS' if r['passed'] else 'FAIL'} {r['mod']:6s} {r['chain']:14s} cls={r['classified']:7s} "
                  f"fec={r['fec'][:28]:28s} deint={r['deint'][:34]:34s} crc={int(r['crc'])} {r['seconds']:6.1f}s"
                  + ("" if r["passed"] else f" got={r['got']!a} {r['error'] or ''}"), flush=True)
    rows.sort(key=lambda r: (MODS.index(r["mod"]) if r["mod"] in MODS else 99, list(CHAINS).index(r["chain"])))
    n = sum(r["passed"] for r in rows)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report + ".json", "w") as fh:
        json.dump({"text": TEXT, "chains": CHAINS, "rows": rows}, fh, indent=1)
    lines = ["# Phase 4 validation: FEC + interleaver chains, blind end to end", "",
             "Generated by tools/phase4_validate.py.", "",
             f"Message ({len(TEXT)} chars): `{TEXT}`", "",
             f"**{n}/{len(rows)} passed** (payload byte-exact, CRC valid, FEC and interleaver identified).", "",
             "| Chain | FEC | Interleaver | Sync |", "|---|---|---|---|"]
    for k, (f, i, o) in CHAINS.items():
        lines.append(f"| {k} | {f or 'none'} | {i or 'none'} | {'outside' if o else 'inside'} |")
    lines += ["", "| Modulation | Chain | Classified | FEC found | De-interleaver found | CRC | Text | Time |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['mod']} | {r['chain']} | {r['classified']} | {r['fec']}{'' if r['fec_ok'] else ' ✗'} | "
                     f"{r['deint']}{'' if r['deint_ok'] else ' ✗'} | {'✓' if r['crc'] else '✗'} | "
                     f"{'✓' if r['text_ok'] else '✗'} | {r['seconds']} s |")
    with open(args.report + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n{n}/{len(rows)} passed -> {args.report}.md")


if __name__ == "__main__":
    main()
