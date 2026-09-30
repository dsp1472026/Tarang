"""
IQ sample-format x full-charset end-to-end validation (GUI path).

For every modulation, a message containing every upper- and lower-case letter, every
digit and every printable ASCII special character is generated (framed with sync/CRC,
and unframed raw ASCII bits), written WITHOUT a sidecar and under a neutral file name in
each SDR sample type:

    CU8 (uint8)  CS8/CI8 (int8)  CS16/CI16/SC16 (int16)  CF32/FC32 (float32)  CF64/FC64 (float64)

and then taken through exactly what the GUI does:
    IQSetupDialog  -> blind sample-type detection (must equal the written type),
                      user enters the recording duration -> sample rate = samples / duration
    UnifiedPipelineWorker.work() -> classification + demodulation + decoded text
Pass = type detected correctly AND decoded text byte-exact (CRC valid when framed).

    python tools/iq_format_charset_validate.py              # full matrix, parallel
    python tools/iq_format_charset_validate.py --mods BPSK,16QAM --formats cu8,cf64
Writes reports/iq_format_charset_report.md (+ .json).
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

import numpy as np  # noqa: E402

FULL_CHARSET = ("The Quick Brown Fox Jumps Over The Lazy Dog! 0123456789 "
                "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ "
                "~!@#$%^&*()_+-=[]{}|;':\",./<>?`\\")
FORMATS = ["cu8", "cs8", "cs16", "cf32", "cf64"]
MODS = ["BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "64QAM", "256QAM", "2FSK", "4FSK", "8FSK", "GMSK"]
ESN0 = {"BPSK": 15, "QPSK": 18, "8PSK": 24, "16PSK": 30, "16QAM": 27, "64QAM": 33, "256QAM": 39}


def make_signal(mod: str, text: str, framed: bool, seed: int):
    if mod.endswith("FSK") or mod == "GMSK":
        from synthetic.fsk_generator import FSKImpairments, FSKSpec, generate_fsk
        order = 2 if mod == "GMSK" else int(mod[0])
        spec = FSKSpec(order, 0.5 if mod == "GMSK" else 1.0, pulse="gaussian" if mod == "GMSK" else "rect",
                       bt=0.3 if mod == "GMSK" else None, payload_text=text, framed=framed,
                       impairments=FSKImpairments(snr_db=12.0, cfo_hz=530.0, phase_deg=40.0, timing_offset=0.31,
                                                  seed=seed))
        x, truth = generate_fsk(spec)
    else:
        from synthetic.generator import Impairments, SignalSpec, generate
        spec = SignalSpec(mod, payload_text=text, framed=framed,
                          impairments=Impairments(esn0_db=ESN0[mod], cfo_hz=23.0, phase_deg=37.0,
                                                  timing_offset=0.3, seed=seed))
        x, truth = generate(spec)
    return x, truth


def run_case(mod: str, fmt: str, framed: bool, text: str, seed: int, outdir: str) -> dict:
    from gui.qt_compat import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841  (dialog widgets need it)
    from gui.widgets.iq_duration_dialog import IQSetupDialog
    from gui.workers.unified_worker import UnifiedPipelineWorker
    from ingestion.iq_formats import encode_iq, normalize_datatype

    x, truth = make_signal(mod, text, framed, seed)
    fs_true = truth["sample_rate_hz"]
    path = os.path.join(outdir, f"rec_{mod}_{fmt}_{'f' if framed else 'u'}_{seed}.iq")   # neutral name
    y = x if fmt in ("cu8", "cs8", "cs16") else 0.37 * x
    with open(path, "wb") as fh:
        fh.write(encode_iq(y, fmt))

    t0 = time.time()
    dlg = IQSetupDialog(path)                       # blind type detection, as in the GUI
    detected = dlg.selected_format
    dlg.duration_edit.setText(f"{len(x) / fs_true * 1e3:.12g}")    # the user types the duration (ms)
    dlg.unit_combo.setCurrentText("Milliseconds (ms)")
    dlg.update_calculations()
    cfg = dlg.get_config()
    config = {"file_path": path, "dtype": normalize_datatype(cfg["dtype"])[0], "endianness": "little",
              "iq_order": "IQ", "byte_offset": cfg["byte_offset"], "sample_rate": cfg["sample_rate"],
              "center_frequency": 0.0, "threshold_db": 6.0}
    try:
        res = UnifiedPipelineWorker(config).work()
        got, crc = res["payload_text"] or "", bool(res["crc_valid"])
        mod_out = str(res["parameters"]["Modulation"].value)
        err = None
    except Exception as e:                          # noqa: BLE001
        got, crc, mod_out, err = "", False, "-", f"{type(e).__name__}: {e}"
    os.remove(path)
    ok_text = got == text if framed else text in got
    return dict(mod=mod, fmt=fmt, framed=framed, detected=detected, fmt_ok=detected == fmt,
                fs_computed=cfg["sample_rate"], fs_true=fs_true, classified=mod_out, crc=crc,
                text_ok=ok_text, passed=detected == fmt and ok_text and (crc or not framed),
                decoded=got, error=err, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mods", default=",".join(MODS))
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--framing", default="framed,unframed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--text", default=FULL_CHARSET)
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "iq_format_charset_report"))
    args = ap.parse_args()

    cases = [(m, f, fr == "framed") for m in args.mods.split(",") for f in args.formats.split(",")
             for fr in args.framing.split(",")]
    outdir = tempfile.mkdtemp(prefix="iqfmt_")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_case, m, f, fr, args.text, 1000 + i, outdir): (m, f, fr)
                for i, (m, f, fr) in enumerate(cases)}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(f"{'PASS' if r['passed'] else 'FAIL'} {r['mod']:7s} {r['fmt']:5s} "
                  f"{'framed  ' if r['framed'] else 'unframed'} det={r['detected']:5s} cls={r['classified']:8s} "
                  f"crc={int(r['crc'])} {r['seconds']:6.1f}s" + ("" if r["passed"] else
                                                                  f"  got={r['decoded'][:70]!a} {r['error'] or ''}"),
                  flush=True)
    rows.sort(key=lambda r: (MODS.index(r["mod"]) if r["mod"] in MODS else 99, FORMATS.index(r["fmt"]),
                             not r["framed"]))
    n_ok = sum(r["passed"] for r in rows)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report + ".json", "w") as fh:
        json.dump({"text": args.text, "rows": rows}, fh, indent=1)
    lines = ["# IQ sample format x full-charset validation", "",
             f"Message ({len(args.text)} chars): `{args.text}`", "",
             f"**{n_ok}/{len(rows)} passed** (type detected blind from the bytes, rate from the entered "
             f"duration, decoded text byte-exact).", "",
             "| Modulation | Format | Framing | Detected | Classified | CRC | Text | Time |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['mod']} | {r['fmt'].upper()} | {'framed' if r['framed'] else 'unframed'} | "
                     f"{r['detected'].upper()}{'' if r['fmt_ok'] else ' ✗'} | {r['classified']} | "
                     f"{'✓' if r['crc'] else '-'} | {'✓' if r['text_ok'] else '✗'} | {r['seconds']} s |")
    with open(args.report + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n{n_ok}/{len(rows)} passed -> {args.report}.md")


if __name__ == "__main__":
    main()
