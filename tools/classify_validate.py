"""
Classification (and optional end-to-end decode) validation over every Test-Data
capture, .iq and .wav side by side.

    python tools/classify_validate.py                  # classification only
    python tools/classify_validate.py --decode         # + blind demodulation, payload vs truth
    python tools/classify_validate.py --only wav --decode

Truth labels come from the sidecars (validation only; the classifier never sees
them). A classification is correct when the family matches and, for PSK/QAM,
the exact modulation matches (4QAM counts as QPSK). FSK variants are named by
the FSK receiver, so with --decode the FSK entry also needs a correct payload.

Writes reports/classification_report.md and .json.
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic.iq_io import read_truth  # noqa: E402
from tools.demod import run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSK_NAMES = {"2FSK", "4FSK", "8FSK", "MSK", "GMSK", "GFSK", "4GFSK", "CPFSK"}


def norm(m: str) -> str:
    m = m.upper().replace("-", "").replace("_", "")
    # 4QAM and QPSK are the same constellation: the classifier cannot (and should not) tell them
    # apart; the demodulator names 4QAM from its square-QAM bit labels.
    return {"4QAM": "QPSK", "BFSK": "2FSK", "FSK": "2FSK"}.get(m, m)


def family(label: str) -> str:
    if "QAM" in label:
        return "QAM"
    if label in FSK_NAMES or "FSK" in label or label.endswith("MSK"):
        return "FSK"
    return "PSK"


def truth_of(path: str):
    """(label, payload text or None) from the sidecars."""
    iq = path[:-4] + ".iq" if path.endswith(".wav") else path
    t = read_truth(iq)
    label, text = t.get("modulation"), t.get("payload_text")
    stem = os.path.splitext(iq)[0]
    for sc in (stem + ".json", stem + "_metadata.json"):
        if os.path.exists(sc):
            with open(sc) as fh:
                d = json.load(fh)
            label = label or d.get("modulation")
            text = text or d.get("message") or d.get("input_text")
    if label is None:
        b = os.path.basename(stem).lower()
        label = next((k for k in ("256qam", "64qam", "16qam", "4qam", "16psk", "8psk", "qpsk", "bpsk",
                                  "8fsk", "4fsk", "2fsk", "gmsk", "msk") if k in b), None)
    return (norm(label) if label else None), text


def captures(only=None):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "Test-Data", "**", "*.iq"), recursive=True)):
        if p.endswith("_f32.iq"):
            continue
        if only != "wav":
            out.append(p)
        w = p[:-3] + ".wav"
        if only != "iq" and os.path.exists(w):
            out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode", action="store_true", help="also demodulate and compare the payload")
    ap.add_argument("--only", choices=["iq", "wav"])
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "classification_report.md"))
    args = ap.parse_args()

    rows = []
    for path in captures(args.only):
        label, text = truth_of(path)
        if label is None:
            continue
        fam = family(label)
        t0 = time.time()
        cap, cls, r = run(path) if args.decode else (None, None, None)
        if not args.decode:
            from ingestion.capture import load_capture
            from modulation.unified_classifier import classify_capture
            cap = load_capture(path)
            cls = classify_capture(cap.samples, cap.sample_rate)
        dt = time.time() - t0
        cls_ok = cls.family == fam and (fam == "FSK" or cls.modulation == label)
        row = dict(file=os.path.relpath(path, ROOT).replace("\\", "/"), kind=path[-3:], truth=label, family=fam,
                   classified=f"{cls.family}/{cls.modulation}", cls_ok=cls_ok, p_fsk=round(cls.p_fsk, 3),
                   confidence=round(cls.confidence, 3), seconds=round(dt, 1), source=cap.source)
        if args.decode:
            decoded = None
            if r is not None:
                decoded = r.primary.text if hasattr(r, "variant") else r.payload_text
                row["demod_modulation"] = r.variant if hasattr(r, "variant") else r.modulation
            row["decoded"] = decoded
            row["payload_ok"] = None if (text is None or r is None) else (decoded == text)
        rows.append(row)
        flag = "ok " if cls_ok and row.get("payload_ok") in (None, True) else "BAD"
        print(f"{flag} {row['file']:58s} {label:6s} -> {row['classified']:12s}"
              + (f" payload {row['payload_ok']}" if args.decode else "") + f"  {dt:.1f}s", flush=True)

    write_report(rows, args)


def write_report(rows, args):
    L = ["# Unified classification report", ""]
    L.append("Blind FSK / PSK / QAM classification over every Test-Data capture (`.iq` and `.wav`). "
             "Truth comes from the sidecars and is used only for scoring.")
    L.append("")
    by = Counter()
    good = Counter()
    for r in rows:
        k = (r["kind"], r["truth"])
        by[k] += 1
        good[k] += r["cls_ok"]
    L += ["| Input | Truth | Files | Classified correctly |" + (" Payload exact |" if args.decode else ""),
          "|---|---|---|---|" + ("---|" if args.decode else "")]
    for k in sorted(by):
        line = f"| {k[0]} | {k[1]} | {by[k]} | {good[k]}/{by[k]} |"
        if args.decode:
            pr = [r for r in rows if (r["kind"], r["truth"]) == k and r.get("payload_ok") is not None]
            line += f" {sum(r['payload_ok'] for r in pr)}/{len(pr)} |" if pr else " n/a |"
        L.append(line)
    tot = sum(r["cls_ok"] for r in rows)
    L += ["", f"**Total: {tot}/{len(rows)} classified correctly** "
              f"(family {sum(r['classified'].split('/')[0] == r['family'] for r in rows)}/{len(rows)})."]
    bad = [r for r in rows if not r["cls_ok"] or r.get("payload_ok") is False]
    if bad:
        L += ["", "## Failures", "", "| File | Truth | Classified | Payload |", "|---|---|---|---|"]
        for r in bad:
            L.append(f"| `{r['file']}` | {r['truth']} | {r['classified']} | {r.get('decoded')!r} |")
    L.append("")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    print("\n".join(L[4:4 + len(by) + 4]))


if __name__ == "__main__":
    main()
