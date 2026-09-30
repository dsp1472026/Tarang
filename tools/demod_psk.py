"""
Demodulate one PSK capture blind and print the evidence report.

    python tools/demod_psk.py --file Test-Data/synthetic_psk/qpsk_impaired.iq
    python tools/demod_psk.py --file Test-Data/hello_8psk.iq --payload "HELLO WORLD"
    python tools/demod_psk.py --file capture.iq --dtype cf32_le --fs 48000 --bits-out bits.txt --json out.json

Only the samples and the sample rate (from --fs or the file's sidecar) are
used. --payload is an optional expected payload for captures WITHOUT a sync
word (resolves the carrier-phase ambiguity); --modulation forces L4.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demodulation.psk_receiver import PSKReceiverConfig, receive_psk  # noqa: E402
from ingestion.capture import load_capture  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--wav-mode", choices=["auto", "iq", "real"], default="auto",
                    help=".wav: stereo = I/Q, mono = real passband (default)")
    ap.add_argument("--dtype", choices=["ci16_le", "cf32_le"], help="default: sidecar, else '_f32' naming")
    ap.add_argument("--fs", type=float, help="sample rate in Hz (default: sidecar; else unknown)")
    ap.add_argument("--payload", help="expected payload text (legacy captures without sync word)")
    ap.add_argument("--modulation", choices=["BPSK", "QPSK", "8PSK", "16PSK"], help="skip classification")
    ap.add_argument("--json", help="write parameters/evidence/bits to this JSON file")
    ap.add_argument("--bits-out", help="write hard bits (one line of 0/1) to this file")
    args = ap.parse_args()

    f = load_capture(args.file, dtype=args.dtype, sample_rate=args.fs, wav_mode=args.wav_mode)
    r = receive_psk(f.samples, f.sample_rate,
                    PSKReceiverConfig(modulation=args.modulation, expected_payload=args.payload))

    print("=" * 72)
    print(f"PSK ANALYSIS  {args.file}  ({f.source}, {len(f.samples)} samples; {f.notes[-1]})")
    print("=" * 72)
    print(r.summary().encode("ascii", "backslashreplace").decode())
    print("\nEvidence:")
    for e in r.evidence:
        print(f"  [{e.source:14s} {e.score:4.2f}] {e.description}")
    if r.classification:
        print("\nClassification posterior: " + ", ".join(f"{k}={v:.3f}" for k, v in r.classification.posterior.items()))
    print(f"\nHard bits: {len(r.hard_bits)}   soft bits (LLR = ln P0/P1): {len(r.llrs)}, "
          f"mean |LLR| {np.mean(np.abs(r.llrs)):.1f}")

    if args.bits_out:
        with open(args.bits_out, "w") as fh:
            fh.write("".join(map(str, r.hard_bits.tolist())) + "\n")
    if args.json:
        out = {
            "file": args.file,
            "modulation": r.modulation,
            "parameters": {k: p.to_dict() for k, p in r.parameters.items()},
            "classification": None if not r.classification else {
                "posterior": r.classification.posterior, "confidence": r.classification.confidence,
                "features": r.classification.features},
            "evidence": [e.to_dict() for e in r.evidence],
            "evm_pct": 100 * r.evm,
            "ambiguity_method": r.decode.method,
            "crc_valid": bool(r.decode.frame and r.decode.frame.crc_ok),
            "payload_hex": r.decode.payload.hex(),
            "payload_text": r.payload_text,
            "hard_bits": "".join(map(str, r.hard_bits.tolist())),
            "llrs": [round(float(v), 3) for v in r.llrs],
        }
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
