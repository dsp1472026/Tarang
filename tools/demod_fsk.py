"""
Demodulate one FSK / CPFSK / MSK / GFSK / GMSK capture blind and print the evidence report.

    python tools/demod_fsk.py --file Test-Data/fsk/hello_gmsk.iq
    python tools/demod_fsk.py --file Test-Data/all_modulations_output/4FSK.iq --payload "BRIGHT VIXENS JUMP; DOZY FOWL QUACK"
    python tools/demod_fsk.py --file capture.iq --dtype cf32_le --fs 200000 --json out.json

Only the samples and the sample rate (from --fs or the sidecar) are used.
--payload resolves the mapping/byte alignment of captures WITHOUT a sync word.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demodulation.fsk_receiver import FSKReceiverConfig, receive_fsk  # noqa: E402
from ingestion.capture import load_capture  # noqa: E402
from synthetic.iq_io import read_truth  # noqa: E402
from tools.fsk_validate import diagnostic_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--wav-mode", choices=["auto", "iq", "real"], default="auto",
                    help=".wav: stereo = I/Q, mono = real passband (default)")
    ap.add_argument("--dtype", choices=["ci16_le", "cf32_le"])
    ap.add_argument("--fs", type=float, help="sample rate in Hz (default: sidecar; else unknown)")
    ap.add_argument("--payload", help="expected payload text (captures without a sync word)")
    ap.add_argument("--json", help="write parameters / evidence / bits / LLRs here")
    args = ap.parse_args()

    f = load_capture(args.file, dtype=args.dtype, sample_rate=args.fs, wav_mode=args.wav_mode)
    r = receive_fsk(f.samples, f.sample_rate, FSKReceiverConfig(expected_payload=args.payload))
    truth = read_truth(args.file[:-4] + ".iq" if args.file.endswith(".wav") else args.file)
    ok = bool(r.primary.crc_ok) if r.primary.frame else None
    print(diagnostic_text(args.file, truth if truth.get("generator") else {}, r, bool(ok)).encode(
        "ascii", "backslashreplace").decode())
    print("\nAll evidence:")
    for e in r.evidence:
        print(f"  [{e.source:15s} {e.score:4.2f}] {e.description}".encode("ascii", "backslashreplace").decode())
    if args.json:
        out = dict(file=args.file, variant=r.variant, family=r.family.family,
                   parameters={k: p.to_dict() for k, p in r.parameters.items()},
                   evidence=[e.to_dict() for e in r.evidence],
                   demodulators={k: dict(resolution=d.resolution, mapping=d.hyp.label(), crc_valid=d.crc_ok,
                                         payload_text=d.text) for k, d in r.decodes.items()},
                   primary=r.primary.method, payload_hex=r.primary.payload.hex(),
                   hard_bits="".join(map(str, r.hard_bits.tolist())),
                   llrs=[round(float(v), 3) for v in r.llrs], fsk_error_pct=r.fsk_error_pct,
                   esn0_db=r.esn0_est_db)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
