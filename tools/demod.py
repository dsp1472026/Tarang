"""
Blind classify + demodulate one capture (.iq or .wav).

    python tools/demod.py --file Test-Data/hello_qpsk.wav
    python tools/demod.py --file Test-Data/all_modulations_output/4FSK.wav --payload "BRIGHT VIXENS JUMP; DOZY FOWL QUACK"
    python tools/demod.py --file capture.iq --dtype cf32_le --fs 48000 --json out.json

Ingestion (ingestion/capture.py):
    .iq   interleaved I/Q (int16 or float32; --layout planar for I-block/Q-block files)
    .wav  stereo = I/Q, mono = real passband -> analytic signal, carrier estimate,
          mix to 0 Hz, decimate (--wav-mode iq|real overrides the stereo/mono rule)
Classification (modulation/unified_classifier.py): FSK vs PSK vs QAM, then order.
Demodulation: PSK (BPSK/QPSK/8PSK/16PSK), QAM (16/64/256) and the FSK family.

--payload is an optional expected payload for captures WITHOUT a sync word.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demodulation.fsk_receiver import FSKReceiverConfig, receive_fsk  # noqa: E402
from demodulation.psk_receiver import PSKReceiverConfig, receive_psk  # noqa: E402
from demodulation.qam_receiver import QAMReceiverConfig, receive_qam  # noqa: E402
from ingestion.capture import load_capture  # noqa: E402
from modulation.unified_classifier import (classify_capture, fsk_fit_is_single_carrier,  # noqa: E402
                                           reclassify_as_psk)


def ascii_safe(s: str) -> str:
    return s.encode("ascii", "backslashreplace").decode()


def run(path, dtype=None, fs=None, layout="interleaved", wav_mode="auto", payload=None, family=None):
    cap = load_capture(path, dtype=dtype, sample_rate=fs, layout=layout, wav_mode=wav_mode)
    cls = classify_capture(cap.samples, cap.sample_rate)
    result, fam = None, family or cls.family
    if fam == "PSK":
        if family and cls.family != "PSK":             # forced: fresh front end, PSK classifier decides the order
            result = receive_psk(cap.samples, cap.sample_rate, PSKReceiverConfig(expected_payload=payload))
        else:
            cfg = cls.front_config
            cfg.modulation, cfg.expected_payload = cls.modulation, payload
            result = receive_psk(cap.samples, cap.sample_rate, cfg, front=cls.front)
    elif fam == "QAM":
        result = receive_qam(cap.samples, cap.sample_rate,
                             QAMReceiverConfig(modulation=cls.modulation, expected_payload=payload),
                             front=cls.front, classification=cls.linear)
    elif fam == "FSK":
        result = receive_fsk(cap.samples, cap.sample_rate, FSKReceiverConfig(expected_payload=payload))
        if not family and fsk_fit_is_single_carrier(result):
            cls = reclassify_as_psk(cap.samples, cap.sample_rate, cls,
                                    f"FSK fit rejected: h = {result.parameters['h'].value:.4f} (one carrier with "
                                    "phase jumps) -> constant-envelope PSK")
            result = None
            if cls.family == "PSK":
                cfg = cls.front_config
                cfg.modulation, cfg.expected_payload = cls.modulation, payload
                result = receive_psk(cap.samples, cap.sample_rate, cfg, front=cls.front)
    return cap, cls, result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--dtype", choices=["ci16_le", "cf32_le"], help=".iq sample type (default: sidecar / '_f32' name)")
    ap.add_argument("--fs", type=float, help="sample rate in Hz (default: WAV header / sidecar)")
    ap.add_argument("--layout", choices=["interleaved", "planar"], default="interleaved")
    ap.add_argument("--wav-mode", choices=["auto", "iq", "real"], default="auto")
    ap.add_argument("--payload", help="expected payload text (captures without a sync word)")
    ap.add_argument("--family", choices=["PSK", "FSK"], help="skip the family decision")
    ap.add_argument("--json", help="write classification / parameters / bits / LLRs here")
    args = ap.parse_args()

    cap, cls, r = run(args.file, args.dtype, args.fs, args.layout, args.wav_mode, args.payload, args.family)

    print("=" * 72)
    print(f"CAPTURE  {args.file}")
    print("=" * 72)
    print(f"Source   : {cap.source}, {len(cap.samples)} complex samples @ "
          + (f"{cap.sample_rate:g} Hz" if cap.sample_rate else "unknown rate"))
    for note in cap.notes:
        print(f"           {note}")
    print(f"\nCLASSIFICATION: {cls.family} / {cls.modulation}   (confidence {cls.confidence:.1%}, P(FSK) {cls.p_fsk:.3f})")
    for e in cls.evidence:
        print(ascii_safe(f"  [{e.source:15s} {e.score:4.2f}] {e.description}"))

    out = {"file": args.file, "capture": {"source": cap.source, "sample_rate": cap.sample_rate,
                                          "carrier_hz": cap.carrier_hz, "decimation": cap.decimation,
                                          "notes": cap.notes},
           "classification": {"family": cls.family, "modulation": cls.modulation, "confidence": cls.confidence,
                              "p_fsk": cls.p_fsk, "features": cls.features,
                              "posterior": cls.linear.posterior if cls.linear else None,
                              "evidence": [e.to_dict() for e in cls.evidence]}}
    print()
    if r is None:
        print(f"{cls.modulation}: no demodulator for this signal.")
    elif hasattr(r, "variant"):                       # FSK
        print(f"FSK variant     : {r.variant}")
        for k, p in r.parameters.items():
            val = f"{p.value:.6g}" if isinstance(p.value, (int, float)) and p.value is not None else str(p.value)
            print(ascii_safe(f"{p.name:16s}: {val} {p.unit}  [{p.source}; {p.method}]"))
        print(f"Primary demod   : {r.primary.method}, CRC {'VALID' if r.primary.crc_ok else 'n/a or FAIL'}")
        print(ascii_safe(f"Payload         : {r.primary.text!r}"))
        out.update(modulation=r.variant, payload_text=r.primary.text, payload_hex=r.primary.payload.hex(),
                   crc_valid=bool(r.primary.crc_ok), parameters={k: p.to_dict() for k, p in r.parameters.items()},
                   hard_bits="".join(map(str, r.hard_bits.tolist())), llrs=[round(float(v), 3) for v in r.llrs])
    else:                                             # PSK / QAM
        print(ascii_safe(r.summary()))
        out.update(modulation=r.modulation, payload_text=r.payload_text, payload_hex=r.decode.payload.hex(),
                   crc_valid=bool(r.decode.frame and r.decode.frame.crc_ok),
                   parameters={k: p.to_dict() for k, p in r.parameters.items()},
                   hard_bits="".join(map(str, r.hard_bits.tolist())), llrs=[round(float(v), 3) for v in r.llrs])
    if r is not None:
        print(f"Hard bits: {len(r.hard_bits)}   soft bits (LLR = ln P0/P1): {len(r.llrs)}, "
              f"mean |LLR| {np.mean(np.abs(r.llrs)) if len(r.llrs) else 0:.1f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
