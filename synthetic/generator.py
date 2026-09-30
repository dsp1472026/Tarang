"""
Known-payload synthetic signal generator (PSK family in Phase 1).

Chain:
    payload text -> bytes -> frame bits (sync | len | payload | crc | pad)
    -> Gray-mapped symbols (optionally differentially encoded)
    -> RRC pulse shaping evaluated in continuous time
       (supports fractional timing offset, sample-clock offset, non-integer sps)
    -> carrier frequency offset + phase offset -> gain + DC offset
    -> AWGN at a specified Es/N0 -> .iq (int16 + float32) + JSON sidecar

SNR convention: `esn0_db` is the symbol-energy-to-noise-density ratio, i.e. the
SNR at the output of the matched filter. Per-sample SNR = Es/N0 - 10*log10(sps).
(The legacy sg.py "snr_db" is per-sample; its 20 dB files are ~29 dB Es/N0.)

Usage:
    python -m synthetic.generator --out Test-Data/synthetic_psk            # full Phase-1 suite
    python -m synthetic.generator --mod QPSK --esn0 15 --cfo 30 --out tmp  # single vector
"""

import argparse
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from dsp.constellations import (differential_encode, get_constellation, labels_of,
                                 positions_to_points)
from dsp.filters import rrc_pulse
from synthetic.framing import build_frame, bytes_to_bits
from synthetic.iq_io import write_iq

DEFAULT_PAYLOAD = "HELLO WORLD " * 8


@dataclass
class Impairments:
    esn0_db: Optional[float] = 30.0   # None -> noiseless
    cfo_hz: float = 0.0               # carrier frequency offset
    phase_deg: float = 0.0            # carrier phase offset
    timing_offset: float = 0.0        # fractional symbol delay in [0, 1)
    clock_ppm: float = 0.0            # sample-clock offset (symbol clock runs slow by this much)
    dc_offset: complex = 0j           # relative to RMS signal amplitude
    gain: float = 1.0
    seed: int = 1234


@dataclass
class SignalSpec:
    modulation: str = "BPSK"
    payload_text: str = DEFAULT_PAYLOAD
    symbol_rate: float = 1200.0
    sample_rate: float = 9600.0
    rrc_beta: float = 0.35
    rrc_span: int = 10
    guard_symbols: int = 24           # silence before and after the burst
    framed: bool = True               # False -> raw payload bits (legacy-style)
    differential: bool = False
    coding: Optional["CodingSpec"] = None   # Phase 4: FEC + interleaver (framed only)
    impairments: Impairments = field(default_factory=Impairments)


def shape(symbols: np.ndarray, sps: float, beta: float, span: int,
          delay_symbols: float, clock_ppm: float, n_samples: int) -> np.ndarray:
    """
    Continuous-time pulse shaping: s(t_n) = sum_k a_k h(t_n / T - k - delay).

    Evaluating the RRC pulse directly at every sample time makes fractional sps,
    fractional delays and clock drift exact (no resampling artefacts).
    Scaled so a unit-energy matched filter returns the symbol amplitudes.
    """
    t_sym = np.arange(n_samples) / sps * (1.0 - clock_ppm * 1e-6)   # sample times in symbol units
    out = np.zeros(n_samples, dtype=np.complex128)
    half = span / 2.0
    for k, a in enumerate(symbols):
        centre = k + delay_symbols
        lo = int(np.searchsorted(t_sym, centre - half))
        hi = int(np.searchsorted(t_sym, centre + half, side="right"))
        if hi > lo:
            out[lo:hi] += a * rrc_pulse(t_sym[lo:hi] - centre, beta)
    return out / np.sqrt(sps)


def generate(spec: SignalSpec):
    """Return (complex baseband samples, truth dict)."""
    imp = spec.impairments
    const = get_constellation(spec.modulation)
    bps = const.bits_per_symbol
    payload = spec.payload_text.encode("ascii")

    coding_info = None
    if spec.framed and spec.coding is not None:
        from synthetic.channel_coding import encode_frame
        bits, coding_info = encode_frame(payload, spec.coding, bps)
        layout = coding_info["layout"]
    elif spec.framed:
        bits, layout = build_frame(payload, bps)
        layout = layout.to_dict()
    else:
        bits = bytes_to_bits(payload)
        bits = np.concatenate([bits, np.zeros((-len(bits)) % bps, dtype=np.uint8)])
        layout = None

    labels = labels_of(bits, bps)
    if spec.differential:
        pos = differential_encode(labels, const.order)
        data_syms = positions_to_points(pos, const)
    else:
        data_syms = const.points[labels]

    sps = spec.sample_rate / spec.symbol_rate
    g = spec.guard_symbols
    n_sym_total = len(data_syms) + 2 * g
    n_samples = int(np.ceil(n_sym_total * sps))
    syms = np.concatenate([np.zeros(g), data_syms, np.zeros(g)])
    x = shape(syms, sps, spec.rrc_beta, spec.rrc_span, imp.timing_offset, imp.clock_ppm, n_samples)

    n = np.arange(n_samples)
    x = x * np.exp(1j * (2 * np.pi * imp.cfo_hz / spec.sample_rate * n + np.deg2rad(imp.phase_deg)))

    rng = np.random.default_rng(imp.seed)
    noise_var = 0.0
    if imp.esn0_db is not None:
        # Es = 1 (unit-power constellation, unit-energy pulse) -> N0 = 10^(-Es/N0 / 10)
        noise_var = 10 ** (-imp.esn0_db / 10.0)
        x = x + np.sqrt(noise_var / 2) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))

    rms = np.sqrt(np.mean(np.abs(data_syms) ** 2) / sps)
    x = imp.gain * (x + imp.dc_offset * rms)

    truth = {
        "generator": "synthetic.generator v1",
        "modulation": const.name,
        "family": const.family,
        "bits_per_symbol": bps,
        "mapping": ("gray; angle = phi0 + 2*pi*k/M, label = k ^ (k >> 1)" if const.family == "PSK" else
                    "square QAM, Gray per axis: label = [I half | Q half], level index = inverse_gray(half), "
                    "levels ascending"),
        "constellation_points": {int(i): [float(p.real), float(p.imag)] for i, p in enumerate(const.points)},
        "differential": spec.differential,
        "framed": spec.framed,
        "frame_layout": layout,
        "coding": coding_info,
        "payload_text": spec.payload_text,
        "payload_hex": payload.hex(),
        "frame_bits": "".join(map(str, bits.tolist())),
        "num_data_symbols": int(len(data_syms)),
        "first_symbol_sample": float((g + imp.timing_offset) * sps),
        "symbol_rate_hz": spec.symbol_rate,
        "sample_rate_hz": spec.sample_rate,
        "samples_per_symbol": sps,
        "rrc_beta": spec.rrc_beta,
        "rrc_span": spec.rrc_span,
        "guard_symbols": g,
        "impairments": {**asdict(imp), "dc_offset": [imp.dc_offset.real, imp.dc_offset.imag],
                        "per_sample_snr_db": None if imp.esn0_db is None else imp.esn0_db - 10 * np.log10(sps),
                        "noise_var_per_symbol": noise_var},
    }
    return x, truth


def save(spec: SignalSpec, stem: str) -> dict:
    x, truth = generate(spec)
    write_iq(stem, x, spec.sample_rate, 0.0, truth)
    return truth


# ---------------------------------------------------------------------------
# Phase-1 PSK test suite
# ---------------------------------------------------------------------------

# Es/N0 for the "impaired" scenarios: comfortably above the ~1e-4 SER point
# for each order so a few-hundred-symbol payload is recovered error-free.
PSK_WORKING_ESN0 = {"BPSK": 12.0, "QPSK": 15.0, "8PSK": 20.0}


def psk_suite():
    """(name, SignalSpec) pairs for the Phase-1 validation suite."""
    cases = []
    for i, mod in enumerate(["BPSK", "QPSK", "8PSK"]):
        tag = mod.lower()
        es = PSK_WORKING_ESN0[mod]
        cases.append((f"{tag}_clean", SignalSpec(mod, impairments=Impairments(esn0_db=30.0, seed=10 + i))))
        cases.append((f"{tag}_impaired", SignalSpec(mod, impairments=Impairments(
            esn0_db=es, cfo_hz=37.0, phase_deg=73.0, timing_offset=0.37, clock_ppm=150.0,
            dc_offset=0.05 + 0.03j, gain=0.3, seed=20 + i))))
        # Non-integer sps (10 kHz / 1200 Bd = 8.333), large CFO (~8% of Rs), negative clock offset.
        cases.append((f"{tag}_fracsps_bigcfo", SignalSpec(mod, sample_rate=10000.0, impairments=Impairments(
            esn0_db=es + 3, cfo_hz=-95.0, phase_deg=-140.0, timing_offset=0.61, clock_ppm=-120.0,
            seed=30 + i))))
        # Different roll-off and higher symbol rate (legacy 4800 Bd style, sps = 8).
        cases.append((f"{tag}_beta025_4800bd", SignalSpec(mod, symbol_rate=4800.0, sample_rate=38400.0, rrc_beta=0.25,
                                                         impairments=Impairments(esn0_db=es + 3, cfo_hz=150.0,
                                                                                 phase_deg=200.0, timing_offset=0.2,
                                                                                 seed=40 + i))))
        cases.append((f"{tag}_differential", SignalSpec(mod, differential=True, impairments=Impairments(
            esn0_db=es + 3, cfo_hz=21.0, phase_deg=111.0, timing_offset=0.45, seed=50 + i))))
        cases.append((f"{tag}_short_burst", SignalSpec(mod, payload_text="HELLO WORLD", impairments=Impairments(
            esn0_db=es + 5, cfo_hz=12.0, phase_deg=-33.0, timing_offset=0.8, seed=60 + i))))
    return cases


# Phase-3 QAM suite. Working Es/N0 ~ the 1e-5 SER point + margin for each order.
QAM_WORKING_ESN0 = {"16QAM": 22.0, "64QAM": 28.0, "256QAM": 34.0}


def qam_suite():
    """(name, SignalSpec) pairs for the Phase-3 validation suite."""
    cases = []
    for i, mod in enumerate(["16QAM", "64QAM", "256QAM"]):
        tag = mod.lower()
        es = QAM_WORKING_ESN0[mod]
        cases.append((f"{tag}_clean", SignalSpec(mod, impairments=Impairments(esn0_db=40.0, seed=110 + i))))
        cases.append((f"{tag}_impaired", SignalSpec(mod, impairments=Impairments(
            esn0_db=es + 2, cfo_hz=37.0, phase_deg=73.0, timing_offset=0.37, clock_ppm=150.0,
            dc_offset=0.03 + 0.02j, gain=0.3, seed=120 + i))))
        cases.append((f"{tag}_fracsps_bigcfo", SignalSpec(mod, sample_rate=10000.0, impairments=Impairments(
            esn0_db=es + 3, cfo_hz=-95.0, phase_deg=-140.0, timing_offset=0.61, clock_ppm=-120.0, seed=130 + i))))
        cases.append((f"{tag}_beta025_4800bd", SignalSpec(mod, symbol_rate=4800.0, sample_rate=38400.0, rrc_beta=0.25,
                                                         impairments=Impairments(esn0_db=es + 3, cfo_hz=150.0,
                                                                                 phase_deg=200.0, timing_offset=0.2,
                                                                                 seed=140 + i))))
        cases.append((f"{tag}_short_burst", SignalSpec(mod, payload_text="HELLO WORLD", impairments=Impairments(
            esn0_db=es + 5, cfo_hz=12.0, phase_deg=-33.0, timing_offset=0.8, seed=150 + i))))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="Test-Data/synthetic_psk", help="output directory")
    ap.add_argument("--mod", help="single vector: BPSK | QPSK | 8PSK (omit for the full suite)")
    ap.add_argument("--name", help="file stem for a single vector")
    ap.add_argument("--payload", default=DEFAULT_PAYLOAD)
    ap.add_argument("--esn0", type=float, default=30.0)
    ap.add_argument("--cfo", type=float, default=0.0, help="Hz")
    ap.add_argument("--phase", type=float, default=0.0, help="deg")
    ap.add_argument("--timing", type=float, default=0.0, help="fraction of a symbol")
    ap.add_argument("--ppm", type=float, default=0.0)
    ap.add_argument("--baud", type=float, default=1200.0)
    ap.add_argument("--fs", type=float, default=9600.0)
    ap.add_argument("--beta", type=float, default=0.35)
    ap.add_argument("--differential", action="store_true")
    ap.add_argument("--unframed", action="store_true", help="raw payload bits, no sync/len/crc")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.mod:
        spec = SignalSpec(args.mod.upper(), args.payload, args.baud, args.fs, args.beta,
                          framed=not args.unframed, differential=args.differential,
                          impairments=Impairments(args.esn0, args.cfo, args.phase, args.timing, args.ppm,
                                                  seed=args.seed))
        stem = os.path.join(args.out, args.name or f"{args.mod.lower()}_custom")
        save(spec, stem)
        print(f"wrote {stem}.iq / _f32.iq / .json")
        return

    for name, spec in psk_suite():
        stem = os.path.join(args.out, name)
        t = save(spec, stem)
        print(f"wrote {name:28s} {t['modulation']:5s} syms={t['num_data_symbols']:4d} "
              f"sps={t['samples_per_symbol']:.3f} Es/N0={spec.impairments.esn0_db} dB")


if __name__ == "__main__":
    main()
