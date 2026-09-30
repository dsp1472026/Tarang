"""
Known-payload FSK / CPFSK / MSK / GFSK / GMSK generator (Phase 2).

Chain:
    payload -> frame bits (96-bit PN sync | len | payload | CRC-16 | pad)
    -> M-ary labels (Gray or natural) -> optional differential encoding
    -> tone levels a = 2k - M + 1
    -> continuous-phase modulation phi(t) = pi*h*sum a_k P(t - c_k), evaluated in
       continuous time (exact for fractional sps, timing offset, clock drift),
       rect (CPFSK/MSK) or Gaussian (GFSK/GMSK, BT) frequency pulse;
       or, with continuous_phase=False, independent oscillators with a random
       phase jump at every symbol boundary
    -> CFO + phase -> IQ imbalance -> DC offset -> AWGN -> .iq + JSON sidecar

SNR convention: `snr_db` is the PER-SAMPLE SNR (as in the legacy sg.py
Test-Data); Es/N0 = snr_db + 10*log10(sps) is recorded in the sidecar too.

    python -m synthetic.fsk_generator --out Test-Data/fsk          # Phase-2 suite
"""

import argparse
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from dsp.constellations import gray, labels_of
from dsp.cpm import cpm_phase, fsk_variant_name
from synthetic.framing import build_frame, bytes_to_bits
from synthetic.iq_io import write_iq


@dataclass
class FSKImpairments:
    snr_db: Optional[float] = 10.0     # per-sample SNR; None -> noiseless
    cfo_hz: float = 0.0
    phase_deg: float = 0.0
    timing_offset: float = 0.0         # fraction of a symbol
    clock_ppm: float = 0.0             # symbol clock slower by this much (+10000 ppm = 1 % slower)
    dc_offset: complex = 0j            # relative to the unit signal amplitude
    iq_gain_db: float = 0.0            # Q-branch gain imbalance
    iq_phase_deg: float = 0.0          # Q-branch phase skew
    seed: int = 4321


@dataclass
class FSKSpec:
    order: int = 2
    h: float = 1.0
    symbol_rate: float = 10_000.0
    sample_rate: float = 200_000.0
    pulse: str = "rect"                # "rect" | "gaussian"
    bt: Optional[float] = None
    mapping: str = "gray"              # "gray" | "natural"
    continuous_phase: bool = True
    differential: bool = False
    payload_text: str = "HELLO WORLD FSK"
    framed: bool = True
    guard_symbols: int = 12
    coding: Optional["CodingSpec"] = None   # Phase 4: FEC + interleaver (framed only)
    impairments: FSKImpairments = field(default_factory=FSKImpairments)


def generate_fsk(spec: FSKSpec):
    imp = spec.impairments
    M = spec.order
    bps = int(np.log2(M))
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

    # labels -> tone positions k (inverse of the mapping), optional differential stage
    if spec.mapping == "gray":
        pos_of_label = np.empty(M, dtype=np.int64)
        pos_of_label[gray(np.arange(M))] = np.arange(M)
    else:
        pos_of_label = np.arange(M)
    if spec.differential:
        # The data select the tone-position INCREMENT: pos_n = pos_{n-1} + pos_of_label[L_n] (mod M),
        # after a reference symbol at position 0. The receiver needs no absolute tone reference.
        positions = np.concatenate([[0], np.cumsum(pos_of_label[labels])]) % M
    else:
        positions = pos_of_label[labels]
    levels = 2 * positions - M + 1

    sps = spec.sample_rate / spec.symbol_rate
    g = spec.guard_symbols
    n_samples = int(np.ceil((len(levels) + 2 * g) * sps))
    t_sym = np.arange(n_samples) / sps * (1.0 - imp.clock_ppm * 1e-6) - (g + imp.timing_offset)
    active = (t_sym >= 0) & (t_sym < len(levels))

    rng = np.random.default_rng(imp.seed)
    if spec.continuous_phase:
        phi = cpm_phase(levels, t_sym, spec.h, spec.pulse, spec.bt)
    else:
        if spec.pulse != "rect":
            raise ValueError("phase-discontinuous FSK is only defined for rect pulses")
        k = np.clip(np.floor(t_sym).astype(np.int64), 0, len(levels) - 1)
        jumps = rng.uniform(0, 2 * np.pi, len(levels))
        phi = np.pi * spec.h * levels[k] * (t_sym - k) + jumps[k]
    x = np.where(active, np.exp(1j * phi), 0.0)

    n = np.arange(n_samples)
    x = x * np.exp(1j * (2 * np.pi * imp.cfo_hz / spec.sample_rate * n + np.deg2rad(imp.phase_deg)))
    if imp.iq_gain_db or imp.iq_phase_deg:
        gq = 10 ** (imp.iq_gain_db / 20)
        ph = np.deg2rad(imp.iq_phase_deg)
        x = x.real + 1j * gq * (np.cos(ph) * x.imag - np.sin(ph) * x.real)
    x = x + imp.dc_offset
    noise_var = 0.0
    if imp.snr_db is not None:
        noise_var = 10 ** (-imp.snr_db / 10)
        x = x + np.sqrt(noise_var / 2) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))

    name = fsk_variant_name(M, spec.h, spec.pulse, spec.bt, spec.continuous_phase)
    truth = {
        "generator": "synthetic.fsk_generator v1",
        "modulation": name,
        "family": "FSK",
        "order": M,
        "bits_per_symbol": bps,
        "h": spec.h,
        "pulse": spec.pulse,
        "bt": spec.bt,
        "mapping": spec.mapping,
        "tone_levels": "a = 2k - M + 1; f = a*h*Rs/2 (+CFO); label = gray(k) or k",
        "continuous_phase": spec.continuous_phase,
        "differential": spec.differential,
        "framed": spec.framed,
        "frame_layout": layout,
        "coding": coding_info,
        "payload_text": spec.payload_text,
        "payload_hex": payload.hex(),
        "frame_bits": "".join(map(str, bits.tolist())),
        "tone_positions": positions.tolist(),
        "num_data_symbols": int(len(levels)),
        "first_symbol_sample": float((g + imp.timing_offset) * sps / (1.0 - imp.clock_ppm * 1e-6)),
        "symbol_rate_hz": spec.symbol_rate * (1.0 - imp.clock_ppm * 1e-6),
        "nominal_symbol_rate_hz": spec.symbol_rate,
        "sample_rate_hz": spec.sample_rate,
        "samples_per_symbol": sps / (1.0 - imp.clock_ppm * 1e-6),
        "tone_spacing_hz": spec.h * spec.symbol_rate,
        "peak_deviation_hz": (M - 1) * spec.h * spec.symbol_rate / 2,
        "guard_symbols": g,
        "impairments": {**asdict(imp), "dc_offset": [imp.dc_offset.real, imp.dc_offset.imag],
                        "esn0_db": None if imp.snr_db is None else imp.snr_db + 10 * np.log10(sps),
                        "noise_var_per_sample": noise_var},
    }
    return x, truth


def save_fsk(spec: FSKSpec, stem: str) -> dict:
    x, truth = generate_fsk(spec)
    write_iq(stem, x, spec.sample_rate, 0.0, truth)
    return truth


# ---------------------------------------------------------------------------
# Phase-2 suite (Test-Data/fsk/)
# ---------------------------------------------------------------------------

def fsk_suite():
    I = FSKImpairments
    base = dict(cfo_hz=730.0, phase_deg=40.0, timing_offset=0.31, clock_ppm=80.0)
    return [
        # --- core matrix (spec 2.1), per-sample SNR 10 dB (Es/N0 23 dB at 20 sps) ---
        ("hello_2fsk_h0.5", FSKSpec(2, 0.5, impairments=I(**base, seed=1))),
        ("hello_2fsk_h1.0", FSKSpec(2, 1.0, impairments=I(**base, seed=2))),
        ("hello_4fsk_h0.5", FSKSpec(4, 0.5, impairments=I(**base, seed=3))),
        ("hello_4fsk_h1.0", FSKSpec(4, 1.0, impairments=I(**base, seed=4))),
        ("hello_8fsk_h0.5", FSKSpec(8, 0.5, impairments=I(**base, seed=5))),
        ("hello_msk", FSKSpec(2, 0.5, symbol_rate=9600.0, sample_rate=153_600.0,
                              impairments=I(cfo_hz=-410.0, phase_deg=200.0, timing_offset=0.66, seed=6))),
        ("hello_gmsk", FSKSpec(2, 0.5, pulse="gaussian", bt=0.3, impairments=I(**base, seed=7))),
        ("hello_gfsk_bt0.3", FSKSpec(2, 0.35, pulse="gaussian", bt=0.3, impairments=I(**base, seed=8))),
        # --- robustness (spec 2.9) ---
        ("edge_cfo_plus5pct", FSKSpec(2, 0.7, impairments=I(cfo_hz=500.0, timing_offset=0.5, seed=9))),
        ("edge_cfo_minus5pct", FSKSpec(4, 1.0, impairments=I(cfo_hz=-500.0, timing_offset=0.2, seed=10))),
        ("edge_rate_plus1pct", FSKSpec(4, 0.5, impairments=I(clock_ppm=-10_000.0, cfo_hz=200.0, seed=11))),
        ("edge_rate_minus1pct", FSKSpec(2, 1.0, impairments=I(clock_ppm=10_000.0, cfo_hz=-300.0, seed=12))),
        ("edge_phase_discontinuous_2fsk", FSKSpec(2, 1.0, continuous_phase=False,
                                                  impairments=I(**base, seed=13))),
        ("edge_phase_discontinuous_4fsk", FSKSpec(4, 1.0, continuous_phase=False,
                                                  impairments=I(**base, seed=14))),
        ("edge_7p3sps_4fsk", FSKSpec(4, 0.5, sample_rate=73_000.0, impairments=I(**base, seed=15))),
        ("edge_7p3sps_gmsk", FSKSpec(2, 0.5, pulse="gaussian", bt=0.3, sample_rate=73_000.0,
                                     impairments=I(**base, seed=16))),
        ("edge_gfsk_bt0.5", FSKSpec(2, 0.5, pulse="gaussian", bt=0.5, impairments=I(**base, seed=17))),
        ("edge_gfsk_bt0.5_4ary", FSKSpec(4, 0.5, pulse="gaussian", bt=0.5, impairments=I(**base, seed=25))),
        # Documented limitation (not a pass criterion): 4-ary GFSK with BT 0.25 has a nearly
        # closed eye; see the report.
        ("limit_gfsk_bt0.25_4ary", FSKSpec(4, 0.5, pulse="gaussian", bt=0.25, impairments=I(**base, seed=18))),
        ("edge_short_burst_msk", FSKSpec(2, 0.5, payload_text="HI", impairments=I(**base, seed=19))),
        ("edge_dc_iq_imbalance", FSKSpec(2, 1.0, impairments=I(**base, dc_offset=0.15 - 0.1j, iq_gain_db=1.0,
                                                               iq_phase_deg=5.0, seed=20))),
        ("edge_differential_4fsk", FSKSpec(4, 1.0, differential=True, impairments=I(**base, seed=21))),
        ("edge_natural_mapping_4fsk", FSKSpec(4, 0.5, mapping="natural", impairments=I(**base, seed=22))),
        ("edge_lowsnr0db_2fsk", FSKSpec(2, 1.0, impairments=I(**base, snr_db=0.0, seed=23))),
        ("edge_lowsnr0db_msk", FSKSpec(2, 0.5, impairments=I(**base, snr_db=0.0, seed=24))),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="Test-Data/fsk")
    args = ap.parse_args()
    for name, spec in fsk_suite():
        t = save_fsk(spec, os.path.join(args.out, name))
        print(f"wrote {name:32s} {t['modulation']:6s} M={t['order']} h={t['h']} syms={t['num_data_symbols']:4d} "
              f"sps={t['samples_per_symbol']:.3f} SNR={spec.impairments.snr_db} dB (Es/N0 "
              f"{t['impairments']['esn0_db']:.1f} dB)")


if __name__ == "__main__":
    main()
