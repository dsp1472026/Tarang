"""
generate_pangram_wavs.py
------------------------
Generates stereo WAV files (I = left, Q = right) containing modulated
pangram payloads for SIH-26147 test bench.

Supported modulations:
    BPSK, QPSK, 8PSK         -> coherent PSK (Gray-coded)
    2-FSK, 4-FSK             -> CPFSK (continuous phase, h = 0.5)
    16-QAM, 64-QAM           -> Gray-coded square QAM

Each WAV is 16-bit PCM stereo at Fs = 96 kHz.
The right channel carries Q; the left channel carries I.
A sidecar JSON stores ground truth (bits, symbol rate, payload, etc.).
"""

import os
import json
import numpy as np
import scipy.signal as sps
from scipy.io import wavfile

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------

FS = 96_000              # sample rate (Hz)
SYMBOL_RATE = 4_800      # symbols / second
SPS = FS // SYMBOL_RATE  # samples per symbol (integer for simplicity)
ROLLOFF = 0.35           # RRC roll-off
RRC_SPAN = 10            # RRC filter span in symbols
AMPLITUDE = 0.7          # headroom to avoid clipping after pulse shaping

OUT_DIR = "pangram_wav"

PANGRAMS = {
    "pack_my_box":      "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS.",
    "sphinx_of_black":  "SPHINX OF BLACK QUARTZ, JUDGE MY VOW.",
    "how_vexingly":     "HOW VEXINGLY QUICK DAFT ZEBRAS JUMP!",
    "waltz_bad_nymph":  "WALTZ, BAD NYMPH, FOR QUICK JIGS VEX.",
}

# Fixed preamble (32 bits) — useful for frame sync + phase ambiguity
PREAMBLE_BITS = np.array(
    [1,0,1,0, 1,1,0,0, 0,1,1,1, 1,0,0,1,
     1,1,0,1, 0,0,1,0, 0,1,0,1, 1,0,1,1], dtype=np.uint8)


# ----------------------------------------------------------------------
# 2. BIT CONVERSION HELPERS
# ----------------------------------------------------------------------

def text_to_bits(text: str) -> np.ndarray:
    """ASCII text -> MSB-first bit array."""
    data = text.encode("ascii")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    """Bit array -> ASCII text (for round-trip verification)."""
    # Pad to a multiple of 8
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes().decode("ascii", errors="replace")


def bits_to_symbols(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Group bits MSB-first into symbol indices 0..2^k-1."""
    pad = (-len(bits)) % bits_per_symbol
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    groups = bits.reshape(-1, bits_per_symbol)
    weights = 1 << np.arange(bits_per_symbol - 1, -1, -1)
    return (groups * weights).sum(axis=1).astype(np.int32)


# ----------------------------------------------------------------------
# 3. GRAY CODE MAPPING
# ----------------------------------------------------------------------

def gray_to_binary(g: int) -> int:
    b = g
    shift = 1
    while (g >> shift) > 0:
        b ^= (g >> shift)
        shift += 1
    return b


def binary_to_gray(b: int) -> int:
    return b ^ (b >> 1)


def gray_map_indices(n_bits: int) -> np.ndarray:
    """Return a length-2^n_bits array mapping Gray index -> binary index."""
    return np.array([gray_to_binary(g) for g in range(1 << n_bits)], dtype=np.int32)


# ----------------------------------------------------------------------
# 4. PULSE SHAPING (RRC)
# ----------------------------------------------------------------------

def rrc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    """Root-Raised-Cosine filter, unit energy."""
    N = span * sps
    t = np.arange(-N // 2, N // 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                - (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + \
                  4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


def upsample_and_shape(symbols: np.ndarray, rrc: np.ndarray, sps: int) -> np.ndarray:
    """Insert SPS-1 zeros between symbols, then convolve with RRC."""
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    shaped = np.convolve(up, rrc.astype(np.float32), mode="same")
    return shaped


# ----------------------------------------------------------------------
# 5. PSK MODULATORS
# ----------------------------------------------------------------------

def modulate_bpsk(bits: np.ndarray) -> np.ndarray:
    """BPSK: bit 0 -> -1+0j, bit 1 -> +1+0j."""
    return (2 * bits.astype(np.float32) - 1).astype(np.complex64)


def modulate_qpsk(bits: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK.
       Symbol idx (Gray): 00->0, 01->1, 11->2, 10->3
       Constellation: exp(j*(2k+1)*pi/4) with k = binary index."""
    sym = bits_to_symbols(bits, 2)
    # Gray -> binary
    bin_idx = np.array([gray_to_binary(g) for g in sym], dtype=np.int32)
    phase = (2 * bin_idx + 1) * np.pi / 4.0
    return np.exp(1j * phase).astype(np.complex64)


def modulate_8psk(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 8PSK: exp(j*2*pi*k/8)."""
    sym = bits_to_symbols(bits, 3)
    bin_idx = np.array([gray_to_binary(g) for g in sym], dtype=np.int32)
    phase = 2 * np.pi * bin_idx / 8.0
    return np.exp(1j * phase).astype(np.complex64)


# ----------------------------------------------------------------------
# 6. QAM MODULATORS (Gray-coded square QAM)
# ----------------------------------------------------------------------

def pam_levels(M: int) -> np.ndarray:
    """1-D PAM levels for sqrt(M)-PAM: -(√M-1), -(√M-3), ..., +(√M-1)."""
    side = int(np.sqrt(M))
    return np.arange(-(side - 1), side, 2, dtype=np.float32)


def gray_pam_symbols(bits: np.ndarray, bits_per_axis: int) -> np.ndarray:
    """Map bits -> Gray-coded PAM levels for one axis."""
    gray_idx = bits_to_symbols(bits, bits_per_axis)      # 0..2^b-1
    bin_idx = np.array([gray_to_binary(g) for g in gray_idx], dtype=np.int32)
    levels = pam_levels(1 << bits_per_axis)
    return levels[bin_idx]


def modulate_qam(bits: np.ndarray, M: int) -> np.ndarray:
    """Square Gray-coded M-QAM (M = 16, 64, 256)."""
    bps = int(np.log2(M))
    bpa = bps // 2                    # bits per axis
    # Pad to a multiple of bps
    pad = (-len(bits)) % bps
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    # Split: even bits -> I, odd bits -> Q (arbitrary but consistent)
    I_bits = bits[0::2][: len(bits) // 2] if False else bits.reshape(-1, bps)[:, :bpa].reshape(-1)
    Q_bits = bits.reshape(-1, bps)[:, bpa:].reshape(-1)
    I = gray_pam_symbols(I_bits, bpa)
    Q = gray_pam_symbols(Q_bits, bpa)
    # Normalize to unit average power
    norm = np.sqrt((np.mean(I ** 2) + np.mean(Q ** 2)) / 2)
    symbols = (I + 1j * Q) / norm
    return symbols.astype(np.complex64)


# ----------------------------------------------------------------------
# 7. FSK MODULATOR (Continuous-Phase FSK)
# ----------------------------------------------------------------------

def modulate_cpfsk(bits: np.ndarray, M: int, sps: int, h: float = 0.5) -> np.ndarray:
    """
    Continuous-phase M-FSK.
        - Symbol index k in [0, M-1] -> freq deviation
              f_k = (2k - M + 1) * h / (2 * Ts)
        - Phase accumulated continuously: phi[n+1] = phi[n] + 2*pi*f_k*Ts
        - Output: exp(j*phi[n])
    Gray-coded symbol mapping.
    """
    bps = int(np.log2(M))
    gray_idx = bits_to_symbols(bits, bps)
    bin_idx = np.array([gray_to_binary(g) for g in gray_idx], dtype=np.int32)

    # Frequency deviation per symbol (in cycles/sample)
    freq_dev = (2 * bin_idx - M + 1) * h / (2 * sps)   # cycles / sample

    # Build the full IF array by repeating each symbol's freq for SPS samples
    f_inst = np.repeat(freq_dev, sps).astype(np.float64)

    # Continuous phase accumulation
    phase = np.cumsum(2 * np.pi * f_inst)
    return np.exp(1j * phase).astype(np.complex64)


# ----------------------------------------------------------------------
# 8. HIGH-LEVEL FRAME BUILDER
# ----------------------------------------------------------------------

def build_frame_bits(payload_text: str) -> np.ndarray:
    """Preamble + payload bits (no CRC yet — can be added later)."""
    payload_bits = text_to_bits(payload_text)
    return np.concatenate([PREAMBLE_BITS, payload_bits]).astype(np.uint8)


def modulate_frame(payload_text: str, modulation: str) -> np.ndarray:
    """Return complex baseband samples for a given modulation family."""
    bits = build_frame_bits(payload_text)

    if modulation == "bpsk":
        symbols = modulate_bpsk(bits)
        rrc = rrc_filter(ROLLOFF, SPS, RRC_SPAN)
        return upsample_and_shape(symbols, rrc, SPS)

    if modulation == "qpsk":
        symbols = modulate_qpsk(bits)
        rrc = rrc_filter(ROLLOFF, SPS, RRC_SPAN)
        return upsample_and_shape(symbols, rrc, SPS)

    if modulation == "8psk":
        symbols = modulate_8psk(bits)
        rrc = rrc_filter(ROLLOFF, SPS, RRC_SPAN)
        return upsample_and_shape(symbols, rrc, SPS)

    if modulation == "16qam":
        symbols = modulate_qam(bits, 16)
        rrc = rrc_filter(ROLLOFF, SPS, RRC_SPAN)
        return upsample_and_shape(symbols, rrc, SPS)

    if modulation == "64qam":
        symbols = modulate_qam(bits, 64)
        rrc = rrc_filter(ROLLOFF, SPS, RRC_SPAN)
        return upsample_and_shape(symbols, rrc, SPS)

    if modulation == "2fsk":
        return modulate_cpfsk(bits, M=2, sps=SPS, h=0.5)

    if modulation == "4fsk":
        return modulate_cpfsk(bits, M=4, sps=SPS, h=0.5)

    raise ValueError(f"Unknown modulation: {modulation}")


# ----------------------------------------------------------------------
# 9. WAV WRITER (stereo I/Q)
# ----------------------------------------------------------------------

def write_iq_wav(path: str, x: np.ndarray, fs: int = FS) -> None:
    """
    Write complex baseband x[n] to a stereo WAV:
        left  = I  (real part)
        right = Q  (imaginary part)
    Samples are normalized to int16 with peak headroom.
    """
    # Normalize to avoid clipping
    peak = max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)), 1e-12)
    scale = (2 ** 15 - 1) * 0.95 / peak
    I = np.clip(x.real * scale, -32768, 32767).astype(np.int16)
    Q = np.clip(x.imag * scale, -32768, 32767).astype(np.int16)
    stereo = np.stack([I, Q], axis=1)     # shape (N, 2)
    wavfile.write(path, fs, stereo)


# ----------------------------------------------------------------------
# 10. MAIN
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []

    modulations = ["bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk"]

    for mod in modulations:
        mod_dir = os.path.join(OUT_DIR, mod)
        os.makedirs(mod_dir, exist_ok=True)

        for key, text in PANGRAMS.items():
            # Build the modulated frame
            x = modulate_frame(text, mod)

            # Save WAV
            wav_path = os.path.join(mod_dir, f"{key}_{mod}.wav")
            write_iq_wav(wav_path, x, FS)

            # Save ground-truth sidecar
            bits = build_frame_bits(text)
            meta = {
                "payload_text": text,
                "payload_ascii_hex": text.encode("ascii").hex(),
                "modulation": mod,
                "sample_rate": FS,
                "symbol_rate": SYMBOL_RATE,
                "samples_per_symbol": SPS,
                "rolloff": ROLLOFF if mod not in ("2fsk", "4fsk") else None,
                "fsk_modulation_index": 0.5 if mod in ("2fsk", "4fsk") else None,
                "preamble_bits": PREAMBLE_BITS.tolist(),
                "num_bits_total": int(len(bits)),
                "num_symbols": int(len(bits) / max(1, int(np.log2(
                    {"bpsk": 2, "qpsk": 4, "8psk": 8,
                     "16qam": 16, "64qam": 64,
                     "2fsk": 2, "4fsk": 4}[mod])))),
                "bitstream_hex": np.packbits(
                    np.concatenate([bits, np.zeros((-len(bits)) % 8, dtype=np.uint8)])
                ).tobytes().hex(),
                "wav_channels": {"left": "I", "right": "Q"},
                "wav_format": "int16 stereo PCM",
                "wav_path": wav_path,
            }
            meta_path = os.path.join(mod_dir, f"{key}_{mod}.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            manifest.append({
                "modulation": mod,
                "pangram_key": key,
                "payload_text": text,
                "wav_path": wav_path,
                "meta_path": meta_path,
                "num_samples": len(x),
                "duration_sec": len(x) / FS,
            })

            print(f"[OK] {wav_path}  ({len(x)} samples, {len(x)/FS:.3f} s)")

    # Write global manifest
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Generated {len(manifest)} WAV files under '{OUT_DIR}/'.")
    print(f"Global manifest: {os.path.join(OUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()