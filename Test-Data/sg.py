"""
Synthetic Signal Generator for SIGINT Multi-Modulation Test Suite.
Generates BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, and 4FSK signals in both .iq and .wav formats.
"""

import os
import argparse
import wave
from math import gcd
import numpy as np
from scipy.signal import resample_poly


def generate_modulation(mod_type: str = "BPSK", n_syms: int = 1000, sps: int = 8, snr_db: float = 20.0) -> np.ndarray:
    """
    Generates baseband IQ samples for specified modulation type with RRC pulse shaping and AWGN.
    """
    np.random.seed(42)
    mod_type = mod_type.upper()

    if mod_type == "BPSK":
        bits = np.random.choice([-1, 1], n_syms)
        syms = bits.astype(np.complex128)
    elif mod_type == "QPSK":
        bits = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], n_syms) / np.sqrt(2)
        syms = bits.astype(np.complex128)
    elif mod_type == "8PSK":
        pts = np.exp(1j * np.pi/4 * np.arange(8))
        syms = np.random.choice(pts, n_syms).astype(np.complex128)
    elif mod_type in ["16QAM", "QAM16"]:
        grid = np.array([a + 1j*b for a in [-3, -1, 1, 3] for b in [-3, -1, 1, 3]]) / np.sqrt(10)
        syms = np.random.choice(grid, n_syms).astype(np.complex128)
    elif mod_type in ["64QAM", "QAM64"]:
        grid = np.array([a + 1j*b for a in range(-7, 8, 2) for b in range(-7, 8, 2)]) / np.sqrt(42)
        syms = np.random.choice(grid, n_syms).astype(np.complex128)
    elif mod_type in ["2FSK", "4FSK"]:
        n_states = 2 if mod_type == "2FSK" else 4
        f_dev = 1200.0
        fs = 9600.0
        t_sym = np.arange(sps) / fs
        symbols = np.random.choice(np.linspace(-1, 1, n_states), n_syms)
        phase = 0.0
        samples = []
        for s in symbols:
            freq = s * f_dev
            p = phase + 2.0 * np.pi * freq * t_sym
            samples.extend(np.exp(1j * p))
            phase = p[-1]
        samples = np.array(samples, dtype=np.complex128)
        
        # Add AWGN
        sig_p = np.mean(np.abs(samples)**2)
        noise_p = sig_p / (10.0**(snr_db / 10.0))
        noise = (np.random.randn(len(samples)) + 1j * np.random.randn(len(samples))) * np.sqrt(noise_p / 2.0)
        return (samples + noise).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")

    # Upsample + RRC Pulse Shaping for PSK / QAM
    iq_up = np.zeros(n_syms * sps, dtype=np.complex128)
    iq_up[::sps] = syms

    beta = 0.35
    span = 10
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1 - beta + 4*beta/np.pi
        elif abs(abs(ti) - 1/(4*beta)) < 1e-8:
            h[i] = (beta/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) - (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1 - (4*beta*ti)**2)
            h[i] = num/den
    h = h / np.sqrt(np.sum(h**2))

    iq_shaped = np.convolve(iq_up, h, mode='same').astype(np.complex128)

    # Add AWGN Noise
    sig_p = np.mean(np.abs(iq_shaped)**2)
    noise_p = sig_p / (10.0**(snr_db / 10.0))
    noise = (np.random.randn(len(iq_shaped)) + 1j * np.random.randn(len(iq_shaped))) * np.sqrt(noise_p / 2.0)
    return (iq_shaped + noise).astype(np.complex64)


def save_signal(mod_type: str, samples: np.ndarray, output_dir: str = "Test-Data", sps: int = 8, baud: int = 1200, fs: int = 48000, fc: int = 6000):
    """
    Saves generated IQ samples as .iq (int16 and float32) and passband audio .wav files.
    """
    os.makedirs(output_dir, exist_ok=True)
    mod_lower = mod_type.lower()

    # 1. Save int16 .iq file
    scale = 32767 / (np.max(np.abs(samples)) + 1e-12) * 0.95
    iq_scaled = samples * scale
    interleaved_i16 = np.empty(2 * len(samples), dtype=np.int16)
    interleaved_i16[0::2] = np.clip(iq_scaled.real, -32768, 32767).astype(np.int16)
    interleaved_i16[1::2] = np.clip(iq_scaled.imag, -32768, 32767).astype(np.int16)
    path_i16 = os.path.join(output_dir, f"hello_{mod_lower}.iq")
    interleaved_i16.tofile(path_i16)
    print(f"Saved {path_i16} ({interleaved_i16.nbytes} bytes, int16 interleaved I/Q)")

    # 2. Save float32 .iq file
    interleaved_f32 = np.empty(2 * len(samples), dtype=np.float32)
    interleaved_f32[0::2] = samples.real
    interleaved_f32[1::2] = samples.imag
    path_f32 = os.path.join(output_dir, f"hello_{mod_lower}_f32.iq")
    interleaved_f32.tofile(path_f32)
    print(f"Saved {path_f32} ({interleaved_f32.nbytes} bytes, float32 interleaved I/Q)")

    # 3. Save mono 16-bit .wav file
    bb_fs = baud * sps
    g = gcd(fs, bb_fs)
    up, down = fs // g, bb_fs // g
    I = resample_poly(samples.real, up, down)
    Q = resample_poly(samples.imag, up, down)
    t = np.arange(len(I)) / fs
    rf = I * np.cos(2 * np.pi * fc * t) - Q * np.sin(2 * np.pi * fc * t)
    rf = rf / (np.max(np.abs(rf)) + 1e-12) * 0.9
    rf_int = (rf * 32767).astype(np.int16)

    path_wav = os.path.join(output_dir, f"hello_{mod_lower}.wav")
    with wave.open(path_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(rf_int.tobytes())
    print(f"Saved {path_wav} ({len(rf_int)} samples @ {fs} Hz, mono 16-bit)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Signal Generator for SIGINT Multi-Modulation Test Suite")
    parser.add_argument("--mod", type=str, default="ALL", help="Modulation type (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK, or ALL)")
    parser.add_argument("--snr", type=float, default=20.0, help="SNR in dB")
    args = parser.parse_args()

    mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK"] if args.mod.upper() == "ALL" else [args.mod.upper()]
    for m in mods:
        iq_samples = generate_modulation(mod_type=m, n_syms=800, sps=8, snr_db=args.snr)
        save_signal(m, iq_samples)
    print("\nSignal Generation Complete.")