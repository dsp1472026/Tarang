#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SPACE DOWNLINK SIMULATOR -- Image -> Modulated IQ / WAV
================================================================================
 Takes an image, packetises it like a spacecraft telemetry frame, modulates it
 with PSK / QAM / FSK, and writes it out as:
 *.iq interleaved int16 (I,Q) -> GNU Radio / SDR#
 *.c64 raw complex64 -> numpy / MATLAB
 *.wav real passband (mono) -> any audio tool
 *.wav IQ stereo (L=I, R=Q) -> SDR# / HDSDR
 A sidecar *.json holds all the link parameters (band, RF carrier, baud, ...).

 Dependencies: numpy, pillow (pip install numpy pillow)
================================================================================
"""

import os, io, sys, json, math, zlib, struct, wave, queue, threading, traceback
import numpy as np

try:
 from PIL import Image
 RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
except ImportError:
 Image = None
 RESAMPLE = None

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ==============================================================================
# CONFIG TABLES
# ==============================================================================
BANDS = {
 "HF (3 - 30 MHz)": 14.0e6,
 "VHF (30 - 300 MHz)": 145.0e6,
 "UHF (300 - 1000 MHz)": 437.0e6,
 "L-band (~1.7 GHz)": 1.70e9,
 "S-band (~2.2 GHz)": 2.20e9,
 "X-band (~8.4 GHz)": 8.40e9,
 "Ka-band (~26 GHz)": 26.0e9,
}

MODULATIONS = ["BPSK", "QPSK", "8PSK",
 "16QAM", "64QAM", "256QAM",
 "2FSK", "4FSK", "8FSK"]

OUT_FORMATS = [
 "IQ int16 interleaved (.iq)",
 "IQ complex64 raw (.c64)",
 "WAV real passband (.wav, mono)",
 "WAV IQ stereo L=I R=Q (.wav)",
]

SYNC_WORD = bytes([0x1A, 0xCF, 0xFC, 0x1D]) # CCSDS Attached Sync Marker
PREAMBLE = bytes([0xAA] * 4) # 1010... for timing recovery


# ==============================================================================
# DSP HELPERS
# ==============================================================================
def rrc_taps(beta, sps, span):
 """Root-Raised-Cosine filter taps (unit energy)."""
 N = span * sps
 t = np.arange(-N // 2, N // 2 + 1) / float(sps)
 if beta <= 1e-9:
 h = np.sinc(t)
 else:
 h = np.zeros_like(t, dtype=float)
 for i, ti in enumerate(t):
 if abs(ti) < 1e-9:
 h[i] = 1.0 - beta + 4.0 * beta / np.pi
 elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-9:
 h[i] = (beta / np.sqrt(2.0)) * (
 (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) -
 (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta)))
 else:
 num = (np.sin(np.pi * ti * (1.0 - beta)) +
 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
 den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
 h[i] = num / den
 return h / np.sqrt(np.sum(h ** 2))


def fft_convolve(x, h):
 """Overlap-add FFT convolution (complex x, real h)."""
 L = len(h)
 nfft = 1
 while nfft < max(2 * L, 4096):
 nfft <<= 1
 nfft = min(nfft, 1 << 17)
 H = np.fft.fft(h, nfft)
 step = nfft - L + 1
 n = len(x) + L - 1
 y = np.zeros(n, dtype=np.complex128)
 pos = 0
 for start in range(0, len(x), step):
 blk = x[start:start + step]
 yb = np.fft.ifft(np.fft.fft(blk, nfft) * H)
 # --- FIX: clip yb to whatever room is actually left in y ---
 room = n - pos
 take = nfft if nfft <= room else room
 y[pos:pos + take] += yb[:take]
 pos += step
 return y[:n]


def bits_to_symbols(bits, k):
 """Pack bits (MSB first) into symbol indices of k bits each."""
 n = len(bits) // k
 b = bits[:n * k].reshape(n, k).astype(np.int64)
 weights = (1 << np.arange(k - 1, -1, -1)).astype(np.int64)
 return b.dot(weights)


def gray_pam_levels(Ms):
 """Gray-coded PAM levels, centred on zero."""
 b = np.arange(Ms, dtype=np.int64)
 g = b ^ (b >> 1)
 return g.astype(np.float64) - (Ms - 1) / 2.0

def make_constellation(mod):
 """Return (constellation points, bits/symbol) for PSK/QAM."""
 mod = mod.upper()

 # --- PSK / QAM order lookup -------------------------------------------
 # BPSK = 2PSK, QPSK = 4PSK (standard aliases)
 ALIASES = {"BPSK": 2, "QPSK": 4, "8PSK": 8,
 "16QAM": 16, "64QAM": 64, "256QAM": 256}

 if mod in ALIASES:
 M = ALIASES[mod]
 is_psk = mod.endswith("PSK")
 elif mod.endswith("PSK"):
 try:
 M = int(mod[:-3])
 except ValueError:
 raise ValueError(f"Unrecognised PSK modulation: {mod}")
 is_psk = True
 elif mod.endswith("QAM"):
 try:
 M = int(mod[:-3])
 except ValueError:
 raise ValueError(f"Unrecognised QAM modulation: {mod}")
 is_psk = False
 else:
 raise ValueError("Unknown modulation " + mod)

 k = int(round(math.log2(M)))
 idx = np.arange(M)

 # --- PSK -------------------------------------------------------------
 if is_psk:
 g = idx ^ (idx >> 1) # gray code
 return np.exp(1j * 2.0 * np.pi * g / M), k

 # --- square QAM ------------------------------------------------------
 Ms = int(round(math.sqrt(M)))
 if Ms * Ms != M:
 raise ValueError(f"{mod}: only square QAM (16/64/256) supported")
 lv = gray_pam_levels(Ms)
 lv = lv / np.sqrt(2.0 * np.mean(lv ** 2)) # unit average power
 I = np.repeat(lv, Ms)
 Q = np.tile(lv, Ms)
 return I + 1j * Q, k


# ==============================================================================
# PACKETISATION
# ==============================================================================
def build_packet(payload: bytes) -> bytes:
 """
 Frame layout (all big endian):
 PREAMBLE 4B | SYNC 4B | LEN 4B | PAYLOAD | CRC32 4B
 """
 hdr = struct.pack(">I", len(payload))
 crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
 return PREAMBLE + SYNC_WORD + hdr + payload + crc


def image_to_payload(path, max_dim, compress):
 """
 Convert image to a payload.
 Returns: (payload_bytes, width, height, encoded_pil_image)
 Channels field: 3 = raw RGB, 0 = PNG-compressed stream.
 """
 img = Image.open(path).convert("RGB")
 if max_dim and (img.width > max_dim or img.height > max_dim):
 img.thumbnail((max_dim, max_dim), RESAMPLE)
 w, h = img.size

 if compress:
 buf = io.BytesIO()
 img.save(buf, format="PNG", optimize=True)
 body = buf.getvalue()
 return struct.pack(">HHB", w, h, 0) + body, w, h, img

 body = np.asarray(img, dtype=np.uint8).tobytes()
 return struct.pack(">HHB", w, h, 3) + body, w, h, img


# ==============================================================================
# FILE WRITERS
# ==============================================================================
def write_iq_int16(path, sig):
 i = np.clip(sig.real, -1.0, 1.0)
 q = np.clip(sig.imag, -1.0, 1.0)
 inter = np.empty(2 * len(sig), dtype=np.int16)
 inter[0::2] = (i * 32767.0).astype(np.int16)
 inter[1::2] = (q * 32767.0).astype(np.int16)
 inter.tofile(path)


def write_complex64(path, sig):
 sig.astype(np.complex64).tofile(path)


def _to_int16(x):
 m = np.max(np.abs(x))
 if m < 1e-12:
 m = 1.0
 return (x / m * 0.95 * 32767.0).astype("<i2")


def write_wav_real(path, sig, fs):
 data = _to_int16(sig.real)
 with wave.open(path, "wb") as w:
 w.setnchannels(1)
 w.setsampwidth(2)
 w.setframerate(int(round(fs)))
 w.writeframes(data.tobytes())


def write_wav_iq(path, sig, fs):
 i = sig.real
 q = sig.imag
 m = max(np.max(np.abs(i)), np.max(np.abs(q)))
 if m < 1e-12:
 m = 1.0
 inter = np.empty(2 * len(sig), dtype="<i2")
 inter[0::2] = (i / m * 0.95 * 32767.0).astype(np.int16)
 inter[1::2] = (q / m * 0.95 * 32767.0).astype(np.int16)
 with wave.open(path, "wb") as w:
 w.setnchannels(2)
 w.setsampwidth(2)
 w.setframerate(int(round(fs)))
 w.writeframes(inter.tobytes())


# ==============================================================================
# MAIN CONVERSION
# ==============================================================================
def convert(p, log):
 if Image is None:
 raise RuntimeError("Pillow is missing. Install with: pip install pillow")

 fs = float(p["fs"]) # sample rate [Hz]
 rs = float(p["rs"]) # symbol rate [baud]
 beta = float(p["beta"])
 span = int(p["span"])
 mod = p["mod"].upper()

 # ---- 1. image -> payload ---------------------------------------------
 log("• Reading image ...")
 payload, w, h, encoded_img = image_to_payload(
 p["image"], p["max_dim"], p["compress"])
 log(f" image : {w} x {h} px")
 log(f" payload : {len(payload):,} bytes")

 # ---- 2. framing -------------------------------------------------------
 packet = build_packet(payload)
 bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
 log(f" packet : {len(packet):,} bytes ({len(bits):,} bits)")

 # ---- 3. sample-rate / symbol-rate sanity ------------------------------
 sps_f = fs / rs
 sps = int(round(sps_f))
 if sps < 2:
 raise ValueError(
 f"sample_rate/symbol_rate = {sps_f:.3f}. "
 "Need at least 2 samples per symbol.")
 if abs(sps - sps_f) > 1e-6:
 log(f" NOTE: fs/rs = {sps_f:.4f} is not an integer.")
 log(f" Using sps={sps} -> effective symbol rate {fs/sps:,.1f} Bd")
 rs_eff = fs / sps
 log(f" samples/symbol : {sps}")

 # ---- 4. modulation ----------------------------------------------------
 if mod.endswith("FSK"):
 M = int(mod[:-3])
 k = int(round(math.log2(M)))
 sym_idx = bits_to_symbols(bits, k)

 delta_f = rs_eff / 2.0 # tone spacing
 freqs = (sym_idx - (M - 1) / 2.0) * delta_f # Hz per symbol
 max_tone = (M - 1) / 2.0 * delta_f + rs_eff / 2.0
 if max_tone > 0.45 * fs:
 raise ValueError(
 f"FSK tones exceed Nyquist ({max_tone:,.0f} Hz > {0.45*fs:,.0f} Hz). "
 "Increase the sample rate or lower the symbol rate.")
 inst = np.repeat(freqs, sps)
 phase = 2.0 * np.pi * np.cumsum(inst) / fs
 bb = np.exp(1j * phase)
 bw_est = (M - 1) * delta_f + 2.0 * rs_eff
 log(f" tone spacing : {delta_f:,.1f} Hz")
 log(f" peak deviation : {(M-1)/2*delta_f:,.1f} Hz")

 else:
 const, k = make_constellation(mod)
 nb = (len(bits) // k) * k
 sym_idx = bits_to_symbols(bits[:nb], k)
 symbols = const[sym_idx]

 up = np.zeros(len(symbols) * sps, dtype=np.complex128)
 up[::sps] = symbols

 log("• Pulse shaping (root-raised-cosine) ...")
 h = rrc_taps(beta, sps, span)
 bb = fft_convolve(up, h)
 bw_est = rs_eff * (1.0 + beta)
 log(f" bits/symbol : {k}")
 log(f" roll-off beta : {beta}")
 log(f" RRC span : {span} symbols ({len(h)} taps)")

 log(f" symbols : {len(sym_idx):,}")

 # ---- 5. normalise -----------------------------------------------------
 peak = np.max(np.abs(bb))
 if peak > 0:
 bb = bb / peak * 0.9

 # ---- 6. digital up-conversion to IF -----------------------------------
 if_frac = float(p["if_frac"])
 f_if = if_frac * fs
 if abs(f_if) > 1e-9:
 log(f"• Up-converting to IF = {f_if:,.1f} Hz "
 f"({if_frac:.3f} x fs)")
 n = len(bb)
 ph = 2.0 * np.pi * f_if * np.arange(n) / fs
 out = bb * np.exp(1j * ph)
 else:
 out = bb

 # ---- 7. write ---------------------------------------------------------
 out_path = p["output"]
 fmt = p["format"]

 log("• Writing file ...")
 if fmt.startswith("IQ int16"):
 write_iq_int16(out_path, out)
 elif fmt.startswith("IQ complex64"):
 write_complex64(out_path, out)
 elif fmt.startswith("WAV real"):
 write_wav_real(out_path, out, fs)
 else:
 write_wav_iq(out_path, out, fs)

 size = os.path.getsize(out_path)
 log(f" saved : {out_path}")
 log(f" file size : {size/1e6:,.2f} MB")

 # ---- 7b. save the exact image that was encoded -----------------------
 preview_path = os.path.splitext(out_path)[0] + "_encoded.png"
 encoded_img.save(preview_path, format="PNG", optimize=True)
 log(f" encoded image : {preview_path}")
 log(f" ({w} x {h} px, "
 f"{'PNG-compressed payload' if p['compress'] else 'raw RGB payload'})")

 # ---- 8. metadata sidecar ---------------------------------------------
 meta = {
 "created_by": "Space Downlink Simulator",
 "source_image": os.path.basename(p["image"]),
 "encoded_image_file": os.path.basename(preview_path),
 "image_size_px": [w, h],
 "payload_bytes": len(payload),
 "packet_bytes": len(packet),
 "modulation": mod,
 "bits_per_symbol": int(k),
 "sample_rate_hz": fs,
 "symbol_rate_bd": rs_eff,
 "samples_per_symbol": sps,
 "rrc_beta": beta if not mod.endswith("FSK") else None,
 "band": p["band"],
 "rf_carrier_hz": float(p["rf"]),
 "if_offset_hz": f_if,
 "if_offset_fraction_of_fs": if_frac,
 "occupied_bandwidth_hz": bw_est,
 "output_format": fmt,
 "output_samples": int(len(out)),
 "duration_s": len(out) / fs,
 "frame": {
 "preamble_bytes": len(PREAMBLE),
 "sync_word": SYNC_WORD.hex(),
 "length_field": "uint32 big endian",
 "crc": "CRC-32 big endian",
 "bit_order": "MSB first",
 },
 }
 with open(out_path + ".json", "w") as f:
 json.dump(meta, f, indent=2)
 log(f" metadata : {out_path}.json")

 # ---- 9. summary -------------------------------------------------------
 log("")
 log("──────────── LINK SUMMARY ────────────")
 log(f" Band : {p['band']}")
 log(f" RF carrier : {float(p['rf'])/1e6:,.4f} MHz")
 log(f" Modulation : {mod}")
 log(f" Sample rate : {fs/1e6:,.4f} MSps")
 log(f" Symbol rate : {rs_eff/1e3:,.3f} kBd")
 log(f" Occ. bandwidth : {bw_est/1e3:,.2f} kHz")
 log(f" Duration : {len(out)/fs:,.3f} s")
 log(f" Samples : {len(out):,}")
 log("──────────────────────────────────────")
 log("Done ✔")


# ==============================================================================
# GUI
# ==============================================================================
class App(tk.Tk):

 def __init__(self):
 super().__init__()
 self.title("Space Downlink Simulator • Image → IQ / WAV")
 self.geometry("920x800")
 self.minsize(820, 640)

 self.q = queue.Queue()
 self.busy = False

 self.var_img = tk.StringVar()
 self.var_band = tk.StringVar(value="UHF (300 - 1000 MHz)")
 self.var_rf = tk.StringVar(value="437000000")
 self.var_mod = tk.StringVar(value="QPSK")
 self.var_fs = tk.StringVar(value="1000000")
 self.var_rs = tk.StringVar(value="250000")
 self.var_beta = tk.StringVar(value="0.35")
 self.var_span = tk.StringVar(value="10")
 self.var_if = tk.StringVar(value="0.25")
 self.var_fmt = tk.StringVar(value=OUT_FORMATS[0])
 self.var_dim = tk.StringVar(value="256")
 self.var_comp = tk.BooleanVar(value=False)

 self._build()
 self.after(100, self._poll)

 # ------------------------------------------------------------------ UI --
 def _build(self):
 P = dict(padx=8, pady=4)

 # ---- 1. image ----------------------------------------------------
 f1 = ttk.LabelFrame(self, text="1 · Source image")
 f1.pack(fill="x", **P)

 ttk.Entry(f1, textvariable=self.var_img).grid(
 row=0, column=0, sticky="ew", padx=6, pady=6)
 ttk.Button(f1, text="Browse…", command=self._pick_image).grid(
 row=0, column=1, padx=6, pady=6)

 ttk.Label(f1, text="Max dimension (px):").grid(row=1, column=0,
 sticky="w", padx=6)
 ttk.Entry(f1, textvariable=self.var_dim, width=10).grid(
 row=1, column=1, sticky="w", padx=6, pady=(0, 6))
 ttk.Checkbutton(f1, text="Compress payload (PNG)",
 variable=self.var_comp).grid(
 row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))
 f1.columnconfigure(0, weight=1)

 # ---- 2. link -----------------------------------------------------
 f2 = ttk.LabelFrame(self, text="2 · Link parameters")
 f2.pack(fill="x", **P)

 ttk.Label(f2, text="Band").grid(row=0, column=0, sticky="w", padx=6, pady=4)
 self.cmb_band = ttk.Combobox(f2, textvariable=self.var_band,
 values=list(BANDS.keys()),
 state="readonly", width=24)
 self.cmb_band.grid(row=0, column=1, sticky="w", padx=6)
 self.cmb_band.bind("<<ComboboxSelected>>", self._on_band)

 ttk.Label(f2, text="RF carrier (Hz)").grid(row=0, column=2,
 sticky="w", padx=6)
 ttk.Entry(f2, textvariable=self.var_rf, width=18).grid(
 row=0, column=3, sticky="w", padx=6)

 ttk.Label(f2, text="Modulation").grid(row=1, column=0,
 sticky="w", padx=6, pady=4)
 ttk.Combobox(f2, textvariable=self.var_mod, values=MODULATIONS,
 state="readonly", width=22).grid(
 row=1, column=1, sticky="w", padx=6)

 ttk.Label(f2, text="IF offset (× fs)").grid(row=1, column=2,
 sticky="w", padx=6)
 ttk.Entry(f2, textvariable=self.var_if, width=18).grid(
 row=1, column=3, sticky="w", padx=6)

 ttk.Label(f2, text="Sample rate (Hz)").grid(row=2, column=0,
 sticky="w", padx=6, pady=4)
 ttk.Entry(f2, textvariable=self.var_fs, width=22).grid(
 row=2, column=1, sticky="w", padx=6)

 ttk.Label(f2, text="Symbol rate (Bd)").grid(row=2, column=2,
 sticky="w", padx=6)
 ttk.Entry(f2, textvariable=self.var_rs, width=18).grid(
 row=2, column=3, sticky="w", padx=6)

 ttk.Label(f2, text="RRC roll-off β").grid(row=3, column=0,
 sticky="w", padx=6, pady=4)
 ttk.Entry(f2, textvariable=self.var_beta, width=22).grid(
 row=3, column=1, sticky="w", padx=6)

 ttk.Label(f2, text="RRC span (symbols)").grid(row=3, column=2,
 sticky="w", padx=6)
 ttk.Entry(f2, textvariable=self.var_span, width=18).grid(
 row=3, column=3, sticky="w", padx=6)

 ttk.Label(f2, text="Presets:").grid(row=4, column=0,
 sticky="w", padx=6, pady=(6, 4))
 pf = ttk.Frame(f2)
 pf.grid(row=4, column=1, columnspan=3, sticky="w")
 for name, fs, rs, m in [
 ("CubeSat UHF 9k6", "48000", "9600", "2FSK"),
 ("CubeSat UHF 19k2", "96000", "19200", "2FSK"),
 ("S-band QPSK 1M", "4000000", "1000000", "QPSK"),
 ("X-band 16QAM 10M", "40000000", "10000000", "16QAM"),
 ("VHF BPSK 1k2", "48000", "1200", "BPSK"),
 ]:
 ttk.Button(pf, text=name, width=17,
 command=lambda a=fs, b=rs, c=m: self._preset(a, b, c)
 ).pack(side="left", padx=2)

 # ---- 3. output ---------------------------------------------------
 f3 = ttk.LabelFrame(self, text="3 · Output")
 f3.pack(fill="x", **P)
 ttk.Label(f3, text="Format").grid(row=0, column=0, sticky="w",
 padx=6, pady=6)
 ttk.Combobox(f3, textvariable=self.var_fmt, values=OUT_FORMATS,
 state="readonly", width=40).grid(
 row=0, column=1, sticky="w", padx=6, pady=6)
 f3.columnconfigure(1, weight=1)

 # ---- 4. generate -------------------------------------------------
 f4 = ttk.Frame(self)
 f4.pack(fill="x", **P)
 self.btn = ttk.Button(f4, text="▶ GENERATE SIGNAL FILE",
 command=self._go)
 self.btn.pack(side="left", padx=6)

 self.pb = ttk.Progressbar(f4, mode="indeterminate", length=260)
 self.pb.pack(side="left", padx=12)

 ttk.Button(f4, text="Clear log",
 command=lambda: self.txt.delete("1.0", tk.END)).pack(
 side="right", padx=6)

 # ---- 5. log ------------------------------------------------------
 f5 = ttk.LabelFrame(self, text="Log")
 f5.pack(fill="both", expand=True, **P)
 self.txt = tk.Text(f5, height=18, wrap="word",
 bg="#101418", fg="#9fe8b0",
 insertbackground="#9fe8b0",
 font=("Consolas", 9))
 sb = ttk.Scrollbar(f5, command=self.txt.yview)
 self.txt.configure(yscrollcommand=sb.set)
 sb.pack(side="right", fill="y")
 self.txt.pack(side="left", fill="both", expand=True)

 self.log("Space Downlink Simulator ready.")
 self.log("Pick an image, set the link parameters, hit GENERATE.\n")

 # -------------------------------------------------------------- helpers --
 def log(self, msg):
 self.q.put(msg)

 def _poll(self):
 try:
 while True:
 self.txt.insert(tk.END, self.q.get_nowait() + "\n")
 self.txt.see(tk.END)
 except queue.Empty:
 pass
 self.after(100, self._poll)

 def _on_band(self, *_):
 self.var_rf.set(str(int(BANDS[self.var_band.get()])))

 def _preset(self, fs, rs, mod):
 self.var_fs.set(fs)
 self.var_rs.set(rs)
 self.var_mod.set(mod)

 def _pick_image(self):
 p = filedialog.askopenfilename(
 title="Select an image",
 filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif"),
 ("All files", "*.*")])
 if p:
 self.var_img.set(p)

 # ---------------------------------------------------------------- run ----
 def _go(self):
 if self.busy:
 return

 img = self.var_img.get().strip()
 if not img or not os.path.isfile(img):
 messagebox.showerror("Error", "Please choose a valid image file.")
 return

 fmt = self.var_fmt.get()
 if fmt.startswith("IQ int16"):
 ext, ftypes = ".iq", [("IQ files", "*.iq")]
 elif fmt.startswith("IQ complex64"):
 ext, ftypes = ".c64", [("Complex64 files", "*.c64")]
 else:
 ext, ftypes = ".wav", [("WAV files", "*.wav")]

 base = os.path.splitext(os.path.basename(img))[0] + "_" + \
 self.var_mod.get().lower()

 out = filedialog.asksaveasfilename(
 title="Save modulated signal as…",
 defaultextension=ext,
 initialfile=base + ext,
 filetypes=ftypes + [("All files", "*.*")])
 if not out:
 return

 try:
 params = {
 "image": img,
 "output": out,
 "band": self.var_band.get(),
 "rf": float(self.var_rf.get()),
 "mod": self.var_mod.get(),
 "fs": float(self.var_fs.get()),
 "rs": float(self.var_rs.get()),
 "beta": float(self.var_beta.get()),
 "span": int(self.var_span.get()),
 "if_frac": float(self.var_if.get()),
 "fmt": fmt,
 "format": fmt,
 "max_dim": int(self.var_dim.get()) if self.var_dim.get().strip() else 0,
 "compress": self.var_comp.get(),
 }
 except ValueError as e:
 messagebox.showerror("Bad parameter", f"Check your numeric fields:\n{e}")
 return

 self.busy = True
 self.btn.config(state="disabled")
 self.pb.start(12)
 self.log("\n" + "=" * 58)
 self.log(f"START {os.path.basename(img)} → {os.path.basename(out)}")
 self.log("=" * 58)

 t = threading.Thread(target=self._worker, args=(params,), daemon=True)
 t.start()

 def _worker(self, params):
 try:
 convert(params, self.log)
 except Exception as e:
 self.log("\n!! ERROR: " + str(e))
 self.log(traceback.format_exc())
 finally:
 self.q.put("__DONE__")

 # patch _poll to stop progress bar
 def _poll(self):
 try:
 while True:
 msg = self.q.get_nowait()
 if msg == "__DONE__":
 self.busy = False
 self.btn.config(state="normal")
 self.pb.stop()
 continue
 self.txt.insert(tk.END, msg + "\n")
 self.txt.see(tk.END)
 except queue.Empty:
 pass
 self.after(100, self._poll)


# ==============================================================================
if __name__ == "__main__":
 if Image is None:
 print("Please install Pillow: pip install pillow")
 App().mainloop()