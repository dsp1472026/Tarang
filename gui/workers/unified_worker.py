"""
GUI worker for the blind receiver (Phases 1-2 + unified classifier).

Layer 1  ingestion/capture.py: .iq (interleaved) and .wav (stereo IQ, or real
         passband -> analytic signal -> carrier estimate -> mix to 0 Hz -> decimate)
Layer 2  FFT SOI detection (unchanged, for the Detection tab)
Layer 4  modulation/unified_classifier.py: FSK vs PSK vs QAM, then the order
Layer 5+6 PSK: demodulation/psk_receiver.py (RRC MF, Gardner, M-th power + Costas/DD PLL,
         sync-word ambiguity resolution); QAM: demodulation/qam_receiver.py (4th-power + DD
         carrier, label-mapping hypotheses); FSK: demodulation/fsk_receiver.py.

Results are returned in the same dict layout as Phase2PipelineWorker so the
existing tabs render them unchanged.
"""

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np

from core.evidence import Evidence, Parameter
from core.signal_buffer import SignalBuffer
from demodulation.demodulator import DemodulationResult
from demodulation.ff_receiver import receive_feedforward
from demodulation.fsk_receiver import FSKReceiverConfig, receive_fsk
from demodulation.psk import best_model, best_text_region
from demodulation.psk_receiver import PSKReceiverConfig, receive_psk
from demodulation.qam_receiver import QAMReceiverConfig, receive_qam
from detection.channelizer import extract, find_signals
from detection.fft_detector import FFTDetector
from gui.workers.base_worker import BaseWorker
from ingestion.capture import Capture, load_capture, reduce_oversampling
from ingestion.format_detector import FormatDetector
from ingestion.iq_formats import normalize_datatype
from postdemod.analyzer import analyze as analyze_postdemod, hypotheses_from_receiver
from postdemod.image_payload import find_image
from modulation.unified_classifier import classify_capture, fsk_fit_is_single_carrier, reclassify_as_psk
from synchronization.agc import AGCResult
from synchronization.costas_loop import CostasResult
from synchronization.timing_recovery import TimingRecoveryResult

logger = logging.getLogger(__name__)


def _load(config: Dict[str, Any]) -> Capture:
    """.wav: rate from the header. .iq: every SDR sample type (CU8, CS8, CS16, CF32, CF64) through
    ingestion/iq_formats.py; the sample rate is the one computed from the user's recording duration."""
    path = config.get("file_path", "")
    fs = config.get("sample_rate") or None
    if path.lower().endswith(".wav"):
        return reduce_oversampling(load_capture(path))
    dtype = config.get("dtype")
    if normalize_datatype(dtype) is None:                  # 'auto': sidecar, else blind byte-structure detection
        det = FormatDetector.auto_detect_iq_format(path)
        dtype = det.format_name + ("_be" if det.endianness == "big" else "")
        offset = config.get("byte_offset") or det.header_offset
    else:
        offset = config.get("byte_offset") or 0
    # an explicit _le/_be in the type name (detector result) wins over the GUI's endianness box
    endian = None if str(dtype).lower().endswith(("_le", "_be")) else config.get("endianness")
    return reduce_oversampling(load_capture(path, dtype=dtype, sample_rate=fs, byte_offset=int(offset or 0),
                                            endianness=endian, iq_order=str(config.get("iq_order", "IQ"))))


def _sample_rate_param(config: Dict[str, Any], cap: Capture) -> Parameter:
    """Sample rate with its provenance: exact (duration / typed / WAV header / sidecar) = KNOWN,
    blind hypothesis = ASSUMED, none = UNKNOWN (results then stay in normalised units)."""
    fs = cap.sample_rate
    src = config.get("sample_rate_source")
    notes = "; ".join(cap.notes)
    if not fs:
        return Parameter("Sample rate", None, "Hz", "UNKNOWN", "not provided; results in normalised units")
    if src == "blind" and config.get("sample_rate"):
        sps = config.get("blind_sps")
        return Parameter("Sample rate", fs, "Hz", "ASSUMED",
                         f"blind hypothesis: measured {sps:.4f} samples/symbol x a standard symbol rate "
                         f"(the absolute rate is not measurable from the samples); {notes}" if sps else
                         f"blind hypothesis; {notes}", 0.3)
    method = {"duration": "samples / user-entered recording duration", "manual": "entered by the user",
              "header": "WAV header"}.get(src, "metadata")
    if not config.get("sample_rate"):
        method = "sidecar metadata" if not str(cap.path).lower().endswith(".wav") else "WAV header"
    return Parameter("Sample rate", fs, "Hz", "KNOWN", f"{method}; {notes}")


LINEAR_ORDERS = ("BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "64QAM", "256QAM")
# Common symbol rates (Bd). Only a hypothesis prior for the fallback below: a rate is adopted
# only when the decode then verifies (CRC-valid frame or decisive text covering the burst).
STANDARD_BAUD = (50, 75, 100, 110, 150, 200, 300, 600, 1000, 1200, 2000, 2400, 3200, 3600, 4800, 5000, 7200,
                 9600, 10000, 12000, 14400, 16000, 19200, 20000, 25000, 28800, 32000, 38400, 48000, 50000,
                 57600, 64000, 76800, 96000, 100000, 115200, 128000, 192000, 250000, 256000, 500000, 1000000)
BAUD_SNAP_TOL = 0.07       # relative distance to a standard rate that makes it a hypothesis
ORDER_CHECK_MAX_SYMBOLS = 600   # longer bursts classify reliably; the fallback is for short ones
MIN_VERIFIED_TEXT_BYTES = 8
MIN_VERIFIED_TEXT_SCORE = -4.3   # English / numeric readings score -2.6 .. -4.0; chance readings among
                                 # the fallback's ~27 hypotheses scored -4.7 .. -6.6 ('"mt22=e3wp fL')
MAX_RATE_HYPOTHESES = 4    # most plausible symbol-rate hypotheses tried (x 7 orders each)
MIN_TEXT_COVERAGE = 0.5
IMAGE_RETRY_MAX_SYMBOLS = 50000   # image CRC retry on the receiver's own front end (cost: one receiver pass)    # an alternative's text must cover this fraction of its demodulated bits


def _linear_receive(samples, fs, cls, modulation: str, front=None, sps=None):
    """PSK or QAM receiver for one forced modulation: on the classifier's front end, or (sps
    given) on the receiver's own front end at that symbol rate, which tests its own coarse
    carrier-offset hypotheses (fixing the classifier's CFO there lost a 256QAM decode at the
    right rate: EVM 5.1 % instead of 2.0 %)."""
    if sps is not None:
        if modulation.endswith("QAM"):
            return receive_qam(samples, fs, QAMReceiverConfig(modulation=modulation, sps=sps))
        return receive_psk(samples, fs, PSKReceiverConfig(modulation=modulation, sps=sps))
    fe = front if front is not None else cls.front
    if modulation.endswith("QAM"):
        return receive_qam(samples, fs, QAMReceiverConfig(modulation=modulation), front=fe,
                           classification=cls.linear)
    cfg = copy.copy(cls.front_config) if cls.front_config is not None else PSKReceiverConfig()
    cfg.modulation = modulation
    return receive_psk(samples, fs, cfg, front=fe)


NUMERIC_CHARS = set(b"0123456789 .,-+:/()")
MIN_NUMERIC_PURITY = 0.9   # a reading scored as a numeric message must be this pure (printable bytes)


def _region_score(payload: bytes) -> float:
    """Text score of the payload's text region (binary header / trailer trimmed, as the decoders
    score it): the reported payload keeps the burst-edge junk, which pulled a correct 256QAM digit
    string (-2.6 nats/char as text) below MIN_VERIFIED_TEXT_SCORE. A reading the numeric model
    explains best must also be >= 90 % digits / separators among its printable bytes: a wrong
    hypothesis on a digit payload recovers half of each byte and reads as digit-like junk
    ('336f73263#1332f', 80 %), which the numeric model alone rated as text."""
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    score, off = best_text_region(bits)
    region = np.packbits(bits[off:off + (len(bits) - off) // 8 * 8]).tobytes()
    tail = [i for i, b in enumerate(region[-8:]) if not 32 <= b < 127]      # trailer, as the decoders cut it
    if tail:
        region = region[:len(region) - len(region[-8:]) + tail[0]]
    printable = bytes(b for b in region if 32 <= b < 127)
    if printable and best_model(printable)[1] == "numeric":
        # purity between the first and last digit (burst-edge characters such as '[' or 'L' next to
        # a genuine digit string are not part of it; junk INSIDE the string is)
        dig = [i for i, b in enumerate(printable) if 48 <= b <= 57]
        span = printable[dig[0]:dig[-1] + 1] if dig else printable
        if sum(b in NUMERIC_CHARS for b in span) < MIN_NUMERIC_PURITY * len(span):
            return -np.inf
    return score


def _image_of(r, family: Optional[str] = None):
    """Image / file frame in a receiver result (postdemod/image_payload.py), searched once and cached on
    the result: a CRC-valid (or structurally confirmed) file frame verifies the modulation / rate
    hypothesis exactly like a CRC-valid text frame does."""
    if not hasattr(r, "_image_frame"):
        fam = family or ("QAM" if r.modulation.endswith("QAM") and r.modulation != "4QAM" else "PSK")
        try:
            r._image_frame = find_image(fam, r)
        except Exception as e:                           # noqa: BLE001 - a hypothesis that cannot be searched
            logger.info("image frame search failed: %s", e)
            r._image_frame = None
    return r._image_frame


def _decode_quality(r, strict: bool = False):
    """(rank, score): 2 = CRC-valid frame, 1 = decisive text / payload match, 0 = nothing.
    strict (alternative hypotheses): the text must be >= MIN_VERIFIED_TEXT_BYTES long and cover
    most of the demodulated bits, since every extra hypothesis is another chance for a
    fragment of garbage to read as text."""
    d = r.decode
    if d.frame is not None and d.frame.crc_ok:
        return 2, 0.0
    img = _image_of(r)
    if img is not None:
        return (2 if img.crc_ok else 1), 0.0
    if d.method in ("sync-word", "differential"):
        # 96-bit sync word found (<= 12 errors): rate / order are right even when the CRC fails
        # (e.g. an FEC-coded frame, decoded later by the L8-L11 search)
        return 1, 0.0
    if d.method in ("text-heuristic", "known-payload"):
        n = len(d.payload)
        score = _region_score(d.payload)
        if strict and (n < MIN_VERIFIED_TEXT_BYTES or 8 * n < MIN_TEXT_COVERAGE * len(r.hard_bits)
                       or score < MIN_VERIFIED_TEXT_SCORE):
            return 0, -np.inf
        return 1, score
    return 0, -np.inf


def _fsk_verified(r) -> bool:
    """FSK result with a CRC-valid frame, or a decisive text reading covering most of the burst."""
    p = r.primary
    if p.crc_ok:
        return True
    n = len(p.payload)
    return (p.resolution == "text-heuristic" and n >= MIN_VERIFIED_TEXT_BYTES
            and 8 * n >= MIN_TEXT_COVERAGE * len(r.hard_bits) and _region_score(p.payload) >= MIN_VERIFIED_TEXT_SCORE)


# 99 %-power bandwidth of an RRC signal relative to its symbol rate (roll-off 0.35: 0.94..1.13 on the
# project generator at 1200 Bd, 1.09..1.41 on pangrams/p..py-style signals at 4800 Bd, whose truncated
# 'same'-mode RRC leaks more; the occupied_band estimate is also quantised). Unlike the -3 dB width,
# it is not pulled onto the narrow data lines of a short repetitive (text) payload: on framed
# 256QAM bursts of 20..400 characters the -3 dB / cyclic estimates ranged 300..1700 Bd for 1200 Bd.
BW99_PER_RS = (0.8, 1.5)
BW99_PER_RS_TYPICAL = 1.05


def _rate_hypotheses(fs, sps0, bw99_norm=None):
    """Samples/symbol hypotheses, most plausible first:
    1. (sample rate known) standard symbol rates consistent with the 99 %-power bandwidth,
       nearest to bw99 / 1.05 first; without a sample rate, bw99 / 1.05 itself;
    2. integer multiples / fractions of the front-end rate (short dense-QAM bursts lock at half
       or a third of it), each snapped to the nearest standard rate when the sample rate is known;
    3. the unsnapped multiples / fractions."""
    bw, std, raw = [], [], []
    if bw99_norm:
        if fs:
            lo, hi = bw99_norm * fs / BW99_PER_RS[1], bw99_norm * fs / BW99_PER_RS[0]
            centre = bw99_norm * fs / BW99_PER_RS_TYPICAL
            for b in sorted((v for v in STANDARD_BAUD if lo <= v <= hi), key=lambda v: abs(np.log(v / centre))):
                bw.append((fs / b, f"{b} Bd (standard rate consistent with the 99 % bandwidth "
                                   f"{bw99_norm * fs:.0f} Hz, ASSUMED)"))
        else:
            bw.append((BW99_PER_RS_TYPICAL / bw99_norm, "99 % bandwidth / 1.05"))
    for k in (1, 2, 3, 0.5):
        s = sps0 / k
        if s < 2.2:
            continue
        if k != 1:
            raw.append((s, f"front-end rate x {k:g}"))
        if fs:
            rate = fs / s
            b = min(STANDARD_BAUD, key=lambda v: abs(rate / v - 1))
            if abs(rate / b - 1) <= BAUD_SNAP_TOL:
                std.append((fs / b, f"{b} Bd (standard rate nearest to {rate:.1f} Bd, ASSUMED)"))
    out = []
    for s, label in bw + std + raw:
        if s >= 2.2 and abs(s / sps0 - 1) > 0.002 and all(abs(s / o[0] - 1) > 0.01 for o in out):
            out.append((s, label))
    return out


def _linear_name(modulation: str) -> str:
    """Receiver label -> constellation name (4QAM is QPSK). Exact match: str.replace("4QAM", "QPSK")
    also rewrote 64QAM into '6QPSK'."""
    return "QPSK" if modulation == "4QAM" else modulation


def _exact_rate_hypotheses(fs, sps0):
    """Exact-rate neighbours of an estimated samples/symbol: the nearest standard symbol rate (sample
    rate known) and the nearest integer samples/symbol, each only if within BAUD_SNAP_TOL and not
    already equal to the estimate."""
    out = []
    if fs:
        b = min(STANDARD_BAUD, key=lambda v: abs(fs / sps0 / v - 1))
        if abs(fs / sps0 / b - 1) <= BAUD_SNAP_TOL:
            out.append((fs / b, f"{b} Bd (nearest standard rate)"))
    k = round(sps0)
    if k >= 2 and abs(sps0 / k - 1) <= BAUD_SNAP_TOL:
        out.append((float(k), f"{k} samples/symbol (nearest integer)"))
    uniq = []
    for s, label in out:
        if abs(s / sps0 - 1) > 1e-7 and all(abs(s / u[0] - 1) > 1e-9 for u in uniq):
            uniq.append((s, label))
    return uniq


def verify_linear_order(samples, fs, cls, r, progress=None):
    """
    Decode-validated order / rate decision. On short bursts (a numeric message is a few dozen
    symbols) the classifier can pick a neighbouring order (8PSK -> 64QAM, 16PSK -> 16QAM) or,
    for dense QAM, lock at an integer fraction of the symbol rate. When the classified
    hypothesis yields neither a CRC-valid frame nor a decisive text reading:
      1. the other PSK/QAM orders on the same front end (most probable first), then
      2. up to MAX_RATE_HYPOTHESES symbol-rate hypotheses (standard rates consistent with the
         99 % bandwidth; x2, x3, /2 of the estimate and their nearest standard rates), each with
         all seven orders (a fresh classification of its symbols only sets the order).
    A hypothesis is adopted only if it verifies: a CRC-valid frame ends the search; otherwise
    the best text reading that covers the burst. The plan is fixed (at most 6 orders + 4 rates
    x 7 orders), not time-limited, so the outcome never depends on machine load; it
    runs only for bursts shorter than ORDER_CHECK_MAX_SYMBOLS symbols.
    Returns (result, evidence or None).
    """
    from demodulation.psk_receiver import psk_front_end
    from modulation.linear_classifier import classify_linear

    n_sym = len(cls.front.idx) if cls.front is not None else 0
    if _decode_quality(r)[0] > 0 or n_sym >= ORDER_CHECK_MAX_SYMBOLS:
        return r, None
    post = dict(getattr(cls.linear, "posterior", None) or {})
    first = _linear_name(r.modulation)
    orders0 = sorted((m for m in LINEAR_ORDERS if m not in (first, cls.modulation)), key=lambda m: -post.get(m, 0.0))
    bw99 = None
    try:
        from dsp.conditioning import normalize_power
        from estimation.fsk_estimator import occupied_band
        bw99 = float(occupied_band(normalize_power(np.asarray(samples, dtype=np.complex128))[0]).bw99)
    except Exception as e:                           # noqa: BLE001
        logger.info("99 %% bandwidth unavailable: %s", e)
    rates = (_rate_hypotheses(fs, float(cls.front.sps), bw99) if cls.front is not None else [])[:MAX_RATE_HYPOTHESES]
    best, best_q, best_how, n_tried = None, (0, -np.inf), "", 0
    total = len(orders0) + len(rates) * len(LINEAR_ORDERS)
    step = 0

    def attempt(m, front, sps, how):
        nonlocal best, best_q, best_how, n_tried, step
        if progress:
            progress(min(step, total - 1), total, m if sps is None else f"{m} at {sps:.3f} sps")
        step += 1
        try:
            ra = _linear_receive(samples, fs, cls, m, front, sps)
        except Exception as e:                       # a hypothesis that cannot even be demodulated
            logger.info("hypothesis %s (%s) failed: %s", m, how, e)
            return
        n_tried += 1
        q = _decode_quality(ra, strict=True)
        if q > best_q:
            best, best_q, best_how = ra, q, how

    # 1. other orders, same front end
    for m in orders0:
        attempt(m, None, None, "classifier front end")
        if best_q[0] == 2:
            break
    # 2. symbol-rate hypotheses (only when no order verified at the estimated rate)
    if best_q[0] == 0:
        for sps, label in rates:
            try:
                fe = psk_front_end(samples, fs, PSKReceiverConfig(sps=sps))
                lin = classify_linear(fe.s_core)
                pr = dict(lin.posterior)
                # every order: on these short bursts the fresh classification is unreliable (a
                # 256QAM burst at its true rate was classified QPSK at 97 %), so it only orders the
                # search; cheap orders first after the classified one (PSK ~1 s, 256QAM ~60 s)
                orders = [lin.modulation] + sorted((m for m in LINEAR_ORDERS if m != lin.modulation),
                                                   key=lambda m: (-pr.get(m, 0.0), LINEAR_ORDERS.index(m)))
            except Exception as e:
                logger.info("rate hypothesis %.3f sps failed: %s", sps, e)
                continue
            for m in orders:
                attempt(m, None, sps, f"{sps:.3f} samples/symbol: {label}")
                if best_q[0] > 0:
                    break
            if best_q[0] > 0:
                break
    if best is None or best_q[0] == 0:
        return r, Evidence(f"Classified {cls.modulation} gave no CRC-valid frame or decisive text; {n_tried} other "
                           f"order / symbol-rate hypotheses did not verify either", "ORDER_CHECK", 0.0)
    if best_q[0] == 2:
        what = "CRC-valid frame"
    elif best.decode.method in ("sync-word", "differential"):
        what = f"sync word ({best.decode.sync_bit_errors}/96 bit errors; frame CRC not valid)"
    else:
        what = f"decisive text reading ({best_q[1]:.2f} nats/char)"
    return best, Evidence(f"Classified {cls.modulation} gave no CRC-valid frame or decisive text; {best.modulation} "
                          f"({best_how}) verified by a {what} and was adopted ({n_tried} hypotheses tried)",
                          "ORDER_CHECK", 1.0 if best_q[0] == 2 else 0.5)

class _Cancelled(Exception):
    """Raised at a cancellation check; BaseWorker.run turns it into the `cancelled` signal."""


class UnifiedPipelineWorker(BaseWorker):
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent=parent)
        self.config = config
        self._lo, self._hi = 0.0, 100.0                  # progress range of the signal being analysed
        self._tag = ""

    def _emit(self, pct, msg):
        """Progress within the current signal's share of the bar (several signals per capture)."""
        self.progress.emit(int(self._lo + (self._hi - self._lo) * pct / 100.0), self._tag + msg)

    def _check_cancel(self):
        """Stop between layers when Cancel was pressed (BaseWorker.run then emits `cancelled`)."""
        if self.is_cancelled:
            raise _Cancelled("cancelled by user")

    def _postdemod_progress(self, p, m):
        self._check_cancel()
        self._emit(70 + int(0.28 * p), m)

    def _find_image(self, family: str, r, pct: int):
        """Image / file frame (ASM | LEN32 | PAYLOAD | CRC32, uncoded or K=7 FEC) in the receiver's bits."""
        try:
            return find_image(family, r,
                              progress=lambda h: self._emit(pct, f"L11: searching for an image / file frame "
                                                                         f"({h})..."),
                              check_cancel=self._check_cancel)
        except _Cancelled:
            raise
        except Exception as e:                            # noqa: BLE001 - never lose the demodulated result
            logger.exception("image frame search failed: %s", e)
            return None

    def _fsk_tone_fallback(self, cap: Capture, fsk_r, image, mod_param):
        """Blind FSK model wrong (a 2FSK h=0.5 burst at 5 samples/symbol was fitted as 4FSK h=0.17):
        tone correlator over standard (M, h) hypotheses at the estimated and exact rates; returns a
        CRC-valid image frame (or the first image if none was found before), else None."""
        from demodulation.fsk_tones import hypotheses
        from estimation.fsk_estimator import occupied_band
        fs = cap.sample_rate
        sps0 = float(fsk_r.model.sps)
        center = occupied_band(np.asarray(cap.samples[:1 << 20], dtype=np.complex128)).center
        rates = [(sps0, "the estimated rate")] + _exact_rate_hypotheses(fs, sps0)
        for sps, rate_label in rates:
            for name, hr in hypotheses(cap.samples, sps, center, fsk_r.model.order, fsk_r.model.h):
                self._check_cancel()
                self._emit(67, f"L11: FSK {name} at {rate_label}...")
                img = find_image("FSK", hr)
                if img is not None and (img.crc_ok or image is None):
                    mod_param.evidence.append(Evidence(
                        f"Blind FSK tone model gave no valid frame; {name} at {rate_label} recovered the file "
                        f"frame ({'CRC-valid' if img.crc_ok else 'CRC not verified'})", "FSK_TONE_FALLBACK",
                        1.0 if img.crc_ok else 0.5))
                    if img.crc_ok:
                        # report the model that decoded the frame, not the blind fit's
                        fsk_r.variant = f"{hr.M}FSK (h = {hr.h:.2f}, tone correlator)"
                        return img
                    image = img
        return image

    def work(self) -> Dict[str, Any]:
        self._emit(2, "Layer 1: Reading capture (.iq / .wav)...")
        cap = _load(self.config)
        whole = SignalBuffer(samples=cap.samples, sample_rate=cap.sample_rate, center_frequency=0.0,
                             source_format=cap.source.upper(),
                             metadata={"notes": cap.notes, "carrier_hz": cap.carrier_hz,
                                       "decimation": cap.decimation, "file_sample_rate": cap.file_sample_rate})
        self._check_cancel()
        self._emit(4, "Layer 2: Locating the signals in the spectrum...")
        bands = find_signals(cap.samples, cap.sample_rate) if self.config.get("multi_signal", True) else []
        # Layer 2 detection runs on the whole capture (a channelised signal's filtered-away edges are
        # not spectrum: detecting on them listed filter skirts as signals)
        whole_sois = FFTDetector.detect_sois(whole, threshold_db_above_noise=float(self.config.get("threshold_db", 6.0)))
        if len(bands) < 2:                                   # one signal: analysed whole, as before
            self._lo = 5.0
            res = self._analyze(cap, sois=whole_sois)
            res["signals"], res["bands"] = [res], bands
            res["capture_buffer"], res["capture_notes_all"], res["capture_sois"] = whole, list(cap.notes), whole_sois
            return res
        results = []
        for i, band in enumerate(bands):
            self._check_cancel()
            self._lo, self._hi = 5.0 + 95.0 * i / len(bands), 5.0 + 95.0 * (i + 1) / len(bands)
            self._tag = f"[Signal {i + 1}/{len(bands)} @ {band.center_hz:+,.0f}] "
            try:
                sub = extract(cap, band)
                mine = [s for s in whole_sois if abs(s.center_freq_hz - band.center_hz) <= 0.5 * band.bandwidth_hz
                        + 0.5 * s.bandwidth_hz]
                res = self._analyze(sub, band, sois=mine)
            except _Cancelled:
                raise
            except Exception as e:                           # noqa: BLE001 - one bad signal must not lose the rest
                logger.exception("signal %d failed: %s", i + 1, e)
                res = {"error": str(e), "band": band, "label": band.label(bool(cap.sample_rate)),
                       "parameters": {}, "payload_text": None, "crc_valid": False, "config": dict(self.config),
                       "capture_notes": list(cap.notes), "image": None}
            results.append(res)
        self._tag = ""
        self._hi = 100.0
        self._emit(100, f"Pipeline complete: {len(bands)} signals analysed.")
        primary = next((r for r in results if "error" not in r), results[0])
        out = dict(primary)
        out["signals"], out["bands"] = results, bands
        out["capture_buffer"], out["capture_notes_all"], out["capture_sois"] = whole, list(cap.notes), whole_sois
        return out

    def _analyze(self, cap: Capture, band=None, sois=None) -> Dict[str, Any]:
        """Layers 2-11 on one signal (the whole capture, or one channelised band of it)."""
        fs = cap.sample_rate
        buffer = SignalBuffer(samples=cap.samples, sample_rate=fs, center_frequency=0.0,
                              source_format=cap.source.upper(),
                              metadata={"notes": cap.notes, "carrier_hz": cap.carrier_hz,
                                        "decimation": cap.decimation, "file_sample_rate": cap.file_sample_rate})
        logger.info("Capture %s: %s", cap.path, "; ".join(cap.notes))

        self._check_cancel()
        self._emit(15, "Layer 2: Detecting signals of interest...")
        if sois is None:
            sois = FFTDetector.detect_sois(buffer, threshold_db_above_noise=float(self.config.get("threshold_db", 6.0)))

        label = band.label(bool(fs)) if band is not None else "Whole capture"
        self._check_cancel()
        self._emit(30, "Layer 4: Classifying FSK / PSK / QAM (DSP features, cumulants, lattice fit)...")
        cls = classify_capture(cap.samples, fs)
        params: Dict[str, Parameter] = {}
        params["Sample rate"] = _sample_rate_param(self.config, cap)
        if cap.carrier_hz is not None:
            params["Passband carrier"] = Parameter("Passband carrier", cap.carrier_hz, "Hz", "ESTIMATED",
                                                   "occupied-band centre of the analytic WAV signal")
        mod_param = Parameter("Modulation", cls.modulation, "", "ESTIMATED",
                              "unified DSP decision tree (envelope, cumulants, IF histogram/kurtosis, cyclic "
                              "lines; symbol-level ALRT over PSK/QAM)", cls.confidence, list(cls.evidence))

        self._check_cancel()
        agc = costas = timing = demod = None
        fsk_r = None
        image = None
        if cls.family == "FSK":
            self._emit(55, "Layers 5-6: FSK receiver (rate, tone model, discriminator/correlator/BCJR)...")
            fsk_r = receive_fsk(cap.samples, fs, FSKReceiverConfig())
            if fsk_fit_is_single_carrier(fsk_r):             # "FSK" with ~0 tone spacing: phase-jump PSK
                cls = reclassify_as_psk(cap.samples, fs, cls, f"FSK fit rejected: h = "
                                        f"{fsk_r.parameters['h'].value:.4f} -> constant-envelope PSK")
                mod_param = Parameter("Modulation", cls.modulation, "", "ESTIMATED", mod_param.method,
                                      cls.confidence, list(cls.evidence))
                fsk_r = None
            elif not _fsk_verified(fsk_r):
                image = self._find_image("FSK", fsk_r, 66)
                if image is None or not image.crc_ok:
                    image = self._fsk_tone_fallback(cap, fsk_r, image, mod_param) or image
        if cls.family in ("PSK", "QAM"):
            if cls.family == "PSK":
                self._emit(55, "Layers 5-6: PSK receiver (RRC MF, Gardner, Costas/DD PLL, sync word)...")
                cfg = cls.front_config
                cfg.modulation = cls.modulation
                r = receive_psk(cap.samples, fs, cfg, front=cls.front)
            else:
                self._emit(55, "Layers 5-6: QAM receiver (RRC MF, Gardner, 4th-power + DD carrier, "
                                       "mapping / sync word)...")
                r = receive_qam(cap.samples, fs, QAMReceiverConfig(modulation=cls.modulation), front=cls.front,
                                classification=cls.linear)
            self._emit(57, "Layers 5-6: verifying the decode (CRC frame, text, image / file frame)...")
            r, order_ev = verify_linear_order(
                cap.samples, fs, cls, r,
                lambda i, n, m: self._emit(58 + int(10 * i / max(n, 1)),
                                                   f"Layers 5-6: order check, trying {m} ({i + 1}/{n})..."))
            if order_ev is not None:
                mod_param.evidence.append(order_ev)
                if order_ev.score > 0:
                    cls.family = "QAM" if r.modulation.endswith("QAM") and r.modulation != "4QAM" else "PSK"
            self._check_cancel()
            image = _image_of(r)
            n_sym = len(cls.front.idx) if cls.front is not None else 0
            # (rank < 2: a 'text-heuristic' reading of a slipped long burst is not a verified decode)
            if ((image is None and _decode_quality(r)[0] < 2) or (image is not None and not image.crc_ok)) \
                    and "sps" in r.parameters:
                # Loop receivers slip on long runs of identical symbols (an unscrambled image: a white
                # background = thousands of equal QPSK symbols, no timing / carrier transitions): the
                # screenshot capture lost lock ~21 500 symbols in and the rest of the image became
                # colour noise. The feed-forward receiver (no loops) is tried at the estimated and the
                # exact standard rates; adopted when its file frame is CRC-valid (or is the only one).
                sps0 = float(r.parameters["sps"].value)
                for sps2, rate_label in [(sps0, "the estimated rate")] + _exact_rate_hypotheses(fs, sps0):
                    self._check_cancel()
                    self._emit(64, f"Layers 5-6: feed-forward receiver (no loops) at {rate_label}...")
                    try:
                        rf = receive_feedforward(cap.samples, fs, _linear_name(r.modulation), sps2)
                    except Exception as e:               # noqa: BLE001
                        logger.info("feed-forward receiver at %.5f sps failed: %s", sps2, e)
                        continue
                    img2 = _image_of(rf, cls.family)
                    if img2 is not None and (img2.crc_ok or image is None):
                        why = "loop receivers slipped: CRC failed" if image is not None else "loop receivers found no frame"
                        r, image = rf, img2
                        mod_param.evidence.append(Evidence(
                            f"Feed-forward receiver at {rate_label} recovered the file frame ({why}); "
                            f"{'CRC-valid' if img2.crc_ok else 'CRC not verified'}", "FF_RECEIVER",
                            1.0 if img2.crc_ok else 0.5))
                        if img2.crc_ok:
                            break
            if image is not None and not image.crc_ok and n_sym < IMAGE_RETRY_MAX_SYMBOLS and "sps" in r.parameters:
                # Image found but its CRC fails. A rate estimate a few ppm off (64QAM, 991 symbols:
                # 7.99990 instead of 8 samples/symbol) leaves bit errors that the exact rate does not,
                # so the receiver is re-run at the nearest standard symbol rate and the nearest integer
                # samples/symbol (one receiver pass each); adopted only if the frame CRC then validates.
                for sps2, rate_label in _exact_rate_hypotheses(fs, float(r.parameters["sps"].value)):
                    self._emit(68, f"L11: image CRC not valid; re-demodulating at {rate_label}...")
                    try:
                        r2 = _linear_receive(cap.samples, fs, cls, _linear_name(r.modulation), sps=sps2)
                    except Exception as e:               # noqa: BLE001
                        logger.info("image retry at %.5f sps failed: %s", sps2, e)
                        continue
                    img2 = _image_of(r2)
                    if img2 is not None and img2.crc_ok:
                        r, image = r2, img2
                        mod_param.evidence.append(Evidence(
                            f"Image frame CRC failed at the estimated rate; re-demodulated at {rate_label} it is "
                            f"CRC-valid and was adopted", "IMAGE_RETRY", 1.0))
                        break
            if (_decode_quality(r)[0] == 0 and image is None and cls.p_fsk > 1e-3
                    and n_sym < ORDER_CHECK_MAX_SYMBOLS):
                # Nothing verified as PSK/QAM: on short wideband FSK bursts the envelope features can
                # lean linear. The FSK receiver is adopted only if IT verifies (CRC / text).
                # Long captures classify reliably, and the FSK receiver's per-symbol Viterbi / BCJR
                # scans over hundreds of thousands of symbols run for hours (a 2 M-sample QPSK
                # capture with a binary, unverifiable payload hung here), so it is skipped for them.
                self._emit(69, "Layers 5-6: nothing verified as PSK/QAM; trying the FSK receiver...")
                try:
                    f_alt = receive_fsk(cap.samples, fs, FSKReceiverConfig())
                except Exception as e:                   # noqa: BLE001
                    logger.info("FSK hypothesis failed: %s", e)
                    f_alt = None
                f_img = None
                if f_alt is not None and not fsk_fit_is_single_carrier(f_alt) and not _fsk_verified(f_alt):
                    f_img = self._find_image("FSK", f_alt, 69)
                if (f_alt is not None and not fsk_fit_is_single_carrier(f_alt)
                        and (_fsk_verified(f_alt) or f_img is not None)):
                    fsk_r, cls.family, image = f_alt, "FSK", f_img
                    how = (f"image / file frame ({f_img.describe()})" if f_img is not None else
                           "CRC-valid frame" if f_alt.primary.crc_ok else "decisive text reading")
                    mod_param.evidence.append(Evidence(
                        f"Family revised: no PSK/QAM hypothesis verified, the FSK receiver ({f_alt.variant}) "
                        f"did ({how})", "FAMILY_CHECK", 1.0 if f_alt.primary.crc_ok or (f_img and f_img.crc_ok)
                        else 0.5))
        if cls.family in ("PSK", "QAM"):
            mod_param.evidence.extend(r.evidence)
            mod_param.value = r.modulation                  # e.g. QPSK -> 4QAM from its bit labels
            for p in r.parameters.values():
                if p.name != "Sample rate":
                    params[p.name] = p
            sym_buf = SignalBuffer(samples=r.symbols.astype(np.complex64),
                                   sample_rate=params.get("Symbol rate").value if "Symbol rate" in params else None)
            agc = AGCResult(buffer=buffer, gain_applied=1.0,
                            parameter=Parameter("AGC", 1.0, "", "ESTIMATED", "block AGC to unit power"))
            cfo = r.parameters.get("cfo")
            costas = CostasResult(buffer=sym_buf, cfo_residual_hz=float(cfo.value) if cfo and cfo.value else 0.0,
                                  phase_error_rad=0.0, parameter=cfo)
            timing = TimingRecoveryResult(buffer=sym_buf, sps_est=float(r.parameters["sps"].value),
                                          timing_error_std=0.0, parameter=r.parameters["sps"])
            demod = DemodulationResult(r.hard_bits.astype(np.uint8), r.llrs, r.symbols, 100.0 * r.evm, mod_param)
            payload, crc = r.payload_text, bool(r.decode.frame and r.decode.frame.crc_ok)
            method = r.decode.method
        elif cls.family == "FSK":
            r = fsk_r
            mod_param.value = r.variant
            mod_param.evidence.extend(r.evidence)
            for p in r.parameters.values():
                if p.name != "Sample rate":
                    params[p.name] = p
            demod = DemodulationResult(r.hard_bits.astype(np.uint8), r.llrs, np.array([], dtype=np.complex64),
                                       float(r.fsk_error_pct), mod_param)
            payload, crc, method = r.primary.text, bool(r.primary.crc_ok), r.primary.method
        else:
            payload, crc, method = None, False, "no demodulator for this signal"

        if image is not None:
            mod_param.evidence.extend(image.evidence)
            payload, crc = f"[{image.describe()}]", image.crc_ok
            method = "L11 file frame (CCSDS ASM | LEN u32 | PAYLOAD | CRC-32)"
            params["Recovered image" if image.is_image else "Recovered file"] = Parameter(
                "Recovered image" if image.is_image else "Recovered file", image.describe(), "",
                "ESTIMATED", method, 1.0 if image.crc_ok else 0.6, list(image.evidence))

        self._check_cancel()
        # Layers 8-11: bitstream -> de-interleaving -> FEC -> framing (joint hypothesis search)
        post = None
        # (skipped when a file frame was recovered: its layout, FEC and CRC are already identified)
        if cls.family in ("PSK", "QAM", "FSK") and demod is not None and len(r.hard_bits) and image is None:
            self._emit(70, "Layers 8-11: bitstream, de-interleaving, FEC, framing...")
            try:
                post = analyze_postdemod(hypotheses_from_receiver(cls.family, r),
                                         progress=self._postdemod_progress,
                                         budget_s=float(self.config.get("postdemod_budget_s", 600.0)),
                                         receiver_frame_ok=crc,
                                         symbols=demod.constellation_points if cls.family in ("PSK", "QAM") else None,
                                         modulation=r.modulation if cls.family in ("PSK", "QAM") else None)
            except Exception as e:                       # never lose the demodulated result
                logger.exception("post-demodulation analysis failed: %s", e)
            self._check_cancel()
            if post is not None and post.histogram_check is not None:
                hc = post.histogram_check                # symbols disagree with the modulation: lower confidence
                mod_param.evidence.extend(hc.evidence)
                if hc.log_odds < 0:
                    mod_param.confidence = float((mod_param.confidence or 0.0) * hc.p_consistent)
            if post is not None and post.verified:
                b = post.best
                chain = []
                if b.deint.kind != "none":
                    chain.append(f"de-interleaver {b.deint.kind} {b.deint.params}")
                if b.fec_name != "none":
                    chain.append(b.fec_name)
                payload, crc = post.payload_text, True
                method = "L8-L11: " + (" + ".join(chain) if chain else "uncoded") +                          f"; frame CRC {b.frames.layout.crc if b.frames.layout else ''}"

        params["Modulation"] = mod_param
        if payload is not None:
            params["Decoded payload"] = Parameter("Decoded payload", payload, "", "ESTIMATED",
                                                  f"{method}; CRC {'VALID' if crc else 'not valid / no frame'}",
                                                  1.0 if crc else 0.5)
        self._emit(100, "Pipeline complete.")
        return {"buffer": buffer, "sois": sois, "parameters": params, "agc_result": agc,
                "timing_result": timing, "costas_result": costas, "demod_result": demod,
                "payload_text": payload, "crc_valid": crc, "classification": cls, "postdemod": post,
                "config": dict(self.config), "capture_notes": list(cap.notes), "image": image,
                "band": band, "label": label}
