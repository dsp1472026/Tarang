"""Text + numeric messages x modulations x generator styles, through the GUI worker.

    python tools/text_numeric_validate.py [styles] [mods] [kinds]
    styles: ext (p..py style: 32-bit preamble + ASCII), unframed, framed   kinds: text, numeric, digits
"""
import importlib.util
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np  # noqa: E402

MSGS = {"text": "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS.",
        "numeric": "9876543210 0123456789 31415926535",
        "digits": "20260926123456789"}
MODS = ["BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "64QAM", "256QAM", "2FSK", "4FSK", "8FSK"]


def _pangram():
    spec = importlib.util.spec_from_file_location("pg", os.path.join(ROOT, "pangrams", "p..py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ext_signal(mod, text, seed, preamble=True):
    """p..py style: [32-bit preamble] + ASCII, Gray labels, RRC, fs 96k, 4800 Bd."""
    pg = _pangram()
    bits = pg.build_frame_bits(text) if preamble else pg.text_to_bits(text)
    rrc = pg.rrc_filter(0.35, 20, 10)
    if mod == "BPSK":
        s = pg.modulate_bpsk(bits)
    elif mod == "QPSK":
        s = pg.modulate_qpsk(bits)
    elif mod in ("8PSK", "16PSK"):
        k = 3 if mod == "8PSK" else 4
        sym = pg.bits_to_symbols(bits, k)
        b = np.array([pg.gray_to_binary(g) for g in sym])
        s = np.exp(2j * np.pi * b / 2 ** k).astype(np.complex64)
    elif mod.endswith("QAM"):
        M = int(mod[:-3]); bps = int(np.log2(M)); bpa = bps // 2
        bits = np.r_[bits, np.zeros((-len(bits)) % bps, dtype=np.uint8)].reshape(-1, bps)
        lev = np.arange(-(2 ** bpa - 1), 2 ** bpa, 2)          # fixed p..py pam_levels
        ax = lambda b: lev[[pg.gray_to_binary(g) for g in pg.bits_to_symbols(b.reshape(-1), bpa)]]
        s = (ax(bits[:, :bpa]) + 1j * ax(bits[:, bpa:])).astype(np.complex64)
    else:
        x = pg.modulate_cpfsk(bits, int(mod[0]), 20, h=1.0 if mod == "8FSK" else 0.5)
        s = None
    if s is not None:
        x = pg.upsample_and_shape(s, rrc, 20)
    x = np.concatenate([np.zeros(200), x, np.zeros(200)]).astype(np.complex128)
    rng = np.random.default_rng(seed)
    x = x * np.exp(1j * (0.7 + 2 * np.pi * 37.0 / 96000 * np.arange(len(x))))
    p = np.mean(np.abs(x[200:-200]) ** 2)
    snr = {"256QAM": 40, "64QAM": 34, "16QAM": 28, "16PSK": 30}.get(mod, 25)
    esn0_per_sample = snr - 10 * np.log10(20)
    nv = p / 10 ** (esn0_per_sample / 10)
    x = x + np.sqrt(nv / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x, 96000.0


def own_signal(mod, text, seed, framed):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import iq_format_charset_validate as v
    x, truth = v.make_signal(mod, text, framed, seed)
    return x, truth["sample_rate_hz"]


def run_case(style, mod, kind, seed):
    from gui.workers.unified_worker import UnifiedPipelineWorker
    from ingestion.iq_formats import encode_iq
    text = MSGS[kind]
    if style == "ext":
        x, fs = ext_signal(mod, text, seed)
    elif style == "ext-nopre":
        x, fs = ext_signal(mod, text, seed, preamble=False)
    else:
        x, fs = own_signal(mod, text, seed, framed=(style == "framed"))
    d = tempfile.mkdtemp()
    path = os.path.join(d, "rec.iq")
    with open(path, "wb") as fh:
        fh.write(encode_iq(0.4 * x / np.max(np.abs(x)), "cf32"))
    cfg = {"file_path": path, "dtype": "cf32_le", "endianness": "little", "iq_order": "IQ", "byte_offset": 0,
           "sample_rate": fs, "center_frequency": 0.0, "threshold_db": 6.0, "postdemod_budget_s": 3600.0}
    t0 = time.time()
    try:
        res = UnifiedPipelineWorker(cfg).work()
        got = res["payload_text"] or ""
        cls = str(res["parameters"]["Modulation"].value)
        meth = res["parameters"].get("Decoded payload")
        meth = meth.method if meth else ""
    except Exception as e:  # noqa: BLE001
        got, cls, meth = "", "-", f"ERR {type(e).__name__}: {e}"
    ok = text in got
    return dict(style=style, mod=mod, kind=kind, ok=ok, cls=cls, got=got, meth=meth, t=time.time() - t0)


if __name__ == "__main__":
    styles = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ext", "unframed", "framed"]
    mods = sys.argv[2].split(",") if len(sys.argv) > 2 else MODS
    kinds = sys.argv[3].split(",") if len(sys.argv) > 3 else list(MSGS)
    cases = [(s, m, k) for s in styles for m in mods for k in kinds]
    rows = []
    with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as ex:
        futs = [ex.submit(run_case, s, m, k, 7 + i) for i, (s, m, k) in enumerate(cases)]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"{'PASS' if r['ok'] else 'FAIL'} {r['style']:9s} {r['mod']:7s} {r['kind']:8s} cls={r['cls']:7s} "
                  f"{r['t']:5.1f}s " + ("" if r["ok"] else f"got={r['got'][:60]!a} | {r['meth'][:110]}"), flush=True)
    print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)}")
