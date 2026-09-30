"""
End-to-end image round trip: test image -> transmit_image -> .iq -> receive_image -> verify.

    python synthetic/run_image_pipeline.py

Writes the signals to synthetic/results/signals/, the reconstructed images to
synthetic/results/reconstructed/ and the report to synthetic/results/image_pipeline_report.json.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic.gen_test_image import IMAGE_DIR, generate_test_images  # noqa: E402
from synthetic.rx_image import receive_image  # noqa: E402
from synthetic.tx_image import transmit_image  # noqa: E402
from synthetic.verify_roundtrip import verify  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "synthetic", "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "image_pipeline_report.json")
DEFAULT_JOBS: List[Tuple[str, str]] = [
    ("test_16x16.jpg", "BPSK"),
    ("test_16x16.jpg", "QPSK"),
    ("test_16x16.jpg", "16QAM"),
    ("test_16x16.jpg", "4FSK"),
    ("test_32x32.png", "QPSK"),
    ("test_8x8.bmp", "BPSK"),
]
DEFAULTS = {"fec": True, "sps": 8, "baud": 1200, "snr_db": 25.0}


def run_job(image_path: str, modulation: str, results_dir: str = RESULTS_DIR, fec: bool = True, sps: int = 8,
            baud: int = 1200, snr_db: float = 25.0) -> Dict[str, Any]:
    """One transmit -> receive -> verify round trip; returns {tx, rx, verify, seconds}."""
    t0 = time.time()
    tx = transmit_image(image_path, modulation, os.path.join(results_dir, "signals"), fec, sps, baud, snr_db)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out = os.path.join(results_dir, "reconstructed", f"{stem}_{modulation.lower()}{os.path.splitext(image_path)[1]}")
    rx = receive_image(tx["iq_cs16_path"], modulation, out, fec, sps, baud)
    return {"image": os.path.basename(image_path), "modulation": modulation.upper(), "tx": tx, "rx": rx,
            "verify": verify(image_path, rx["output_image_path"]), "seconds": round(time.time() - t0, 2)}


def run_pipeline(jobs: Sequence[Tuple[str, str]] = DEFAULT_JOBS, image_dir: str = IMAGE_DIR,
                 results_dir: str = RESULTS_DIR, report_path: Optional[str] = REPORT_PATH,
                 **params: Any) -> List[Dict[str, Any]]:
    """Run every (image file name, modulation) job; generates the test images if missing."""
    cfg = {**DEFAULTS, **params}
    if any(not os.path.exists(os.path.join(image_dir, img)) for img, _ in jobs):
        generate_test_images(image_dir)
    rows = [run_job(os.path.join(image_dir, img), mod, results_dir, **cfg) for img, mod in jobs]
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as fh:
            json.dump({"settings": cfg, "jobs": rows}, fh, indent=2, default=str)
    return rows


def format_table(rows: List[Dict[str, Any]]) -> str:
    """The summary table: image | modulation | FEC | CRC | format | bytes | match | time."""
    head = f"{'image':16s} | {'modulation':10s} | {'FEC':5s} | {'CRC':4s} | {'format':7s} | {'bytes':>5s} | " \
           f"{'match':5s} | {'time':>6s}"
    lines = [head, "-" * len(head)]
    for r in rows:
        rx, v = r["rx"], r["verify"]
        lines.append(f"{r['image']:16s} | {r['modulation']:10s} | {str(r['tx']['fec']):5s} | "
                     f"{'ok' if rx['crc_ok'] else 'FAIL':4s} | {rx['detected_format']:7s} | "
                     f"{v['reconstructed_bytes']:5d} | {str(v['match']):5s} | {r['seconds']:5.1f}s")
    ok = sum(r["verify"]["match"] for r in rows)
    lines.append(f"{ok}/{len(rows)} byte-exact (SHA-256)")
    return "\n".join(lines)


def main() -> int:
    rows = run_pipeline()
    print(format_table(rows))
    print(f"report: {os.path.relpath(REPORT_PATH, ROOT)}")
    return 0 if all(r["verify"]["match"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
