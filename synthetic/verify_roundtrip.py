"""
Byte-exact comparison of an original file and its reconstruction (SHA-256).
"""

import hashlib
import os
from typing import Any, Dict, Optional


def _sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def verify(image_path: str, output_image_path: Optional[str]) -> Dict[str, Any]:
    """
    SHA-256 of both files and whether they are identical. A missing reconstruction (e.g. the
    receiver's CRC failed and nothing was written) gives match = False and 0 bytes.
    """
    a, b = _sha256(image_path), _sha256(output_image_path)
    return {"original_sha256": a, "reconstructed_sha256": b,
            "match": a is not None and a == b,
            "original_bytes": os.path.getsize(image_path) if a else 0,
            "reconstructed_bytes": os.path.getsize(output_image_path) if b else 0}
