"""
L8 forensics of a bit stream as one module-level function, so the GUI can run it in a worker
process (gui/widgets/background.BackgroundRunner.submit_process). In a thread its long numpy calls
hold Python's GIL and still froze the window for 1-2 s over a million bits.
"""

import numpy as np

from bitstream import sync_search
from bitstream.correlation import BitstreamCorrelation
from bitstream.entropy import BitstreamEntropy
from deinterleaving.block import rank_profile
from fec.identify import blind_conv_params


def run(bits, llrs=None):
    """(randomness, periodicity, autocorrelation, sync detections, rank profile, blind conv estimate,
    frame length) of `bits`, as shown by the Bitstream tab."""
    bits = np.asarray(bits, dtype=np.uint8)
    syncs = sync_search.search(bits, llrs)
    prof = rank_profile(bits, 64)
    per = BitstreamCorrelation.find_periodicity(bits, max_lag=min(512, len(bits) - 1))
    ac = BitstreamCorrelation.autocorrelation(bits, max_lag=min(512, len(bits) - 1))
    det = sync_search.best_detection(syncs)
    return (BitstreamEntropy.estimate_randomness(bits), per, ac, syncs, prof, blind_conv_params(prof),
            det.period if det and det.period else (per[0][0] if per else None))
