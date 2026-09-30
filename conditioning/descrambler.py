from typing import List, Dict, Any, Tuple
import numpy as np


class LFSRDescrambler:
    """
    Linear Feedback Shift Register (LFSR) descrambler/scrambler for bitstream descrambling.
    Supports standard PN sequences (PN9, PN11, PN15, DVB-S, CCSDS) and custom polynomials.
    """

    PRESETS: Dict[str, Dict[str, Any]] = {
        "DISABLED": {"taps": [], "order": 0, "seed": 0},
        "PN9": {"taps": [9, 5], "order": 9, "seed": 0x1FF},
        "PN11": {"taps": [11, 9], "order": 11, "seed": 0x7FF},
        "PN15": {"taps": [15, 14], "order": 15, "seed": 0x7FFF},
        "DVB-S": {"taps": [15, 14], "order": 15, "seed": 0x4A80},
        "CCSDS": {"taps": [8, 7, 5, 3], "order": 8, "seed": 0xFF},
    }

    @classmethod
    def candidate_generator(cls) -> List[str]:
        """
        Yields preset candidate names for downstream hypothesis testing.
        """
        return list(cls.PRESETS.keys())

    @classmethod
    def generate_pn_sequence(
        cls,
        length: int,
        taps: List[int],
        seed: int,
        order: int
    ) -> np.ndarray:
        """
        Generates a 1D binary pseudo-random sequence of given length from LFSR taps and seed.
        """
        if length <= 0 or order <= 0 or not taps:
            return np.zeros(length, dtype=np.uint8)

        seq = np.zeros(length, dtype=np.uint8)
        state = seed & ((1 << order) - 1)
        if state == 0:
            state = 1  # LFSR non-zero initial state fallback

        for i in range(length):
            # Compute feedback bit from specified taps (1-indexed tap positions)
            fb = 0
            for tap in taps:
                fb ^= (state >> (order - tap)) & 1

            # Output MSB bit of state
            out_bit = (state >> (order - 1)) & 1
            seq[i] = out_bit

            # Shift state left and insert feedback bit
            state = ((state << 1) | fb) & ((1 << order) - 1)

        return seq

    @classmethod
    def process(
        cls,
        bits: np.ndarray,
        preset: str = "DISABLED",
        taps: List[int] = None,
        seed: int = None,
        order: int = None
    ) -> np.ndarray:
        """
        Descrambles (or scrambles) input bits via XOR with generated LFSR sequence.

        Args:
            bits: 1D input array of bits (0 or 1)
            preset: Preset configuration name ('DISABLED', 'PN9', 'PN11', 'PN15', 'DVB-S', 'CCSDS')
            taps: Custom LFSR tap positions if preset is not used
            seed: Custom initial seed
            order: Custom LFSR register size

        Returns:
            Descrambled 1D bit array
        """
        if len(bits) == 0:
            return np.array([], dtype=np.uint8)

        bits = np.asarray(bits, dtype=np.uint8) & 1

        preset_upper = preset.upper() if preset else "DISABLED"
        if preset_upper == "DISABLED" or preset_upper == "NONE":
            return bits.copy()

        if preset_upper in cls.PRESETS:
            cfg = cls.PRESETS[preset_upper]
            use_taps = taps if taps is not None else cfg["taps"]
            use_seed = seed if seed is not None else cfg["seed"]
            use_order = order if order is not None else cfg["order"]
        else:
            if taps is None or seed is None or order is None:
                raise ValueError(f"Unknown preset '{preset}' requires explicit taps, seed, and order.")
            use_taps, use_seed, use_order = taps, seed, order

        pn_seq = cls.generate_pn_sequence(len(bits), use_taps, use_seed, use_order)
        return (bits ^ pn_seq).astype(np.uint8)
