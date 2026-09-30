import numpy as np
import pytest
from conditioning.bit_order import BitOrderConverter


def test_bit_order_candidate_generator():
    candidates = BitOrderConverter.candidate_generator()
    assert "MSB" in candidates
    assert "LSB" in candidates
    assert "SWAP_BITS_PER_BYTE" in candidates


def test_bit_order_conversion():
    # 8-bit pattern: 1, 0, 1, 1, 0, 0, 0, 1 (0xB1 = 177)
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)

    msb_bits = BitOrderConverter.convert_bit_order(bits, mode="MSB")
    assert np.array_equal(msb_bits, bits)

    lsb_bits = BitOrderConverter.convert_bit_order(bits, mode="LSB")
    assert np.array_equal(lsb_bits, bits[::-1])

    swap_bits = BitOrderConverter.convert_bit_order(bits, mode="SWAP_BITS_PER_BYTE")
    assert np.array_equal(swap_bits, bits[::-1])


def test_bit_order_packing_unpacking():
    original_data = b"HELLO 123"
    bits = BitOrderConverter.unpack_bytes(original_data, mode="MSB")
    packed = BitOrderConverter.pack_bits(bits, mode="MSB")
    assert packed == original_data

    # Test roundtrip with SWAP_BITS_PER_BYTE
    swap_bits = BitOrderConverter.unpack_bytes(original_data, mode="SWAP_BITS_PER_BYTE")
    packed_swap = BitOrderConverter.pack_bits(swap_bits, mode="SWAP_BITS_PER_BYTE")
    assert packed_swap == original_data


def test_bit_order_empty_and_invalid():
    empty_bits = BitOrderConverter.convert_bit_order(np.array([], dtype=np.uint8), mode="MSB")
    assert len(empty_bits) == 0

    with pytest.raises(ValueError):
        BitOrderConverter.convert_bit_order(np.array([1, 0]), mode="INVALID_MODE")
