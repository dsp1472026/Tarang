import pytest
from core.evidence import Evidence, Parameter, EvidenceCollection


def test_evidence_creation():
    ev = Evidence(
        description="Peak found at 10.5 MHz",
        source="FFT_DETECTOR",
        score=0.92,
        metadata={"peak_power_db": 15.2}
    )

    assert ev.description == "Peak found at 10.5 MHz"
    assert ev.source == "FFT_DETECTOR"
    assert ev.score == 0.92
    assert ev.metadata["peak_power_db"] == 15.2
    assert ev.to_dict()["score"] == 0.92


def test_parameter_creation():
    ev = Evidence(description="Metadata header read", source="SIGMF_READER", score=1.0)
    param = Parameter(
        name="sample_rate",
        value=10e6,
        unit="Hz",
        source="KNOWN",
        method="HEADER_PARSER",
        confidence=1.0,
        evidence=[ev]
    )

    assert param.name == "sample_rate"
    assert param.value == 10e6
    assert param.source == "KNOWN"
    assert len(param.evidence) == 1
    assert param.to_dict()["source"] == "KNOWN"


def test_evidence_collection():
    ec = EvidenceCollection()
    e1 = Evidence(description="DSP Cumulant match", source="DSP_FEAT", score=0.8)
    e2 = Evidence(description="CNN output match", source="ML_MODEL", score=0.9)

    ec.add(e1)
    ec.add(e2)

    assert len(ec.items) == 2
    assert ec.mean_score() == pytest.approx(0.85)
    assert ec.weighted_score([0.4, 0.6]) == pytest.approx((0.4 * 0.8 + 0.6 * 0.9) / 1.0)

    dsp_items = ec.filter_by_source("DSP_FEAT")
    assert len(dsp_items) == 1
    assert dsp_items[0].score == 0.8
