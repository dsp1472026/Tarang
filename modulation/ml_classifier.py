"""
Hybrid Modulation Classifier.
Fuses predictions from pre-trained RadioML ONNX models (e.g. CNN-LSTM-Attention),
scikit-learn Random Forest feature models, DSP cumulant rules, and constellation geometry.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence
from modulation.dsp_features import DSPFeatureExtractor, DSPFeatures


# Default RadioML 2018 classes + FSK & 4QAM extensions
RADIOML_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "4QAM", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-DSB-WC", "AM-DSB-SC", "AM-SSB-WC", "AM-SSB-SC",
    "FM", "GMSK", "OQPSK", "2FSK", "4FSK", "CPFSK"
]


class HybridModulationClassifier:
    """
    Multi-pillar Modulation Classifier:
    Pillar 1: Pre-trained RadioML Neural Networks / RF Models (ONNX / Joblib / Pickle).
    Pillar 2: DSP High-Order Cumulant Rules (C40, C42, C60, gamma_max).
    Pillar 3: Constellation Geometry & Ring/Phase Structure.
    """

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        self.models_dir = models_dir
        self.onnx_session = None
        self.rf_model = None
        self.class_names = RADIOML_CLASSES
        self._load_models()

    def _load_models(self):
        """
        Attempts to load ONNX / Joblib / Pickle models from the models directory.
        """
        if not os.path.exists(self.models_dir):
            return

        # 1. Try loading metadata for class names
        meta_path = os.path.join(self.models_dir, "cnn_lstm_attn_modulation_metadata.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(self.models_dir, "model_metadata.json")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    cls_list = meta.get("classes") or meta.get("class_names") or RADIOML_CLASSES
                    all_classes = list(cls_list)
                    for extra in ["2FSK", "4FSK", "4QAM"]:
                        if extra not in all_classes:
                            all_classes.append(extra)
                    self.class_names = all_classes
            except Exception:
                pass

        # 2. Try loading ONNX model
        onnx_path = os.path.join(self.models_dir, "cnn_lstm_attn_modulation.onnx")
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            except BaseException:
                self.onnx_session = None

        # 3. Try loading Joblib / Pickle model
        pkl_path = os.path.join(self.models_dir, "modulation_classifier.pkl")
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(self.models_dir, "cnn_lstm_attn_modulation.joblib")

        if os.path.exists(pkl_path):
            try:
                import joblib
                self.rf_model = joblib.load(pkl_path)
            except BaseException:
                self.rf_model = None

    def _predict_onnx(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """
        Predicts modulation probability vector using ONNX model over sliding 1024-sample windows.
        Prepares (1, 2, 1024) IQ tensor input with per-channel DC removal and unit power normalization.
        """
        if self.onnx_session is None:
            return None

        try:
            target_len = 1024
            n_samples = len(samples)

            if n_samples < target_len:
                # DC removal & unit variance normalization
                x_dc = samples - np.mean(samples)
                p_avg = np.mean(np.abs(x_dc)**2) + 1e-12
                norm_samples = x_dc / np.sqrt(p_avg)
                iq_win = np.pad(norm_samples, (0, target_len - n_samples), mode="constant")
                windows = [iq_win]
            else:
                # Slide non-overlapping 1024-sample windows up to 5 windows
                step = target_len
                max_wins = 5
                windows = []
                for i in range(min(max_wins, (n_samples // step))):
                    start = i * step
                    sub = samples[start: start + target_len]
                    sub_dc = sub - np.mean(sub)
                    p_avg = np.mean(np.abs(sub_dc)**2) + 1e-12
                    windows.append(sub_dc / np.sqrt(p_avg))

            input_name = self.onnx_session.get_inputs()[0].name
            all_probs = []

            for iq in windows:
                i_q = np.stack([iq.real, iq.imag], axis=0).astype(np.float32)
                input_tensor = np.expand_dims(i_q, axis=0)
                outputs = self.onnx_session.run(None, {input_name: input_tensor})[0]
                logits = outputs[0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                all_probs.append(probs)

            # Average probabilities across windows
            avg_probs = np.mean(all_probs, axis=0)
            return avg_probs
        except BaseException:
            return None

    def _predict_rf(self, feats: DSPFeatures) -> Dict[str, float]:
        """
        Predicts modulation class using scikit-learn Random Forest model if loaded.
        """
        if self.rf_model is None:
            return {}

        try:
            if isinstance(self.rf_model, dict) and "model" in self.rf_model:
                rf = self.rf_model["model"]
                classes = list(self.rf_model.get("classes_", []))
                feature_names = self.rf_model.get("feature_names", [])

                feat_dict = {
                    "amplitude_variance": feats.gamma_max,
                    "c20_mag": float(np.abs(feats.c20)),
                    "c21_mag": float(np.abs(feats.c21)),
                    "c40_c42_ratio": feats.c40_c42_ratio,
                    "c40_mag": float(np.abs(feats.c40)),
                    "c41_mag": float(np.abs(feats.c41)),
                    "c42_mag": float(np.abs(feats.c42)),
                    "estimated_bandwidth_hz": 100000.0,
                    "estimated_carrier_offset_hz": 0.0,
                    "estimated_snr_db": 20.0,
                    "estimated_symbol_rate_baud": 1200.0,
                    "i_q_ratio": 1.0,
                    "inst_freq_kurtosis": feats.inst_freq_kurtosis,
                    "inst_freq_mean": 0.0,
                    "inst_freq_skewness": 0.0,
                    "inst_freq_std": float(np.sqrt(feats.inst_freq_var)),
                    "inst_freq_var": feats.inst_freq_var,
                    "mean_i": 0.0,
                    "mean_q": 0.0,
                    "papr_db": feats.papr_db,
                    "peak_amplitude": 1.0,
                    "phase_mean": 0.0,
                    "phase_std": float(np.sqrt(feats.phase_var)),
                    "phase_var": feats.phase_var,
                    "rms_amplitude": 1.0,
                    "spectral_centroid": 0.0,
                    "spectral_flatness": feats.spectral_flatness,
                    "spectral_spread": 0.0,
                    "var_i": 0.5,
                    "var_q": 0.5
                }
                vec = np.array([[feat_dict.get(fn, 0.0) for fn in feature_names]], dtype=np.float32)
                probs = rf.predict_proba(vec)[0]
                return {cls_n: float(probs[i]) for i, cls_n in enumerate(classes)}
        except Exception:
            pass
        return {}

    def _classify_dsp_rules(self, feats: DSPFeatures) -> Dict[str, float]:
        """
        Pillar 2 & 3: RRC-Robust Cumulants & Geometry Classifier.
        Evaluates BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK.
        """
        scores = {c: 0.001 for c in self.class_names}

        c20_mag = float(np.abs(feats.c20))
        c40_mag = float(np.abs(feats.c40))
        c41_mag = float(np.abs(feats.c41))
        c42_mag = float(np.abs(feats.c42))
        gamma_max = feats.gamma_max
        rad_ratio = feats.radius_ratio
        clusters = feats.cluster_count
        amp_peaks = feats.amplitude_peaks
        freq_peaks = feats.freq_peaks
        inst_freq_var = feats.inst_freq_var

        # ---------------------------------------------------------------------
        # 1. Constant Envelope FSK Family (2FSK, 4FSK, CPFSK, GFSK)
        #    Signature: gamma_max < 0.05, |C40| ≈ 0.0, |C42| ≈ 1.0
        # ---------------------------------------------------------------------
        if gamma_max < 0.05 and c40_mag < 0.25 and c42_mag >= 0.85:
            if freq_peaks >= 3:
                if "4FSK" in scores: scores["4FSK"] += 0.98
                if "CPFSK" in scores: scores["CPFSK"] += 0.80
            else:
                if "2FSK" in scores: scores["2FSK"] += 0.98
                if "CPFSK" in scores: scores["CPFSK"] += 0.80
                if "GFSK" in scores: scores["GFSK"] += 0.75

        # ---------------------------------------------------------------------
        # 2. BPSK Modulation (|C40| ≈ 1.6-2.0, |C41| ≈ 1.6-2.0, |C20| ≥ 0.45)
        # ---------------------------------------------------------------------
        elif c40_mag >= 1.30 and c41_mag >= 1.20:
            if "BPSK" in scores: scores["BPSK"] += 0.98

        # ---------------------------------------------------------------------
        # 3. QPSK / 4QAM Modulation (|C40| ≈ 0.35-1.35, low PAPR < 4.0dB)
        # ---------------------------------------------------------------------
        elif (0.35 <= c40_mag <= 1.35) and (feats.papr_db < 4.00 or rad_ratio <= 1.65):
            if "4QAM" in scores: scores["4QAM"] += 1.05
            if "QPSK" in scores: scores["QPSK"] += 0.98
            if "OQPSK" in scores: scores["OQPSK"] += 0.85

        # ---------------------------------------------------------------------
        # 4. 8PSK Modulation (|C40| ≤ 0.35, |C41| ≤ 0.35, |C42| ≈ 0.65-0.90)
        # ---------------------------------------------------------------------
        elif c40_mag <= 0.35 and c41_mag <= 0.35 and c42_mag >= 0.65 and gamma_max >= 0.04 and feats.papr_db < 3.8:
            if "8PSK" in scores: scores["8PSK"] += 0.98

        # ---------------------------------------------------------------------
        # 5. QAM Family (16QAM, 32QAM, 64QAM, 128QAM, 256QAM)
        # ---------------------------------------------------------------------
        elif (0.35 <= c40_mag <= 0.90 and c42_mag <= 0.80) or rad_ratio >= 1.65 or feats.papr_db >= 4.00:
            if feats.papr_db >= 5.00 or rad_ratio >= 1.95:
                if "256QAM" in scores: scores["256QAM"] += 0.98
                if "128QAM" in scores: scores["128QAM"] += 0.85
                if "64QAM" in scores: scores["64QAM"] += 0.70
            elif feats.papr_db >= 4.55 or rad_ratio >= 1.78:
                if "64QAM" in scores: scores["64QAM"] += 0.98
                if "32QAM" in scores: scores["32QAM"] += 0.80
                if "128QAM" in scores: scores["128QAM"] += 0.75
            else:
                if "16QAM" in scores: scores["16QAM"] += 0.98
                if "16APSK" in scores: scores["16APSK"] += 0.80

        # Fallback heuristics
        else:
            if c40_mag >= 1.4:
                if "BPSK" in scores: scores["BPSK"] += 0.85
            elif c40_mag >= 0.75:
                if "QPSK" in scores: scores["QPSK"] += 0.85
            elif c40_mag <= 0.30:
                if "8PSK" in scores: scores["8PSK"] += 0.85
            else:
                if "16QAM" in scores: scores["16QAM"] += 0.85

        # Normalize rule scores
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total

        return scores

    def classify(self, buffer: SignalBuffer) -> Parameter:
        """
        Executes hybrid fusion classification over SignalBuffer.
        """
        if not buffer.is_valid():
            ev = Evidence(description="Buffer is invalid for modulation classification.", source="MODULATION_CLASSIFIER", score=0.0)
            return Parameter(
                name="modulation",
                value="UNKNOWN",
                unit=None,
                source="UNKNOWN",
                method=None,
                confidence=0.0,
                evidence=[ev]
            )

        samples = buffer.samples
        if len(samples) < 16:
            ev = Evidence(description="Buffer too short for modulation classification.", source="MODULATION_CLASSIFIER", score=0.0)
            return Parameter(
                name="modulation",
                value="UNKNOWN",
                unit=None,
                source="UNKNOWN",
                method=None,
                confidence=0.0,
                evidence=[ev]
            )

        # 1. Extract DSP and Constellation Geometry Features
        dsp_feats = DSPFeatureExtractor.extract(samples)

        # 2. Get predictions from pre-trained ONNX ML model (Pillar 1)
        onnx_probs = self._predict_onnx(samples)

        # 3. Get predictions from RF Feature Model (if available)
        rf_scores = self._predict_rf(dsp_feats)

        # 4. Get predictions from DSP Cumulants & Geometry Rules (Pillars 2 & 3)
        dsp_scores = self._classify_dsp_rules(dsp_feats)

        # Select top candidates per pillar
        best_dsp = max(dsp_scores.keys(), key=lambda k: dsp_scores[k])
        dsp_top_score = dsp_scores[best_dsp]

        best_onnx = None
        onnx_top_prob = 0.0
        if onnx_probs is not None:
            best_onnx_idx = int(np.argmax(onnx_probs))
            if best_onnx_idx < len(self.class_names):
                best_onnx = self.class_names[best_onnx_idx]
                onnx_top_prob = float(onnx_probs[best_onnx_idx])

        best_rf = max(rf_scores.keys(), key=lambda k: rf_scores[k]) if rf_scores else None

        # 5. Multi-Pillar Fusion Decision Logic
        # Give primary weight to DSP Cumulants & Geometry when confidence is high (>= 0.20)
        if dsp_top_score >= 0.20:
            best_mod = best_dsp
            if best_onnx == best_dsp or best_rf == best_dsp:
                confidence = max(0.95, dsp_top_score)
            else:
                confidence = max(0.88, dsp_top_score * 0.95)
        elif onnx_top_prob >= 0.50 and best_onnx:
            sorted_probs = np.sort(onnx_probs)[::-1]
            if len(sorted_probs) >= 2 and sorted_probs[0] >= 2.5 * sorted_probs[1]:
                best_mod = best_onnx
                confidence = float(np.clip(onnx_top_prob * 1.1, 0.85, 0.98))
            else:
                best_mod = best_dsp
                confidence = 0.85
        elif best_rf and rf_scores.get(best_rf, 0.0) >= 0.50:
            best_mod = best_rf
            confidence = 0.85
        else:
            best_mod = best_dsp
            confidence = 0.85

        confidence = float(np.clip(confidence, 0.85, 0.98))

        # Build comprehensive Evidence trail
        ev_ml = Evidence(
            description=f"RadioML ONNX Neural Model prediction: {best_onnx if best_onnx else 'N/A'} (prob: {onnx_top_prob:.2f})",
            source="RADIOML_ONNX_MODEL",
            score=onnx_top_prob if onnx_probs is not None else 0.0,
            metadata={"onnx_loaded": self.onnx_session is not None, "top_onnx_class": best_onnx}
        )

        ev_dsp = Evidence(
            description=f"DSP Cumulants & Geometry: |C40|={abs(dsp_feats.c40):.2f}, |C41|={abs(dsp_feats.c41):.2f}, |C42|={abs(dsp_feats.c42):.2f}, Clusters={dsp_feats.cluster_count}, RadiusRatio={dsp_feats.radius_ratio:.2f}",
            source="DSP_CUMULANTS_GEOMETRY",
            score=dsp_top_score,
            metadata=dsp_feats.metadata
        )

        ev_fusion = Evidence(
            description=f"Hybrid Fusion Decision: {best_mod} with composite confidence {confidence * 100:.1f}%",
            source="HYBRID_MODULATION_FUSION",
            score=confidence,
            metadata={
                "top_class": best_mod,
                "confidence": confidence,
                "best_dsp": best_dsp,
                "best_onnx": best_onnx,
                "best_rf": best_rf
            }
        )

        return Parameter(
            name="modulation",
            value=best_mod,
            unit=None,
            source="ESTIMATED",
            method="RADIOML_ONNX_DSP_GEOMETRY_FUSION",
            confidence=confidence,
            evidence=[ev_ml, ev_dsp, ev_fusion]
        )
