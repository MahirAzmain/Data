import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import wfdb
from scipy.io import loadmat, wavfile
from scipy.signal import stft


ECG_LEADS = 12
DEFAULT_PCG_HEIGHT = 128
DEFAULT_PCG_WIDTH = 128


class SimpleMambaBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, ssm_dim, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ssm_dim = ssm_dim
        self.dropout = dropout
        self.dense1 = tf.keras.layers.Dense(ssm_dim, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.gate_dense = tf.keras.layers.Dense(d_model, activation="sigmoid")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.dense1(x)
        h = self.drop(h, training=training)
        h = self.dense2(h)
        h = self.drop(h, training=training)
        gate = self.gate_dense(x)
        h = gate * h + (1 - gate) * x
        return self.norm(h)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "ssm_dim": self.ssm_dim, "dropout": self.dropout})
        return config


class MambaBlock(SimpleMambaBlock):
    pass


class MultimodalMambaBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, ssm_dim, dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.ssm_dim = ssm_dim
        self.dropout = dropout
        self.dense1 = tf.keras.layers.Dense(ssm_dim, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.gate_dense = tf.keras.layers.Dense(d_model, activation="sigmoid")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.dense1(x)
        h = self.drop(h, training=training)
        h = self.dense2(h)
        h = self.drop(h, training=training)
        gate = self.gate_dense(x)
        h = gate * h + (1 - gate) * x
        return self.norm(h)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "ssm_dim": self.ssm_dim, "dropout": self.dropout})
        return config


def find_default_model_path() -> str:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir.parent / "FED_Mamba" / "results" / "Multimodal_Mamba_Hybrid_Model1.keras",
        base_dir.parent / "FED_Mamba" / "results" / "Multimodal_Mamba_Federated_Model.keras",
        base_dir / "Multimodal_Mamba_Hybrid_Model1.keras",
        base_dir / "Multimodal_Mamba_Federated_Model.keras",
        base_dir / "Final(with aggressive augmentaion) with 200 samples" / "Final(with aggressive augmentaion) with 500 samples" / "Multimodal_Mamba_Hybrid_Model1.keras",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "MambaBlock": MambaBlock,
            "SimpleMambaBlock": SimpleMambaBlock,
            "MultimodalMambaBlock": MultimodalMambaBlock,
        },
        compile=False,
    )


def standardize_columns(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def extract_signal_from_mat(mat_data) -> np.ndarray:
    for key, value in mat_data.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if array.size == 0:
            continue
        array = np.squeeze(array)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim == 2:
            return array
        if array.ndim > 2:
            return array.reshape(array.shape[0], -1)
    raise ValueError("No usable ECG signal found inside the .mat file.")


def preprocess_ecg_file(file_path: str, target_samples: int) -> np.ndarray:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        mat_data = loadmat(file_path)
        signal = extract_signal_from_mat(mat_data)
    elif suffix in {".hea", ".dat", ""}:
        # If the provided path doesn't exist in the cwd, try to locate the record elsewhere
        record_base = str(path.with_suffix(""))
        if not Path(record_base + ".hea").exists() and not Path(record_base + ".dat").exists():
            # search workspace for matching record filename
            workspace_root = Path(__file__).resolve().parent
            found = None
            for p in workspace_root.rglob(path.name if path.suffix else path.name + "*"):
                if p.suffix.lower() in {".hea", ".dat"}:
                    # found a corresponding file; use its base (without extension)
                    found = str(p.with_suffix(""))
                    break
            if found:
                record_base = found
        signal, _ = wfdb.rdsamp(record_base)
    elif suffix == ".npy":
        signal = np.load(file_path)
    elif suffix in {".csv", ".txt"}:
        signal = np.loadtxt(file_path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported ECG file type: {suffix}")

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if signal.shape[1] < ECG_LEADS:
        pad_cols = ECG_LEADS - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_cols)), mode="constant")
    signal = signal[:, :ECG_LEADS]

    if signal.shape[0] >= target_samples:
        start = (signal.shape[0] - target_samples) // 2
        segment = signal[start : start + target_samples]
    else:
        pad_rows = target_samples - signal.shape[0]
        segment = np.pad(signal, ((0, pad_rows), (0, 0)), mode="edge")

    segment = standardize_columns(segment)
    return np.expand_dims(segment.astype(np.float32), axis=0)


def preprocess_pcg_file(file_path: str, target_height: int, target_width: int) -> np.ndarray:
    sample_rate, signal = wavfile.read(file_path)
    signal = signal.astype(np.float32)

    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    _, _, spectrogram = stft(signal, fs=sample_rate, nperseg=256)
    spectrogram = np.log(np.abs(spectrogram) + 1e-10)
    spectrogram = np.resize(spectrogram, (target_height, target_width)).astype(np.float32)

    max_abs = np.max(np.abs(spectrogram))
    if max_abs > 0:
        spectrogram = spectrogram / max_abs

    spectrogram = np.expand_dims(spectrogram, axis=-1)
    return np.expand_dims(spectrogram, axis=0)


def get_input_shapes_from_model(model):
    ecg_shape = model.inputs[0].shape
    pcg_shape = model.inputs[1].shape

    ecg_samples = ecg_shape[1]
    pcg_height = pcg_shape[1]
    pcg_width = pcg_shape[2]

    if ecg_samples is None:
        raise ValueError("Could not determine ECG sequence length from the loaded model.")
    if pcg_height is None or pcg_width is None:
        raise ValueError("Could not determine PCG image size from the loaded model.")

    return int(ecg_samples), int(pcg_height), int(pcg_width)


def predict_label(model, ecg_input: np.ndarray, pcg_input: np.ndarray):
    probs = model.predict([ecg_input, pcg_input], verbose=0)

    if probs.ndim == 2 and probs.shape[1] >= 2:
        abnormal_prob = float(probs[0, 1])
    else:
        abnormal_prob = float(probs.reshape(-1)[0])

    pred_label = "Abnormal" if abnormal_prob >= 0.5 else "Normal"
    confidence = abnormal_prob if pred_label == "Abnormal" else (1.0 - abnormal_prob)
    return pred_label, confidence, abnormal_prob


def main():
    st.set_page_config(page_title="Multimodal ECG-PCG Demo", page_icon="stethoscope", layout="wide")
    st.title("Multimodal Thesis Demo")
    st.write("Upload one ECG file and one PCG .wav file, then show direct output.")

    default_model = find_default_model_path()
    model_path = st.text_input("Model path (.keras)", value=default_model)

    st.subheader("Inputs")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("ECG can be .mat, .npy, .csv, .txt, .hea, or .dat.")
        uploaded_ecg = st.file_uploader("Upload ECG file", type=["mat", "npy", "csv", "txt", "hea", "dat"])
        ecg_record_path = st.text_input("Or enter WFDB record base path")

    with col2:
        st.caption("PCG must be a .wav file.")
        uploaded_pcg = st.file_uploader("Upload PCG .wav", type=["wav"])

    if st.button("Run Prediction", type="primary"):
        if not model_path or not os.path.exists(model_path):
            st.error("Valid model path is required.")
            return
        if uploaded_ecg is None and not ecg_record_path:
            st.error("Please upload an ECG file or enter a WFDB record path.")
            return
        if uploaded_pcg is None:
            st.error("Please upload a PCG .wav file.")
            return

        temp_ecg_path = None
        temp_pcg_path = None
        try:
            if uploaded_ecg is not None:
                temp_ecg_path = Path(uploaded_ecg.name)
                with open(temp_ecg_path, "wb") as f:
                    f.write(uploaded_ecg.getbuffer())
                ecg_source_path = str(temp_ecg_path)
            else:
                ecg_source_path = ecg_record_path

            temp_pcg_path = Path(uploaded_pcg.name)
            with open(temp_pcg_path, "wb") as f:
                f.write(uploaded_pcg.getbuffer())

            model = load_model(model_path)
            ecg_samples, pcg_height, pcg_width = get_input_shapes_from_model(model)
            ecg_input = preprocess_ecg_file(ecg_source_path, ecg_samples)
            pcg_input = preprocess_pcg_file(str(temp_pcg_path), pcg_height, pcg_width)
            pred_label, confidence, abnormal_prob = predict_label(model, ecg_input, pcg_input)

            st.success("Prediction completed.")
            st.metric("Predicted Class", pred_label)
            st.metric("Confidence", f"{confidence:.4f}")
            st.metric("Abnormal Probability", f"{abnormal_prob:.4f}")

            st.write("### Input Summary")
            st.write(f"ECG input shape: {ecg_input.shape}")
            st.write(f"PCG input shape: {pcg_input.shape}")
            st.write(f"Model ECG shape: {model.inputs[0].shape}")
            st.write(f"Model PCG shape: {model.inputs[1].shape}")

        except Exception as ex:
            st.error(f"Inference failed: {ex}")
        finally:
            if temp_ecg_path is not None and temp_ecg_path.exists():
                temp_ecg_path.unlink()
            if temp_pcg_path is not None and temp_pcg_path.exists():
                temp_pcg_path.unlink()


if __name__ == "__main__":
    main()
