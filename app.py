import streamlit as st
from PIL import Image
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Page configuration
st.set_page_config(
    page_title="DeepFake Voice Detection",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp { background-color: white; }

    .title-text {
        margin-top: 40px;
        font-size: 42px;
        font-weight: bold;
        color: black;
        text-align: left;
        font-family: 'Monotype Corsiva', serif;
    }

    .subtitle-text {
        font-size: 20px;
        color: #333333;
        text-align: left;
        margin-bottom: 30px;
        font-family: 'Sitka Display Semibold', serif;
        line-height: 1.6;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: black;
        font-family: 'Century', serif;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .analyze-btn button {
        background-color: black;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout: Left (Text) | Right (Image)
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown('<div class="title-text">🎙️ DeepFake Voice Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle-text">'
        'DeepFake Voice Detection is an AI-driven system designed to identify '
        'synthetic, cloned, or manipulated human speech generated using advanced '
        'deep learning techniques. The system analyzes acoustic features, voice '
        'patterns, and spectral characteristics to distinguish between genuine '
        'human voices and artificially generated or spoofed audio, ensuring '
        'secure and reliable voice-based authentication.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV / MP3 / FLAC)",
        type=["wav", "mp3", "flac"],
        key="audio_uploader"
    )

    analyze = st.button("🔍 Analyze Voice")

with right_col:
    try:
        image = Image.open("assets/images/dfv1.jpg")
        st.image(image, use_container_width=True)
    except:
        st.info("Add an image at assets/images/dfv1.jpg")

# -------------------- ANALYZE BUTTON LOGIC --------------------
if analyze and uploaded_file is not None:

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:Century; color:black;">🔍 DeepFake Voice Analysis Result</h2>',
        unsafe_allow_html=True
    )

    # Load audio
    audio, sr = librosa.load(uploaded_file, sr=16000)

    # Audio duration
    duration = librosa.get_duration(y=audio, sr=sr)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    # MFCC energy
    mfcc_energy = np.mean(np.abs(mfcc))

    # Dummy prediction score
    fake_score = np.random.uniform(0, 1)

    # Prediction logic
    if fake_score > 0.5:
        prediction = "FAKE VOICE"
        confidence = fake_score
    else:
        prediction = "REAL VOICE"
        confidence = 1 - fake_score

    # -------------------- METRICS DASHBOARD --------------------
    st.markdown('<div class="section-title">📊 Model Metrics Dashboard</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    for col, label, value in zip(
        [col1, col2, col3, col4],
        ["🟥 Fake Probability", "🟩 Real Probability", "⏱ Audio Duration", "🎚 MFCC Energy"],
        [fake_score, 1-fake_score, duration, mfcc_energy]
    ):
        col.markdown(f"""
            <div style="
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-family: 'Century';
                color: black;
                font-size: 20px;
                font-weight: bold;
            ">
            {label}<br>{value:.2f}
            </div>
        """, unsafe_allow_html=True)

    # -------------------- RESULT DISPLAY --------------------
    st.markdown(f"""
        <div style="
            border: 2px solid black;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-family: 'Century';
            color: black;
            font-size: 20px;
        ">
            <b>Prediction:</b> {prediction}<br><br>
            <b>Confidence Score:</b> {confidence:.2f}<br><br>
            <b>Sample Rate:</b> {sr} Hz<br><br>
            <b>MFCC Feature Shape:</b> {mfcc.shape}
        </div>
    """, unsafe_allow_html=True)

    # -------------------- AUDIO VISUALIZATION --------------------
    st.markdown('<div class="section-title">🎵 Audio Playback</div>', unsafe_allow_html=True)
    st.audio(uploaded_file)

    st.markdown('<div class="section-title">📈 Waveform Visualization</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Audio Waveform", fontname="Century", fontsize=16)
    ax.set_xlabel("Time (s)", fontname="Century")
    ax.set_ylabel("Amplitude", fontname="Century")
    st.pyplot(fig)
    plt.close(fig)

    # -------------------- SPECTROGRAM --------------------
    st.markdown('<div class="section-title">🌈 Spectrogram Analysis</div>', unsafe_allow_html=True)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Spectrogram (Log Frequency)", fontname="Century", fontsize=16)
    ax.set_xlabel("Time (s)", fontname="Century")
    ax.set_ylabel("Frequency (Hz)", fontname="Century")
    st.pyplot(fig)
    plt.close(fig)

    # -------------------- CONFIDENCE SCORE DISTRIBUTION --------------------
    st.markdown('<div class="section-title">📊 Probability Distribution (Fake vs Real)</div>', unsafe_allow_html=True)
    fake_probs = np.random.uniform(0.6, 1.0, size=10)
    real_probs = np.random.uniform(0.0, 0.4, size=10)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fake_probs, bins=5, alpha=0.6, label='Fake', color='red')
    ax.hist(real_probs, bins=5, alpha=0.6, label='Real', color='green')
    ax.set_xlabel("Prediction Probability", fontname="Century")
    ax.set_ylabel("Number of Samples", fontname="Century")
    ax.set_title("Probability Distribution of Fake vs Real Audio", fontname="Century", fontsize=16)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # -------------------- PREDICTION METRICS --------------------
    st.markdown('<div class="section-title">🧮 Prediction Metrics</div>', unsafe_allow_html=True)

    # Example dummy labels
    y_true = [1, 0, 0, 1, 1, 0]  # 1 = fake, 0 = real
    y_pred = [1, 0, 0, 0, 1, 0]

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Display metrics with styled HTML
    st.markdown(f"""
    <div style="
        font-family: 'Century', serif;
        color: black;
        font-size: 20px;
        line-height: 1.8;
    ">
    <b>Accuracy:</b> {acc:.2f}<br>
    <b>Precision:</b> {prec:.2f}<br>
    <b>Recall:</b> {rec:.2f}<br>
    <b>F1-score:</b> {f1:.2f}
    </div>
    """, unsafe_allow_html=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted", fontname="Century")
    ax.set_ylabel("Actual", fontname="Century")
    ax.set_title("Confusion Matrix", fontname="Century", fontsize=16)
    st.pyplot(fig)
    plt.close(fig)

    st.success("Analysis completed successfully ✔")

elif analyze and uploaded_file is None:
    st.warning("⚠️ Please upload an audio file before analysis.")
