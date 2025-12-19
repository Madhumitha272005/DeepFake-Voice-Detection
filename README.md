🎙️ DeepFake Voice Detection using AI & Machine Learning

🔍 Project Overview
DeepFake Voice Detection is an AI/ML-based system designed to identify whether an uploaded audio sample is Real or AI-Generated (DeepFake). With the rapid advancement of voice cloning and text-to-speech models, synthetic audio has become highly realistic, posing serious threats in fraud detection, identity theft, misinformation, and cybersecurity 🛡️.
This project detects such fake voices by analyzing low-level and high-level audio features that reveal subtle inconsistencies in synthetic speech 🎧📊.

🎯 Aim of the Project
The main goal is to build an intelligent, reliable, and scalable voice authentication system that can:
✅ Detect DeepFake audio
✅ Improve trust in voice-based systems
✅ Assist in forensic and security applications 🔐

🧠 How It Works
1️⃣ Audio Input – Users upload a voice sample in audio format 🎧
2️⃣ Feature Extraction – Important audio features such as MFCCs, pitch, energy, and spectral features are extracted using Librosa 📊
3️⃣ Model Training – Machine learning models like Random Forest, SVM, and Neural Networks are trained to learn patterns of real vs fake voices 🤖
4️⃣ Prediction – The trained model predicts whether the uploaded audio is Real or DeepFake with confidence 🎯

🧠 Feature Extraction (Core of the Project)
To accurately classify voices, the following audio features are extracted using Librosa 📚:
🎼 MFCC (Mel-Frequency Cepstral Coefficients)
Represents the human auditory perceptio
Captures timbre and vocal tract characteristics
DeepFake voices often show unnatural MFCC distributions
📈 Pitch (Fundamental Frequency – F0)
Measures voice frequency variations
Fake voices usually have flat or irregular pitch patterns
🔊 Energy / RMS
Indicates loudness and intensity of speech
Synthetic audio lacks natural energy fluctuations

🌈 Spectral Features
Spectral Centroid – brightness of sound
Spectral Bandwidth – frequency spread
Spectral Roll-off – high-frequency cutoff
These features help detect artificial frequency artifacts present in DeepFake audio

🤖 Machine Learning Algorithms Used
🌳 Random Forest
Handles complex, non-linear patterns
Robust against overfitting
Provides high accuracy for classification

📐 Support Vector Machine (SVM)
Finds an optimal boundary between real and fake voices
Effective in high-dimensional feature spaces

🧠 Neural Network (Optional / Extended)
Learns deep patterns from extracted features
Useful for future scalability and improvement

🛠️ Technologies Used
🐍 Python – Core programming language
📚 Librosa – Audio signal processing
🔢 NumPy & Pandas – Data handling
🤖 Machine Learning Algorithms – Classification models

🌐 Streamlit – Interactive web interface
