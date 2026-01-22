# 🫀 Cardiac Arrhythmia Detection System

A full-stack AI application that detects cardiac arrhythmia from ECG signals using
deep learning and machine learning techniques.

This project demonstrates real-world medical AI deployment with
proper validation, error handling, and explainability.

---

## 🚀 Features
- ECG signal analysis using MIT-BIH dataset
- Hybrid Deep Learning + Random Forest model
- Beat-level arrhythmia classification
- Automatic ECG plot generation
- PDF medical report generation
- Robust validation for low-quality ECG signals
- Flask-based web application

---

## 🧠 Tech Stack
- **Backend:** Python, Flask
- **ML/DL:** TensorFlow, Keras, Scikit-Learn
- **Signal Processing:** WFDB, NeuroKit2
- **Visualization:** Matplotlib
- **Deployment:** Render
- **Dataset:** MIT-BIH Arrhythmia Database

---

## ⚙️ How It Works
1. User uploads an ECG ZIP file (`.dat`, `.hea`, `.atr`)
2. System validates ECG quality
3. Heartbeats (R-peaks) are detected
4. Features are extracted from ECG signals
5. Hybrid model predicts arrhythmia class
6. Risk level is assigned
7. ECG plot and PDF report are generated

---

## ⚠️ Important Note (Medical Safety)
If ECG quality is insufficient or heartbeats cannot be detected reliably,
the system **safely rejects the input** instead of producing an incorrect diagnosis.

This behavior reflects real-world medical AI safety standards.

---

## 🖥️ Running Locally

pip install -r requirements.txt
python app.py
Open browser:


http://127.0.0.1:5000
🌐 Deployment
The application is deployed on Render and runs in a production-like environment.

📌 Why This Project Matters
Demonstrates end-to-end AI system design

Shows understanding of real-world ML limitations

Focuses on correctness and safety over blind predictions

Suitable for healthcare-oriented AI applications




---


