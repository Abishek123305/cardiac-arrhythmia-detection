**ECG Arrhythmia Detection Web Application**

This project is a Flask-based web application for automated ECG (Electrocardiogram) arrhythmia detection. It leverages a hybrid deep learning architecture combining CNN, Bi-LSTM, and BR-SquareNet for heartbeat-level feature extraction, followed by a Meta Random Forest classifier for robust recording-level diagnosis.

The system supports end-to-end ECG analysis, from raw signal preprocessing to risk assessment and PDF report generation.

---

🔍 **Key Features**

**ECG Record Analysis**

Accepts ECG recordings as ZIP files containing .dat, .hea, and .atr files (MIT-BIH format).

**Hybrid Deep Learning Pipeline**

CNN for spatial and morphological feature extraction.  
Bi-LSTM for capturing temporal dependencies in ECG signals.  
BR-SquareNet for residual and deep feature refinement.

**Meta Random Forest Classifier**

Aggregates beat-level predictions into a final recording-level diagnosis.

**Risk Stratification**

Normal, Low Risk, High Risk, Critical, or Uncertain.

**ECG Visualization & Reporting**

Generates ECG plots with detected R-peaks highlighted and a downloadable PDF report.

**REST API Support**

API endpoint available for integration with external systems.

---

🧪 **Preprocessing Pipeline**

Raw ECG signals undergo multiple preprocessing stages:

**Signal Filtering**

Bandpass Filter: 4th-order Butterworth (0.5–40 Hz).  
High-pass Filter: 2nd-order Butterworth (0.5 Hz).

**Normalization**

Z-score normalization:  
(x − μ) / σ

**Beat Segmentation**

R-peaks detected / annotated.  
Fixed window segmentation (280 samples: 90 before, 190 after R-peak).

**AAMI Class Mapping**

Beat annotations mapped to 5 AAMI classes:  
N – Normal  
S – Supraventricular  
V – Ventricular  
F – Fusion  
Q – Unknown / Paced

---

📊 **Evaluation Metrics**

Accuracy – Overall performance  
F1-Score – Macro & weighted  
Confusion Matrix – Absolute & normalized  
ROC & AUC – Class-wise performance  
Training Curves – Accuracy/Loss over epochs

---

🛠️ **Technology Stack**

Backend: Flask (Python)  
Signal Processing: NumPy, SciPy, WFDB, NeuroKit2  
Machine Learning: TensorFlow (Keras), Scikit-Learn, Joblib  
Visualization: Matplotlib  
Reporting: ReportLab

---

📁 **Project Structure**

app.py – Flask application entry point  
model.py – Core ECG inference logic  
model_pipeline.py – ECG processing pipeline  
utils/pdf_report.py – PDF report generation  
templates/index.html – Frontend UI  
static/ – CSS/JS  
uploads/ – Uploaded ZIPs  
results/plots/ – ECG graphs  
results/reports/ – PDF reports  
models/best_final_hybrid.h5  
meta_rf/meta_rf.pkl  
requirements.txt – Python dependencies

---

⚙️ **Setup & Installation**

Clone repository  
git clone <your-repo-url>

Create virtual environment  
python -m venv venv  
venv\Scripts\activate

Install dependencies  
pip install -r requirements.txt

Ensure model files exist:  
models/best_final_hybrid.h5  
meta_rf/meta_rf.pkl

---

▶️ **Usage**

Run application:  
python app.py

Open in browser:  
http://127.0.0.1:5000

Upload ZIP ECG record  
View prediction, confidence, risk score  
Download PDF report

---

🔌 **API Endpoint – POST /analyze**

Uploads and analyzes ECG ZIP.  
Request: multipart/form-data with file (ZIP).  
Response example:

{ "prediction": "Normal", "confidence": 0.92, "risk": "Low", "plot": "ecg_plot.png", "pdf": "report.pdf" }

---

🤖 **Models Used**

Hybrid Deep Learning (CNN + Bi-LSTM + BR-SquareNet)  
Meta Random Forest classifier

