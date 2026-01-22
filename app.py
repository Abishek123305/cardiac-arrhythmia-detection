from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import zipfile
import uuid

from model import predict_and_plot_record
from utils.pdf_report import generate_pdf

# ---------------- CONFIG ----------------
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")
REPORT_DIR = os.path.join(RESULT_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

app = Flask(__name__)

# ---------------- RISK MAPPING ----------------
def risk_info(label):
    return {
        "N": ("Normal", "#2ecc71"),
        "S": ("Low Risk", "#f39c12"),
        "V": ("High Risk", "#e74c3c"),
        "F": ("Critical", "#8e0000"),
        "Q": ("Uncertain", "#7f8c8d")
    }.get(label, ("Unknown", "#7f8c8d"))

# ---------------- FRONTEND ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- ANALYZE ECG ----------------
@app.route("/analyze", methods=["POST"])
def analyze():

    # 1️⃣ Check file upload
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".zip"):
        return jsonify({"error": "Upload ZIP containing .dat + .hea + .atr"}), 400

    # 2️⃣ Create working directory
    uid = str(uuid.uuid4())
    workdir = os.path.join(UPLOAD_DIR, uid)
    os.makedirs(workdir, exist_ok=True)

    zip_path = os.path.join(workdir, "ecg.zip")
    file.save(zip_path)

    # 3️⃣ Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(workdir)

    # 4️⃣ Find .hea file (RECURSIVE – VERY IMPORTANT)
    record = None
    record_base = None

    for root, _, files in os.walk(workdir):
        for f in files:
            if f.endswith(".hea"):
                record = f.replace(".hea", "")
                record_base = root
                break

    if not record:
        return jsonify({"error": "Missing .hea file in ZIP"}), 400

    # 5️⃣ Model inference (SAFE)
    try:
        label, conf, plot_file = predict_and_plot_record(
            rec_id=record,
            base_path=record_base,
            save_dir=PLOT_DIR
        )
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "hint": "ECG quality too low or no detectable heartbeats"
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Unexpected processing error",
            "details": str(e)
        }), 500

    # 6️⃣ Risk mapping
    risk, color = risk_info(label)

    # 7️⃣ Generate PDF
    pdf_name = generate_pdf(
        record,
        label,
        conf,
        risk,
        os.path.join(PLOT_DIR, plot_file),
        REPORT_DIR
    )

    # 8️⃣ Send result
    return jsonify({
        "prediction": label,
        "confidence": float(conf),
        "risk": risk,
        "color": color,
        "plot": plot_file,
        "pdf": pdf_name
    })

# ---------------- FILE SERVING ----------------
@app.route("/results/plots/<path:filename>")
def serve_plots(filename):
    return send_from_directory(PLOT_DIR, filename)

@app.route("/results/reports/<path:filename>")
def serve_reports(filename):
    return send_from_directory(REPORT_DIR, filename)

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
