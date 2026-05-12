from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from utils.predict_lstm import predict_technique  # pastikan fungsi predict sudah update

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder upload jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect(url_for("index"))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for("index"))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        relative_video_path = f"uploads/{filename}"

        label, confidence, quality, final_score, dtw_score, detail = predict_technique(filepath)

        return render_template("result.html",
            video_path=relative_video_path,
            label=label,
            confidence=confidence,
            quality=quality,
            final_score=final_score, 
            dtw_score=dtw_score,
            detail=detail
        )


        


@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
