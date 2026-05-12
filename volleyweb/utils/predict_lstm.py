import numpy as np
import cv2
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from types import SimpleNamespace

# ==========================
# KONFIGURASI
# ==========================
MAX_FRAMES = 190
NUM_KEYPOINTS = 17
KEYPOINTS_DIM = 3
IMAGE_WIDTH = 640
MODEL_PATH = "volley_lstm_model.h5"
IDEAL_TEMPLATE_DIR = "templates"
CLASS_NAMES = ["passing atas", "passing bawah", "service atas", "service bawah", "smash"]

# ==========================
# LOAD MODEL
# ==========================
model = load_model(MODEL_PATH)
pose_model = YOLO("yolov8n-pose.pt")

# ==========================
# UTILITAS KEYPOINT
# ==========================
def normalize_keypoints(kp):
    kp[:, :, :2] /= IMAGE_WIDTH
    return kp

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pose_model(frame)

        if len(results) == 0 or results[0].keypoints is None:
            all_keypoints.append(np.zeros((NUM_KEYPOINTS, KEYPOINTS_DIM)))
            continue

        keypoints = results[0].keypoints.xy.cpu().numpy()
        if keypoints.ndim == 3:
            keypoints = keypoints[0]

        if keypoints.shape[1] == 2:
            keypoints = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))

        padded = np.zeros((NUM_KEYPOINTS, KEYPOINTS_DIM))
        padded[:min(NUM_KEYPOINTS, keypoints.shape[0])] = keypoints[:NUM_KEYPOINTS]
        all_keypoints.append(padded)

    cap.release()
    return np.array(all_keypoints) if all_keypoints else None

# ==========================
# FUNGSI EVALUASI
# ==========================
def calculate_dtw_distance(seq1, seq2):
    s1 = [frame.flatten() for frame in seq1]
    s2 = [frame.flatten() for frame in seq2]
    distance, _ = fastdtw(s1, s2, dist=euclidean)
    return distance

def calculate_stability_score(kp):
    left = kp[:, 5, :2]
    right = kp[:, 6, :2]

    valid_mask = np.logical_and(~np.all(left == 0, axis=1), ~np.all(right == 0, axis=1))
    valid_count = np.sum(valid_mask)

    if valid_count < 5:
        print("Frame valid terlalu sedikit.")
        return 0

    center = (left[valid_mask] + right[valid_mask]) / 2.0
    std = np.std(center, axis=0)
    mean_std = np.mean(std)

    max_std = 0.2  # bisa ditingkatkan jika terlalu sensitif
    score = max(0, 100 - (mean_std / max_std) * 100)
    return min(score, 100)


def calculate_speed_score(kp):
    # Coba pergelangan tangan kanan (index 10)
    wrist = kp[:, 10, :2]

    # Jika semua nol, coba tangan kiri (index 9)
    if np.all(wrist == 0):
        wrist = kp[:, 9, :2]
        if np.all(wrist == 0):
            return 0  # Keduanya tidak terdeteksi

    # Hitung pergerakan antar frame
    deltas = np.linalg.norm(np.diff(wrist, axis=0), axis=1)

    # Jika hanya 1 frame valid, tidak bisa hitung kecepatan
    if len(deltas) == 0:
        return 0

    avg_speed = np.mean(deltas)
    max_speed = 0.05  # bisa kamu sesuaikan berdasarkan eksperimen
    score = min((avg_speed / max_speed) * 100, 100)
    return score

def calculate_rom_score(kp):
    wrist_x = kp[:, 10, 0]
    if np.all(wrist_x == 0):
        return 0
    rom = np.max(wrist_x) - np.min(wrist_x)
    max_rom = 0.5
    return min((rom / max_rom) * 100, 100)

# ==========================
# PREDIKSI UTAMA
# ==========================
def predict_technique(video_path):
    keypoints = extract_keypoints_from_video(video_path)
    if keypoints is None:
        return "Tidak Terdeteksi", 0.0, "Tidak Terbaca", None, None, None

    keypoints = normalize_keypoints(keypoints)

    if keypoints.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - keypoints.shape[0], NUM_KEYPOINTS, KEYPOINTS_DIM))
        keypoints = np.vstack((keypoints, pad))
    else:
        keypoints = keypoints[:MAX_FRAMES]

    input_data = keypoints.reshape(1, MAX_FRAMES, NUM_KEYPOINTS * KEYPOINTS_DIM)
    prediction = model.predict(input_data)
    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    label = CLASS_NAMES[class_idx]

    if confidence < 0.5:
        return "Tidak Dikenali", confidence, "-", None, None, None

    # DTW Score
    template_path = os.path.join(IDEAL_TEMPLATE_DIR, label.replace(" ", "_") + ".npy")
    if os.path.exists(template_path):
        ideal = np.load(template_path)
        dtw_score = calculate_dtw_distance(keypoints, ideal)
        dtw_score_scaled = min(dtw_score / 800, 1.0) * 100
    else:
        dtw_score = None
        dtw_score_scaled = 100

    # Komponen
    stability = calculate_stability_score(keypoints)
    speed = calculate_speed_score(keypoints)
    rom = calculate_rom_score(keypoints)

    # Skor final
    confidence_score = confidence * 100
    evaluasi_score = (stability + speed + rom + (100 - dtw_score_scaled)) / 4
    final_score = (confidence_score * 0.5) + (evaluasi_score * 0.5)

    if final_score >= 85:
        quality = "Sangat Bagus"
    elif final_score >= 80:
        quality = "Bagus"
    elif final_score >= 50:
        quality = "Lumayan"
    else:
        quality = "Tidak Bagus"

    detail = SimpleNamespace(
        avg_speed=speed,
        stability_score=stability,
        rom_score=rom,
        total_score=final_score
    )

    return label, confidence, quality, final_score, dtw_score, detail
