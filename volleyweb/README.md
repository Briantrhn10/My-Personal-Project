# 🏐 VolleyVision: Technical Overview & Implementation

Selamat datang di sub-direktori **VolleyVision**. Folder ini berisi seluruh kode sumber, model Machine Learning, dan aset web untuk sistem analisis teknik bola voli yang menggunakan integrasi **YOLOv8** dan **LSTM**.

## 🛠️ Arsitektur Sistem
Sistem ini bekerja dengan alur pipeline sebagai berikut:
1. **Input:** Video gerakan voli diunggah melalui antarmuka Flask.
2. **Preprocessing:** Video dipecah menjadi frame dan diproses oleh model **YOLOv8-Pose** untuk mengekstraksi 17 koordinat titik tubuh.
3. **Classification:** Model **LSTM** memprediksi jenis teknik berdasarkan pola urutan koordinat tersebut.

## 📊 Detail Metrik Penilaian
* **Stability Score:** Memantau keseimbangan pusat massa tubuh.
* **Speed Score:** Mengukur kecepatan perpindahan pergelangan tangan.
* **ROM Score:** Menilai rentang gerak horizontal.
* **DTW Distance:** Membandingkan gerakan dengan template ideal atlet profesional.

## 🚀 Cara Menjalankan
1. Masuk ke folder: `cd volleyweb`
2. Aktifkan venv: `venv\Scripts\activate`
3. Instalasi: `pip install -r requirements.txt`
4. Run: `python app.py`

---
**Author:** Brian Tarihoran
