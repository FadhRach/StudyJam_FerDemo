# Face Emotion Recognition App 🎭

Aplikasi web untuk mendeteksi emosi dari wajah menggunakan Streamlit dan PyTorch. Upload gambar atau gunakan kamera untuk analisis emosi real-time.

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/username/StudyJam_FerDemo.git
cd StudyJam_FerDemo
```

### 2. Setup berdasarkan OS Anda

#### 🍎 macOS
```bash
# Install Python (jika belum ada)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python

# Setup project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 🐧 Linux (Ubuntu/Debian)
```bash
# Install Python
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Setup project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 🪟 Windows
```powershell
# Download Python dari python.org (jika belum ada)

# Setup project
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Siapkan Model Files
**PENTING**: Model files (.pth) tidak termasuk di repository karena ukuran yang besar.

Buat folder `model/` dan letakkan file model Anda:
```
StudyJam_FerDemo/
├── model/
│   ├── custom_cnn_model.pth
│   ├── face_emotion_model.pth
│   └── convnext_model.pth
└── ...
```

### 4. Jalankan Aplikasi
```bash
# Aktivasi virtual environment (jika belum)
# macOS/Linux: source venv/bin/activate
# Windows: venv\Scripts\activate

# Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di browser: `http://localhost:8501`

## 📱 Fitur Utama

- ✅ **Upload Gambar**: Analisis emosi dari foto
- ✅ **Kamera Real-time**: Deteksi emosi langsung dari webcam
- ✅ **Multi-Model**: Bandingkan hasil dari berbagai model AI
- ✅ **Face Detection**: Deteksi wajah otomatis dengan OpenCV
- ✅ **Confidence Score**: Lihat tingkat kepercayaan prediksi

## 🎯 Cara Menggunakan

1. **Pilih Model** di sidebar (minimal 1 model)
2. **Load Models** dengan klik tombol "Load Selected Models"
3. **Input Gambar**: Upload foto atau gunakan kamera
4. **Analisis**: Klik "Analyze Emotion" 
5. **Hasil**: Lihat prediksi emosi dan confidence score

## 📂 Struktur Project

```
StudyJam_FerDemo/
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Dependencies Python
├── model/                 # Model files (.pth) - tidak di-commit
└── utils/                 # Helper modules
    ├── model_loader.py    # Loading model PyTorch
    ├── face_detector.py   # Deteksi wajah OpenCV
    └── image_processor.py # Processing gambar
```

## 🔧 Troubleshooting

### Model tidak bisa di-load
```
❌ Error: No such file or directory: 'model/xxx_model.pth'
```
**Solusi**: Pastikan file .pth ada di folder `model/`

### Library conflicts (macOS)
```
❌ objc: Class AVFFrameReceiver is implemented in both...
```
**Solusi**: Set environment variable:
```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
streamlit run app.py
```

### Webcam tidak berfungsi
**Solusi**: 
- Berikan permission kamera ke browser/Python
- Tutup aplikasi lain yang menggunakan kamera
- Restart browser jika perlu

### Error import dependencies
**Solusi**: Install ulang requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## ⚡ Performance Tips

- **GPU**: Install PyTorch dengan CUDA untuk inference lebih cepat
- **Real-time**: Gunakan 1 model untuk webcam real-time
- **Compare**: Gunakan multi-model untuk analisis gambar static

## 📋 Requirements

- Python 3.8+
- Webcam (optional, untuk real-time detection)
- Model files (.pth) - train sendiri atau download dari sumber terpercaya
- 4GB+ RAM (8GB+ recommended untuk multi-model)

## 🐛 Issues & Support

Jika menemukan masalah:
1. Cek bagian troubleshooting di atas
2. Pastikan semua dependencies terinstall
3. Verifikasi model files ada di folder yang benar
4. Buat issue di repository dengan detail error

---

**Selamat mencoba! 🎉** 

Aplikasi ini dibuat untuk tujuan edukasi. Model files tidak disertakan - Anda perlu melatih atau mendapatkan model sendiri.