# Face Emotion Recognition App

Aplikasi web sederhana untuk mendeteksi emosi dari wajah menggunakan Streamlit dan PyTorch.

## Fitur Utama

- **Upload Gambar**: Upload foto untuk analisis emosi
- **Kamera Langsung**: Ambil foto menggunakan kamera
- **Multi-Model**: Bandingkan hasil dari beberapa model sekaligus
- **Analytics**: Lihat riwayat dan statistik prediksi

## Cara Menjalankan Aplikasi

### 1. Persiapan
Pastikan Python 3.8+ sudah terinstall di komputer Anda.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Siapkan Model
Letakkan file model (.pth) di folder `model/`:
- `custom_cnn_model.pth`
- `face_emotion_model.pth` 
- `convnext_model.pth`

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## Cara Menggunakan

1. **Pilih Model**: Di sidebar, pilih model yang ingin digunakan
2. **Load Model**: Klik tombol "Load Models" 
3. **Input Gambar**: Upload foto atau gunakan kamera
4. **Analisis**: Klik "Run Emotion Prediction"
5. **Hasil**: Lihat prediksi emosi dan tingkat confidence

## Struktur Project

```
StudyJam_FerDemo/
├── app.py                 # Aplikasi utama
├── requirements.txt       # Dependencies  
├── model/                # File model (.pth)
└── utils/                # Helper modules
    ├── model_loader.py   # Loading model
    ├── face_detector.py  # Deteksi wajah
    └── image_processor.py # Processing gambar
```

## Troubleshooting

**Model tidak bisa di-load:**
- Pastikan file .pth ada di folder `model/`
- Cek nama file sesuai dengan yang ada di kode

**Wajah tidak terdeteksi:**
- Pastikan pencahayaan cukup
- Wajah harus terlihat jelas
- Coba angle yang berbeda

## Support

Aplikasi ini dibuat untuk tujuan edukasi. Jika ada masalah, pastikan dependencies terinstall dengan benar dan file model tersedia.

- Python 3.8 or higher
- Webcam (optional, for real-time detection)
- GPU (optional, for faster inference)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd StudyJam_FerDemo
   ```

2. **Create virtual environment**
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   # Option 1: Use the enhanced launcher (recommended)
   python launch.py
   
   # Option 2: Direct Streamlit run
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The app should automatically open in your default browser

## Setup Guide

### Platform-Specific Instructions

#### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Clone and setup project
git clone <repository-url>
cd StudyJam_FerDemo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Clone and setup project
git clone <repository-url>
cd StudyJam_FerDemo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

#### Windows
```powershell
# Install Python from python.org or Microsoft Store

# Clone and setup project
git clone <repository-url>
cd StudyJam_FerDemo
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### GPU Support (Optional)

For faster inference with NVIDIA GPUs:

1. **Install CUDA** (11.8 or 12.1 recommended)
2. **Install PyTorch with CUDA support**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## Configuration

### Model Files
Ensure your trained model files (.pth) are placed in the `model/` directory:
```
StudyJam_FerDemo/
├── model/
│   ├── custom_cnn_model.pth
│   ├── face_emotion_model.pth
│   └── convnext_model.pth
└── ...
```

## Usage

### 1. Image Upload Mode
- Select a model from the sidebar
- Upload an image (PNG, JPG, JPEG)
- View detected faces and emotion predictions
- Analyze confidence scores and performance metrics

### 2. Real-time Camera Mode
- Ensure webcam is connected and accessible
- Click "Start Camera" to begin real-time detection
- Adjust confidence threshold for sensitivity
- Use "Capture Frame" to save interesting results

### 3. Analytics and Insights
- View session history and patterns
- Access AI-powered emotion insights via Gemini
- Compare model performance and accuracy
- Track inference speed and system performance

## Project Structure

```
StudyJam_FerDemo/
├── app.py                     # Main Streamlit application
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── model/                    # Pre-trained model files
│   ├── custom_cnn_model.pth
│   ├── face_emotion_model.pth
│   └── convnext_model.pth
└── utils/                    # Utility modules
    ├── __init__.py
    ├── model_loader.py       # Model loading and initialization
    ├── image_processor.py    # Image processing and prediction
    ├── face_detector.py      # Face detection utilities
    └── gemini_integration.py # Gemini AI integration
```

## API Reference

### Model Loader (`utils/model_loader.py`)
- `ModelLoader`: Main class for loading different model architectures
- `get_available_models()`: List available .pth files
- `load_model()`: Load specific model by type
- `get_model_info()`: Get model statistics and information

### Image Processor (`utils/image_processor.py`)
- `ImageProcessor`: Handle image preprocessing and prediction
- `predict_emotion()`: Run emotion inference with timing
- `preprocess_image()`: Transform images for model input
- `get_prediction_chart_data()`: Prepare data for visualization

### Face Detector (`utils/face_detector.py`)
- `FaceDetector`: OpenCV-based face detection
- `detect_faces()`: Locate faces in images
- `extract_faces()`: Extract face regions with padding
- `draw_face_boxes()`: Annotate images with detection boxes

### Gemini Integration (`utils/gemini_integration.py`)
- `GeminiAnalyzer`: AI-powered emotion analysis
- `analyze_emotion_results()`: Get insights on predictions
- `get_emotion_insights()`: Learn about specific emotions
- `analyze_emotion_patterns()`: Understand session patterns

## Troubleshooting

### Common Issues

1. **Library Conflicts (macOS)**
   ```
   objc[xxx]: Class AVFFrameReceiver is implemented in both...
   ```
   **Solution**: Use the launcher script which handles these warnings:
   ```bash
   python launch.py
   ```
   Or set environment variable:
   ```bash
   export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
   streamlit run app.py
   ```

2. **Gemini API Deprecation Warning**
   ```
   FutureWarning: All support for the google.generativeai package has ended
   ```
   **Solution**: Install the new package:
   ```bash
   pip install google-genai
   ```

3. **Model Loading Errors**
   - Ensure .pth files are in the `model/` directory
   - Check that model files are not corrupted
   - Verify sufficient RAM/VRAM for model loading
   - Try loading models one at a time first

4. **Webcam Not Working**
   - Grant camera permissions to your browser/Python
   - Ensure no other applications are using the webcam
   - Try different camera indices if multiple cameras are available
   - Check webcam drivers on Windows

5. **Low Performance**
   - Use GPU acceleration if available
   - Disable multi-model comparison for real-time use
   - Reduce webcam resolution or detection interval
   - Close other resource-intensive applications

6. **Streamlit Deprecation Warnings**
   These are fixed in the latest version. If you see them, update Streamlit:
   ```bash
   pip install --upgrade streamlit
   ```

### Performance Tips

- **GPU Usage**: Install CUDA-enabled PyTorch for faster inference
- **Memory Management**: Use launcher script for optimized settings
- **Model Selection**: Custom CNN is fastest, transformers are more accurate
- **Real-time Settings**: Adjust detection interval (5-30 frames) for performance
- **Multi-Model Mode**: Use for accuracy comparison, not real-time applications

### Logs and Debugging

The application logs important information to the console. For debugging:

1. **Increase Log Level**: Modify `LOGGING_CONFIG` in `config.py`
2. **Check Console**: Monitor terminal output for errors
3. **Test Components**: Run individual utility modules to isolate issues

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone for development
git clone <repository-url>
cd StudyJam_FerDemo

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/

# Code formatting
black .
isort .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FER-2013 dataset for emotion recognition training
- HuggingFace for pre-trained transformer models
- OpenCV for computer vision capabilities
- Streamlit for the web application framework
- Google Gemini AI for intelligent insights

## Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Include system information, error messages, and steps to reproduce

## Version History

### v1.1.0 (Latest)
- **Multi-Model Comparison**: Compare up to 3 models simultaneously
- **Enhanced Performance**: Optimized webcam processing with reduced latency
- **Fixed Deprecation Warnings**: Updated for latest Streamlit and libraries
- **Improved Error Handling**: Better webcam error management and recovery
- **Enhanced Launcher**: Smart dependency checking and environment setup
- **Library Conflict Resolution**: Automatic handling of macOS OpenCV/PyAV conflicts
- **Updated Gemini Integration**: Support for new google-genai package
- **Performance Monitoring**: Real-time FPS and detailed inference statistics
- **Better UI**: Cleaner interface with improved model selection

### v1.0.0
- Initial release with single model support
- Basic real-time webcam detection
- Gemini AI integration
- Performance monitoring
- Session analytics

---

**Happy Emotion Detection!** 🎭

## Key Improvements in This Version

✅ **Fixed all deprecation warnings** (Streamlit, Plotly, OpenCV)  
✅ **Multi-model comparison** - Compare 3 models simultaneously  
✅ **Enhanced webcam performance** - Smoother real-time detection  
✅ **Better error handling** - Robust webcam and model error management  
✅ **Library conflict resolution** - Automated handling of macOS conflicts  
✅ **Improved launcher script** - Smart environment setup and checks  
✅ **Updated dependencies** - Latest package versions with compatibility  
✅ **Performance optimizations** - Reduced latency and better FPS control