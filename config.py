"""
Configuration settings for Face Emotion Recognition Demo
Contains app constants, model parameters, and default settings
"""

import os
from typing import Dict, List

# Application Settings
APP_CONFIG = {
    'title': 'Face Emotion Recognition Demo',
    'version': '1.0.0',
    'description': 'Real-time facial emotion detection using deep learning models',
    'author': 'AI Engineering Team',
    'license': 'MIT'
}

# Model Configuration
MODEL_CONFIG = {
    'input_size': 224,
    'num_classes': 7,
    'model_directory': 'model',
    'supported_formats': ['.pth'],
    'device': 'auto'  # 'auto', 'cpu', or 'cuda'
}

# Emotion Labels
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust', 
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

LABEL_TO_ID = {v.lower(): k for k, v in EMOTION_LABELS.items()}

# Emotion Color Mapping for Visualization
EMOTION_COLORS = {
    'Angry': '#FF4B4B',      # Red
    'Disgust': '#9B59B6',    # Purple
    'Fear': '#2E86AB',       # Blue
    'Happy': '#F39C12',      # Orange
    'Sad': '#3498DB',        # Light Blue
    'Surprise': '#E74C3C',   # Dark Red
    'Neutral': '#95A5A6'     # Gray
}

# Image Processing Settings
IMAGE_CONFIG = {
    'max_upload_size_mb': 10,
    'supported_formats': ['png', 'jpg', 'jpeg'],
    'resize_max_size': 512,
    'face_detection_min_size': (30, 30),
    'face_padding': 20,
    'default_confidence_threshold': 0.5
}

# Webcam Settings
WEBCAM_CONFIG = {
    'default_width': 640,
    'default_height': 480,
    'default_fps': 30,
    'detection_interval': 10,  # Process every N frames
    'max_fps_display': 15      # Limit display FPS for performance
}

# Face Detection Settings
FACE_DETECTION_CONFIG = {
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'min_size': (30, 30),
    'cascade_file': 'haarcascade_frontalface_default.xml',
    'quality_threshold': 0.3,
    'min_area': 2500
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'batch_size': 1,
    'precision': 'float32',
    'enable_gpu_optimization': True,
    'memory_optimization': True,
    'inference_timeout': 10.0  # seconds
}

# Gemini AI Settings
GEMINI_CONFIG = {
    'model_name': 'gemini-1.5-flash',
    'temperature': 0.7,
    'max_tokens': 500,
    'safety_settings': {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE'
    }
}

# UI Settings
UI_CONFIG = {
    'theme': 'light',
    'sidebar_width': 300,
    'chart_height': 400,
    'max_history_display': 50,
    'refresh_interval_ms': 100,
    'animation_duration': 500
}

# Logging Settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'fer_demo.log',
    'max_size_mb': 10,
    'backup_count': 3
}

# File Paths
PATHS = {
    'models': 'model',
    'logs': 'logs',
    'temp': 'temp',
    'cache': '.cache',
    'config': 'config'
}

# Model-specific configurations
MODEL_SPECIFIC_CONFIG = {
    'custom_cnn': {
        'architecture': 'Custom Deep CNN',
        'input_channels': 1,  # Grayscale
        'description': 'Custom 4-layer CNN with batch normalization',
        'preprocessing': 'grayscale_normalize'
    },
    'face_emotion': {
        'architecture': 'Vision Transformer',
        'input_channels': 3,  # RGB (converted from grayscale)
        'description': 'Pre-trained transformer for emotion detection',
        'preprocessing': 'grayscale_to_rgb_normalize',
        'huggingface_model': 'abhilash88/face-emotion-detection'
    },
    'convnext': {
        'architecture': 'ConvNeXt Tiny',
        'input_channels': 3,  # RGB (converted from grayscale)
        'description': 'Modern efficient CNN architecture',
        'preprocessing': 'grayscale_to_rgb_normalize',
        'huggingface_model': 'facebook/convnext-tiny-224'
    }
}

# Error Messages
ERROR_MESSAGES = {
    'model_not_loaded': 'Please load a model first',
    'no_face_detected': 'No face detected in the image',
    'low_confidence': 'Prediction confidence is below threshold',
    'invalid_image': 'Invalid image format or corrupted file',
    'webcam_not_available': 'Webcam is not available or already in use',
    'model_load_failed': 'Failed to load the selected model',
    'gemini_api_error': 'Error communicating with Gemini AI',
    'file_too_large': 'File size exceeds the maximum allowed limit'
}

# Success Messages
SUCCESS_MESSAGES = {
    'model_loaded': 'Model loaded successfully',
    'prediction_success': 'Emotion prediction completed',
    'webcam_started': 'Webcam session started',
    'gemini_connected': 'Gemini AI connected successfully',
    'analysis_complete': 'AI analysis completed'
}

# Default Values
DEFAULTS = {
    'confidence_threshold': 0.5,
    'image_size': 224,
    'batch_size': 1,
    'temperature': 0.7,
    'max_faces': 10,
    'history_limit': 100
}

# Feature Flags
FEATURES = {
    'enable_webcam': True,
    'enable_gemini': True,
    'enable_face_detection': True,
    'enable_batch_processing': False,
    'enable_model_comparison': True,
    'enable_emotion_history': True,
    'enable_performance_metrics': True
}

def get_model_path(model_name: str) -> str:
    """Get full path to model file"""
    return os.path.join(PATHS['models'], f"{model_name}.pth")

def get_available_models() -> List[str]:
    """Get list of available model names"""
    if not os.path.exists(PATHS['models']):
        return []
    
    models = []
    for file in os.listdir(PATHS['models']):
        if file.endswith('.pth'):
            models.append(file.replace('.pth', ''))
    
    return models

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check if model directory exists
        if not os.path.exists(PATHS['models']):
            print(f"Warning: Model directory '{PATHS['models']}' not found")
            return False
        
        # Check if models are available
        available_models = get_available_models()
        if not available_models:
            print("Warning: No models found in model directory")
            return False
        
        print(f"Configuration valid. Found {len(available_models)} model(s)")
        return True
        
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False

# Initialize directories if they don't exist
def init_directories():
    """Create necessary directories"""
    for path in PATHS.values():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    # Validate configuration when run directly
    init_directories()
    if validate_config():
        print("Configuration is valid!")
        print(f"Available models: {get_available_models()}")
    else:
        print("Configuration validation failed!")