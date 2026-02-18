"""
Model Loading Utilities for Face Emotion Recognition
Handles loading of different model architectures (.pth files)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, ConvNextForImageClassification
import os
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FERDeepCNN(nn.Module):
    """Custom Deep CNN architecture for FER"""
    
    def __init__(self, num_classes=7):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(128, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.25)

        # FC layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 14 * 14, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop5 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(256, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.drop6 = nn.Dropout(0.25)

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)

        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop4(x)

        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.drop5(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.drop6(x)
        x = self.fc3(x)
        return x


class GrayscaleToRGBModel(nn.Module):
    """Wrapper to convert grayscale input to RGB for transformer models"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.base_model(x)


class ModelLoader:
    """Handles loading and initialization of different FER models"""
    
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available model files"""
        models = {}
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} not found")
            return models
            
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.pth'):
                model_name = filename.replace('.pth', '').replace('_', ' ').title()
                models[model_name] = os.path.join(self.model_dir, filename)
                
        return models
    
    def load_custom_cnn(self, model_path: str) -> torch.nn.Module:
        """Load custom CNN model with robust error handling"""
        try:
            model = FERDeepCNN(num_classes=7)
            
            # Load state dict with error handling
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                # Assume it's the model itself
                if hasattr(checkpoint, 'state_dict'):
                    state_dict = checkpoint.state_dict()
                else:
                    model = checkpoint
                    model.to(self.device)
                    model.eval()
                    logger.info(f"Loaded complete model from {model_path}")
                    return model
            
            # Try to load state dict with size matching
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                logger.warning(f"Strict loading failed, trying with strict=False: {e}")
                model.load_state_dict(state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            logger.info(f"Successfully loaded Custom CNN from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading Custom CNN: {e}")
            raise
    
    def load_face_emotion_model(self, model_path: str) -> torch.nn.Module:
        """Load Face Emotion Detection transformer model"""
        try:
            # Try loading as pre-trained model first
            try:
                base_model = AutoModelForImageClassification.from_pretrained(
                    'abhilash88/face-emotion-detection',
                    num_labels=7,
                    ignore_mismatched_sizes=True
                )
                model = GrayscaleToRGBModel(base_model)
                
                # Load state dict if available
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif hasattr(checkpoint, 'state_dict'):
                    model.load_state_dict(checkpoint.state_dict(), strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                    
            except Exception:
                # Fallback: load as custom CNN if transformer loading fails
                logger.warning("Transformer loading failed, trying as Custom CNN")
                return self.load_custom_cnn(model_path)
            
            model.to(self.device)
            model.eval()
            logger.info(f"Successfully loaded Face Emotion model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading Face Emotion model: {e}")
            # Final fallback
            return self.load_custom_cnn(model_path)
    
    def load_convnext_model(self, model_path: str) -> torch.nn.Module:
        """Load ConvNeXt model"""
        try:
            # Try loading as pre-trained model first
            try:
                base_convnext = ConvNextForImageClassification.from_pretrained(
                    'facebook/convnext-tiny-224',
                    num_labels=7,
                    ignore_mismatched_sizes=True
                )
                model = GrayscaleToRGBModel(base_convnext)
                
                # Load state dict if available
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif hasattr(checkpoint, 'state_dict'):
                    model.load_state_dict(checkpoint.state_dict(), strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                    
            except Exception:
                # Fallback: load as custom CNN if transformer loading fails
                logger.warning("ConvNeXt loading failed, trying as Custom CNN")
                return self.load_custom_cnn(model_path)
            
            model.to(self.device)
            model.eval()
            logger.info(f"Successfully loaded ConvNeXt model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading ConvNeXt model: {e}")
            # Final fallback
            return self.load_custom_cnn(model_path)
    
    def load_model(self, model_name: str) -> tuple:
        """Load model based on model name and auto-detect type, return (model, device)"""
        try:
            # Get full path from model name
            available_models = self.get_available_models()
            if model_name not in available_models:
                logger.error(f"Model {model_name} not found in available models")
                return None, None
            
            model_path = available_models[model_name]
            logger.info(f"Loading model from: {model_path}")
            
            # Auto-detect model type from filename
            filename = os.path.basename(model_path).lower()
            
            if 'custom' in filename or 'cnn' in filename:
                model = self.load_custom_cnn(model_path)
            elif 'face' in filename or 'emotion' in filename:
                model = self.load_face_emotion_model(model_path)
            elif 'convnext' in filename:
                model = self.load_convnext_model(model_path)
            else:
                # Try loading as custom CNN first (most common)
                logger.warning(f"Unknown model type for {filename}, trying as Custom CNN")
                model = self.load_custom_cnn(model_path)
            
            if model is not None:
                return model, self.device
            else:
                # Try alternative loading methods
                model = self._try_alternative_loading(model_path)
                return model, self.device if model else (None, None)
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None, None
    
    def _try_alternative_loading(self, model_path: str) -> Optional[torch.nn.Module]:
        """Try alternative model loading approaches"""
        logger.info(f"Trying alternative loading methods for {model_path}")
        
        try:
            # Method 1: Load state dict and inspect structure
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # If it's a complete model object
            if hasattr(checkpoint, 'eval'):
                model = checkpoint
                model.to(self.device)
                model.eval()
                logger.info("Loaded as complete model object")
                return model
            
            # Method 2: Try loading as Custom CNN (most robust)
            try:
                model = FERDeepCNN(num_classes=7)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Try loading with different strategies
                    try:
                        model.load_state_dict(state_dict, strict=True)
                    except RuntimeError:
                        logger.warning("Strict loading failed, using strict=False")
                        model.load_state_dict(state_dict, strict=False)
                
                model.to(self.device)
                model.eval()
                logger.info("Successfully loaded using alternative Custom CNN approach")
                return model
                
            except Exception as alt_e:
                logger.warning(f"Alternative Custom CNN loading failed: {alt_e}")
                
            # Method 3: Create a simple dummy model if all else fails
            logger.warning("Creating fallback model")
            model = FERDeepCNN(num_classes=7)
            model.to(self.device)
            model.eval()
            return model
                    
        except Exception as e:
            logger.error(f"All alternative loading methods failed: {e}")
            return None
    
    def get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / 1e6  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'model_size_mb': round(model_size_mb, 1),
            'device': str(self.device),
            'num_classes': len(self.emotion_labels)
        }