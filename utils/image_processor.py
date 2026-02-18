"""
Image Processing Utilities for Face Emotion Recognition
Handles image transformations, predictions, and inference timing
"""

import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import time
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image preprocessing and prediction inference"""
    
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # For grayscale
        ])
        
        self.emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
        # Emotion colors for visualization
        self.emotion_colors = {
            'Angry': '#FF4B4B',      # Red
            'Disgust': '#9B59B6',    # Purple
            'Fear': '#2E86AB',       # Blue
            'Happy': '#F39C12',      # Orange
            'Sad': '#3498DB',        # Light Blue
            'Surprise': '#E74C3C',   # Dark Red
            'Neutral': '#95A5A6'     # Gray
        }
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model inference"""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Apply transformations
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict_emotion(self, model: torch.nn.Module, image: Image.Image, 
                       confidence_threshold: float = 0.5) -> Dict:
        """Predict emotion from image with timing"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Inference timing
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(input_tensor)
                
                # Handle different model output formats
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Get predictions
            probs = probabilities.cpu().numpy()[0]
            predicted_class = np.argmax(probs)
            confidence = float(probs[predicted_class])
            
            # Create prediction results
            predictions = {}
            for i, emotion in self.emotion_labels.items():
                predictions[emotion] = float(probs[i])
            
            # Sort by confidence
            sorted_predictions = dict(sorted(predictions.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            result = {
                'predicted_emotion': self.emotion_labels[predicted_class],
                'confidence': confidence,
                'all_predictions': sorted_predictions,
                'inference_time_ms': round(inference_time, 2),
                'above_threshold': confidence >= confidence_threshold
            }
            
            logger.info(f"Prediction: {result['predicted_emotion']} "
                       f"({result['confidence']:.2f}) in {result['inference_time_ms']}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def resize_image(self, image: Image.Image, max_size: int = 512) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image
    
    def numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        try:
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image)
            else:
                # Grayscale
                return Image.fromarray(image, mode='L')
        except Exception as e:
            logger.error(f"Error converting numpy to PIL: {e}")
            raise
    
    def get_prediction_chart_data(self, predictions: Dict[str, float]) -> Tuple[List[str], List[float], List[str]]:
        """Prepare data for prediction chart"""
        emotions = list(predictions.keys())
        confidences = list(predictions.values())
        colors = [self.emotion_colors.get(emotion, '#95A5A6') for emotion in emotions]
        
        return emotions, confidences, colors
    
    def format_inference_stats(self, inference_time: float, 
                             model_info: Dict) -> Dict[str, str]:
        """Format inference statistics for display"""
        fps = 1000 / inference_time if inference_time > 0 else 0
        
        return {
            'Inference Time': f"{inference_time:.2f} ms",
            'FPS (approx)': f"{fps:.1f}",
            'Device': model_info.get('device', 'Unknown'),
            'Model Size': f"{model_info.get('model_size_mb', 0):.1f} MB",
            'Parameters': f"{model_info.get('total_parameters', 0):,}"
        }


class WebcamProcessor:
    """Handles webcam frame processing"""
    
    def __init__(self):
        self.frame_processor = ImageProcessor()
    
    def process_frame(self, frame: np.ndarray) -> Image.Image:
        """Process webcam frame"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            # Resize for display
            image = self.frame_processor.resize_image(image, max_size=640)
            return image
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            raise
    
    def get_webcam_settings(self) -> Dict[str, int]:
        """Get optimal webcam settings"""
        return {
            'width': 640,
            'height': 480,
            'fps': 30
        }