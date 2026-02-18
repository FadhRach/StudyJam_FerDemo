"""
Face Detection Utilities for Face Emotion Recognition
Handles face detection and extraction from images and video streams
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using OpenCV Haar Cascades"""
    
    def __init__(self):
        try:
            # Initialize face cascade classifier
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade classifier")
            
            # More sensitive detection parameters for better face detection
            self.scale_factor = 1.05  # Reduced for better detection
            self.min_neighbors = 3   # Reduced for more sensitive detection
            self.min_size = (20, 20) # Smaller minimum size
            self.max_size = (500, 500) # Add maximum size limit
            
            logger.info("Face detector initialized successfully with enhanced parameters")
            
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            raise
    
    def detect_faces(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in PIL Image with multiple approaches for better detection
        Returns list of bounding boxes (x, y, width, height)
        """
        try:
            # Convert PIL to numpy array
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply histogram equalization for better contrast
            gray_eq = cv2.equalizeHist(gray)
            
            faces = []
            
            # Try multiple detection approaches
            detection_params = [
                # Default parameters
                {
                    'scaleFactor': self.scale_factor,
                    'minNeighbors': self.min_neighbors,
                    'minSize': self.min_size,
                    'maxSize': self.max_size
                },
                # More sensitive detection
                {
                    'scaleFactor': 1.03,
                    'minNeighbors': 2,
                    'minSize': (15, 15),
                    'maxSize': self.max_size
                },
                # Less sensitive but more accurate
                {
                    'scaleFactor': 1.2,
                    'minNeighbors': 6,
                    'minSize': (40, 40),
                    'maxSize': self.max_size
                }
            ]
            
            # Try detection on both original and equalized image
            for gray_img in [gray, gray_eq]:
                for params in detection_params:
                    detected = self.face_cascade.detectMultiScale(
                        gray_img,
                        scaleFactor=params['scaleFactor'],
                        minNeighbors=params['minNeighbors'],
                        minSize=params['minSize'],
                        maxSize=params['maxSize'],
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(detected) > 0:
                        faces.extend(detected.tolist())
                        break  # Stop if we found faces
                
                if faces:  # If we found faces, no need to try more
                    break
            
            # Remove duplicate faces (overlapping detections)
            if len(faces) > 1:
                faces = self._remove_duplicate_faces(faces)
            
            logger.info(f"Detected {len(faces)} faces in image")
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _remove_duplicate_faces(self, faces: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping face detections"""
        if len(faces) <= 1:
            return faces
        
        # Calculate overlap between faces and keep only non-overlapping ones
        filtered_faces = []
        
        for face in faces:
            x1, y1, w1, h1 = face
            
            is_duplicate = False
            for existing_face in filtered_faces:
                x2, y2, w2, h2 = existing_face
                
                # Calculate intersection area
                intersection_x = max(x1, x2)
                intersection_y = max(y1, y2)
                intersection_w = max(0, min(x1 + w1, x2 + w2) - intersection_x)
                intersection_h = max(0, min(y1 + h1, y2 + h2) - intersection_y)
                intersection_area = intersection_w * intersection_h
                
                # Calculate union area
                union_area = w1 * h1 + w2 * h2 - intersection_area
                
                # If overlap is significant, consider it a duplicate
                if intersection_area / union_area > 0.3:  # 30% overlap threshold
                    # Keep the larger face
                    if w1 * h1 > w2 * h2:
                        # Remove the existing smaller face and add the current larger one
                        filtered_faces.remove(existing_face)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def extract_faces(self, image: Image.Image, 
                     padding: int = 20) -> List[Dict]:
        """
        Extract face regions from image with padding
        Returns list of face info dicts
        """
        try:
            faces = self.detect_faces(image)
            extracted_faces = []
            
            img_width, img_height = image.size
            
            for i, (x, y, w, h) in enumerate(faces):
                # Add padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_width, x + w + padding)
                y2 = min(img_height, y + h + padding)
                
                # Extract face region
                face_img = image.crop((x1, y1, x2, y2))
                
                # Calculate face area and confidence score
                face_area = w * h
                confidence_score = min(1.0, face_area / (100 * 100))  # Normalize by 100x100
                
                face_info = {
                    'face_id': i,
                    'bbox': (x, y, w, h),
                    'bbox_padded': (x1, y1, x2-x1, y2-y1),
                    'face_image': face_img,
                    'area': face_area,
                    'confidence': confidence_score,
                    'center': (x + w//2, y + h//2)
                }
                
                extracted_faces.append(face_info)
            
            # Sort by face area (largest first)
            extracted_faces.sort(key=lambda x: x['area'], reverse=True)
            
            logger.info(f"Extracted {len(extracted_faces)} faces from image")
            return extracted_faces
            
        except Exception as e:
            logger.error(f"Error extracting faces: {e}")
            return []
    
    def draw_face_boxes(self, image: Image.Image, 
                       faces: List[Tuple], 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> Image.Image:
        """Draw bounding boxes around detected faces"""
        try:
            # Convert PIL to numpy for drawing
            img_array = np.array(image.convert('RGB'))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(img_array, f'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
            
            # Convert back to PIL
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.error(f"Error drawing face boxes: {e}")
            return image
    
    def get_largest_face(self, image: Image.Image) -> Optional[Dict]:
        """Get the largest detected face from image"""
        faces = self.extract_faces(image)
        return faces[0] if faces else None
    
    def is_valid_face(self, face_info: Dict, 
                     min_area: int = 800) -> bool:  # Reduced from 2500 to 800
        """Check if detected face meets quality criteria"""
        return (
            face_info['area'] >= min_area and
            face_info['confidence'] >= 0.2  # Reduced from 0.3 to 0.2
        )
    
    def get_detection_stats(self, faces: List[Dict]) -> Dict:
        """Get face detection statistics"""
        if not faces:
            return {
                'total_faces': 0,
                'valid_faces': 0,
                'avg_area': 0,
                'avg_confidence': 0
            }
        
        valid_faces = [f for f in faces if self.is_valid_face(f)]
        
        return {
            'total_faces': len(faces),
            'valid_faces': len(valid_faces),
            'avg_area': int(np.mean([f['area'] for f in faces])),
            'avg_confidence': round(np.mean([f['confidence'] for f in faces]), 3),
            'largest_face_area': faces[0]['area'] if faces else 0
        }


class RealtimeFaceDetector:
    """Real-time face detection for video streams"""
    
    def __init__(self):
        self.detector = FaceDetector()
        self.detection_interval = 5  # Process every N frames for performance
        self.frame_count = 0
        self.last_faces = []
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process video frame for face detection
        Returns annotated frame and face information
        """
        try:
            self.frame_count += 1
            
            # Convert frame to PIL Image for detection
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
            else:
                pil_image = Image.fromarray(frame, mode='L')
            
            # Detect faces every N frames for performance
            if self.frame_count % self.detection_interval == 0:
                self.last_faces = self.detector.extract_faces(pil_image)
            
            # Draw bounding boxes
            annotated_frame = frame.copy()
            for face in self.last_faces:
                x, y, w, h = face['bbox']
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Face {face["face_id"]}', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return annotated_frame, self.last_faces
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []
    
    def reset(self):
        """Reset detector state"""
        self.frame_count = 0
        self.last_faces = []