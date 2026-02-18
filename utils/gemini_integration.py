"""
Gemini API Integration for Face Emotion Recognition
Provides emotion analysis and insights using Google's Gemini AI
"""

try:
    import google.genai as genai
except ImportError:
    try:
        import google.generativeai as genai
        print("Warning: Using deprecated google.generativeai. Please upgrade to google.genai")
    except ImportError:
        genai = None
        print("Warning: No Gemini library found. AI insights will be disabled.")

from typing import Dict, Optional, List
import logging
import json
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


class GeminiAnalyzer:
    """Gemini AI integration for emotion analysis and insights"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.is_configured = False
        
        if api_key:
            self.configure_gemini(api_key)
    
    def configure_gemini(self, api_key: str) -> bool:
        """Configure Gemini AI with simplified error handling"""
        if not genai:
            logger.error("Gemini library not available. Install: pip install google-generativeai")
            return False
            
        try:
            self.api_key = api_key.strip()
            
            # Use google.generativeai (standard approach)
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            
            # Simple connection test
            test_response = self.model.generate_content("Hi")
            if test_response and test_response.text:
                self.is_configured = True
                logger.info("Gemini AI connected successfully")
                return True
            else:
                logger.error("Gemini API returned empty response")
                return False
            
        except Exception as e:
            logger.error(f"Gemini connection failed: {str(e)}")
            self.is_configured = False
            return False
    
    def analyze_emotion_results(self, prediction_results: Dict) -> Optional[str]:
        """Generate simple emotion analysis using Gemini AI"""
        if not self.is_configured:
            return "Gemini AI not configured. Please add your API key."
        
        try:
            emotion = prediction_results.get('predicted_emotion', 'Unknown')
            confidence = prediction_results.get('confidence', 0.0)
            
            prompt = f"""Give a brief 2-3 sentence analysis of this emotion detection:
            
Emotion: {emotion}
Confidence: {confidence:.1%}
            
Explain what this emotion typically means and the confidence level."""
            
            response = self.model.generate_content(prompt)
            if response and response.text:
                return response.text
            else:
                return "Unable to generate analysis"
                
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}")
            return f"Analysis error: {str(e)[:50]}"
    
    def get_emotion_insights(self, emotion: str) -> Optional[str]:
        """Get general insights about a specific emotion"""
        if not self.is_configured:
            return "Gemini AI not configured. Please add your API key."
        
        try:
            prompt = f"""
            Provide insights about the emotion '{emotion}':
            
            1. What triggers this emotion?
            2. How is it typically expressed?
            3. What are the psychological benefits or purposes?
            4. How can someone manage or work with this emotion?
            5. Cultural variations in expression
            
            Keep it educational and helpful (max 250 words).
            """
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text
            else:
                return "Unable to generate insights at this time."
                
        except Exception as e:
            logger.error(f"Error getting emotion insights: {e}")
            return f"Error: {str(e)[:100]}"
    
    def analyze_emotion_patterns(self, emotion_history: List[Dict]) -> Optional[str]:
        """Analyze patterns in emotion detection history"""
        if not self.is_configured:
            return "Gemini AI not configured. Please add your API key."
        
        if not emotion_history:
            return "No emotion history available for analysis."
        
        try:
            # Prepare history data
            emotions = [item.get('predicted_emotion', 'Unknown') for item in emotion_history]
            confidences = [item.get('confidence', 0.0) for item in emotion_history[-10:]]  # Last 10
            
            # Count emotion frequencies
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            prompt = f"""
            Analyze this emotion detection session pattern:
            
            Total Detections: {len(emotions)}
            Emotion Frequencies: {json.dumps(emotion_counts, indent=2)}
            Recent Confidence Levels: {[round(c, 2) for c in confidences]}
            
            Please provide:
            1. Pattern analysis (most/least frequent emotions)
            2. Confidence trends
            3. Emotional state interpretation
            4. Potential factors influencing the results
            5. Suggestions for emotional awareness
            
            Be supportive and insightful (max 300 words).
            """
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text
            else:
                return "Unable to analyze patterns at this time."
                
        except Exception as e:
            logger.error(f"Error analyzing emotion patterns: {e}")
            return f"Error: {str(e)[:100]}"
    
    def get_model_comparison_insights(self, model_results: Dict) -> Optional[str]:
        """Analyze differences between model predictions"""
        if not self.is_configured:
            return "Gemini AI not configured. Please add your API key."
        
        try:
            prompt = f"""
            Compare these facial emotion recognition model results:
            
            {json.dumps(model_results, indent=2)}
            
            Analyze:
            1. Agreement/disagreement between models
            2. Which model seems most uncertain or confident
            3. Potential reasons for differences
            4. Which prediction might be most reliable
            5. Technical insights about model behavior
            
            Be technical but accessible (max 250 words).
            """
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text
            else:
                return "Unable to generate comparison at this time."
                
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            return f"Error: {str(e)[:100]}"
    
    def get_tips_and_suggestions(self) -> Optional[str]:
        """Get general tips for emotion recognition and AI"""
        if not self.is_configured:
            return "Gemini AI not configured. Please add your API key."
        
        try:
            prompt = """
            Provide helpful tips for using facial emotion recognition technology:
            
            1. Best practices for getting accurate results
            2. Lighting and camera positioning tips
            3. Understanding AI limitations
            4. Privacy and ethical considerations
            5. How to interpret confidence scores
            
            Make it practical and user-friendly (max 300 words).
            """
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text
            else:
                return "Unable to generate tips at this time."
                
        except Exception as e:
            logger.error(f"Error getting tips: {e}")
            return f"Error: {str(e)[:100]}"
    
    def test_connection(self) -> bool:
        """Test Gemini AI connection with current API key"""
        if not self.api_key:
            return False
        
        try:
            if not self.is_configured:
                return self.configure_gemini(self.api_key)
            
            # Simple test
            response = self.model.generate_content("Test")
            return bool(response and response.text)
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False