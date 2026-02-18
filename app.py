"""
Simplified Face Emotion Recognition Demo - Streamlit Application
Clean emotion detection without external AI integrations
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import utils
from utils.model_loader import ModelLoader
from utils.image_processor import ImageProcessor
from utils.face_detector import FaceDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Face Emotion Recognition Demo",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stAlert > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class FERApp:
    """Main Face Emotion Recognition Application"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.image_processor = ImageProcessor()
        self.face_detector = FaceDetector()
        self.emotion_history = []
        self.loaded_models = {}
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            'models': {},
            'model_info': {},
            'emotion_history': [],
            'comparison_mode': False,
            'selected_models': [],
            'current_image': None,
            'prediction_results': None,
            'input_method': 'Upload Image'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_sidebar(self):
        """Render sidebar with model selection and settings"""
        st.sidebar.title("Settings")
        
        # Model selection
        st.sidebar.subheader("Model Configuration")
        available_models = self.model_loader.get_available_models()
        
        if not available_models:
            st.sidebar.error("No models found! Please ensure .pth files are in the model/ directory.")
            return {}, 0.5, False
        
        # Multi-model comparison toggle
        comparison_mode = st.sidebar.checkbox(
            "Multi-Model Comparison",
            value=st.session_state.comparison_mode,
            help="Compare predictions from multiple models simultaneously"
        )
        st.session_state.comparison_mode = comparison_mode
        
        if comparison_mode:
            # Multi-select for models
            selected_models = st.sidebar.multiselect(
                "Select Models to Compare",
                options=list(available_models.keys()),
                default=st.session_state.selected_models,
                help="Choose multiple models for comparison"
            )
            st.session_state.selected_models = selected_models
            
            if len(selected_models) < 2:
                st.sidebar.warning("Please select at least 2 models for comparison.")
                return {}, 0.5, comparison_mode
        else:
            # Single model selection
            selected_model = st.sidebar.selectbox(
                "Select Model",
                options=list(available_models.keys()),
                help="Choose the emotion recognition model to use"
            )
            selected_models = [selected_model] if selected_model else []
        
        # Load models button
        if st.sidebar.button("Load Models", type="primary"):
            success_count = 0
            with st.spinner("Loading models..."):
                for model_name in selected_models:
                    try:
                        model, device = self.model_loader.load_model(model_name)
                        if model:
                            st.session_state.models[model_name] = model
                            st.session_state.model_info[model_name] = {
                                'name': model_name,
                                'device': str(device),
                                'loaded_at': datetime.now().strftime("%H:%M:%S")
                            }
                            success_count += 1
                    except Exception as e:
                        st.sidebar.error(f"Failed to load {model_name}: {e}")
            
            if success_count > 0:
                st.sidebar.success(f"Loaded {success_count} models successfully!")
            else:
                st.sidebar.error("No models loaded.")
        
        # Loaded models info
        if st.session_state.models:
            with st.sidebar.expander("Loaded Models", expanded=True):
                for model_name, info in st.session_state.model_info.items():
                    st.write(f"**{model_name}**")
                    st.write(f"• Device: {info['device']}")
                    st.write(f"• Loaded: {info['loaded_at']}")
        
        # Prediction settings
        st.sidebar.subheader("Prediction Settings")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence level for predictions"
        )
        
        # Performance settings
        st.sidebar.subheader("Performance Settings")
        enable_gpu = st.sidebar.checkbox("Enable GPU Acceleration", value=True)
        batch_inference = st.sidebar.checkbox("Batch Inference", value=False, 
                                            help="Process multiple models in parallel")
        
        return st.session_state.models, confidence_threshold, comparison_mode
    
    def run_multi_model_prediction(self, models, face_image, confidence_threshold):
        """Run prediction on multiple models and return comparison results"""
        results = {}
        
        def predict_single_model(model_name, model):
            try:
                return model_name, self.image_processor.predict_emotion(
                    model, face_image, confidence_threshold
                )
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                return model_name, None
        
        # Run predictions in parallel for better performance
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_model = {
                executor.submit(predict_single_model, name, model): name 
                for name, model in models.items()
            }
            
            for future in as_completed(future_to_model):
                model_name, result = future.result()
                if result:
                    results[model_name] = result
        
        return results
    
    def render_multi_model_comparison_charts(self, model_results):
        """Display simple multi-model comparison"""
        if not model_results:
            st.error("No valid predictions to compare")
            return
            
        # Simple confidence comparison
        st.subheader("Model Comparison")
        
        # Create comparison data
        models = list(model_results.keys())
        emotions = [result['predicted_emotion'] for result in model_results.values()]
        confidences = [result['confidence'] for result in model_results.values()]
        inference_times = [result['inference_time_ms'] for result in model_results.values()]
        
        # Confidence comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Confidence Levels:**")
            fig = go.Figure([
                go.Bar(
                    x=models,
                    y=confidences,
                    text=[f'{emotion}\\n{conf:.1%}' for emotion, conf in zip(emotions, confidences)],
                    textposition='auto',
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(models)]
                )
            ])
            fig.update_layout(
                title="Confidence Comparison",
                yaxis_title="Confidence",
                yaxis=dict(tickformat='.0%'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Processing Speed:**")
            fig2 = go.Figure([
                go.Bar(
                    x=models,
                    y=inference_times,
                    text=[f'{time:.1f}ms' for time in inference_times],
                    textposition='auto',
                    marker_color='lightcoral'
                )
            ])
            fig2.update_layout(
                title="Inference Time Comparison",
                yaxis_title="Time (ms)",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Agreement analysis
        unique_emotions = set(emotions)
        if len(unique_emotions) == 1:
            st.success(f"**Perfect Agreement!** All models predicted: {list(unique_emotions)[0]}")
        else:
            st.warning(f"**Models Disagree** - Different emotions detected: {', '.join(unique_emotions)}")
    
    def render_prediction_charts(self, prediction_results):
        """Render prediction visualization charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Confidence")
            emotions = list(prediction_results['all_predictions'].keys())
            confidences = list(prediction_results['all_predictions'].values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=emotions,
                    y=confidences,
                    marker_color=['#FF6B6B' if conf == max(confidences) else '#4ECDC4' 
                                 for conf in confidences],
                    text=[f'{c:.1%}' for c in confidences],
                    textposition='outside',
                )
            ])
            
            fig.update_layout(
                title="Emotion Confidence Scores",
                xaxis_title="Emotions",
                yaxis_title="Confidence",
                yaxis=dict(range=[0, max(confidences) * 1.2], tickformat='.0%'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Summary")
            
            # Simple metrics instead of charts
            predicted_emotion = prediction_results.get('predicted_emotion', 'Unknown')
            confidence = prediction_results.get('confidence', 0.0)
            
            st.metric("Primary Emotion", predicted_emotion)
            st.metric("Confidence Level", f"{confidence:.1%}")
            
            # Top 3 emotions list
            st.write("**Top 3 Emotions:**")
            top_emotions = sorted(
                prediction_results['all_predictions'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for i, (emotion, conf) in enumerate(top_emotions):
                st.write(f"{i+1}. {emotion.title()}: {conf:.1%}")
    
    def run(self):
        """Main application runner - Simplified Interface"""
        # Header
        st.title("Face Emotion Recognition")
        st.markdown("*Simple emotion detection with improved face detection*")
        
        # Sidebar
        models, confidence_threshold, comparison_mode = self.render_sidebar()
        
        if not models:
            st.warning("Please load at least one model from the sidebar to get started")
            return
        
        # Display loaded models info
        model_names = list(models.keys())
        if comparison_mode and len(model_names) > 1:
            st.info(f"**Comparison Mode**: {len(model_names)} models - {', '.join(model_names)}")
        else:
            st.info(f"**Single Model**: {model_names[0]}")
        
        # Input method selection
        st.subheader("Choose Input Method")
        input_method = st.radio(
            "How would you like to provide an image?",
            ["Upload Image", "Take Photo with Camera"],
            horizontal=True
        )
        
        # Handle input based on selection
        if input_method == "Upload Image":
            self.handle_file_upload()
        else:
            self.handle_camera_capture()
        
        # Analysis section
        if st.session_state.current_image is not None:
            self.show_image_analysis()
            
            # Prediction button
            if st.button("Run Emotion Prediction", type="primary", use_container_width=True):
                self.run_prediction(models, confidence_threshold, comparison_mode)
        
        # Results section
        if st.session_state.prediction_results:
            self.display_results()
        
        # Tips section
        with st.expander("Tips for Better Results"):
            st.markdown("""
            - **Good lighting**: Ensure face is well-lit
            - **Clear view**: Face should be unobstructed  
            - **Front-facing**: Avoid extreme angles
            - **Single face**: Works best with one clear face
            - **High resolution**: Better quality = better detection
            """)
    
    def handle_file_upload(self):
        """Handle image file upload"""
        uploaded_file = st.file_uploader(
            "Choose an image file:",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a clear image with a visible face"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image = self.image_processor.resize_image(image)
                st.session_state.current_image = image
                st.success("Image uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    def handle_camera_capture(self):
        """Handle camera photo capture"""
        st.write("**Camera Capture**")
        camera_input = st.camera_input("Take a photo:")
        
        if camera_input is not None:
            try:
                image = Image.open(camera_input)
                image = self.image_processor.resize_image(image)
                st.session_state.current_image = image
                st.success("Photo captured successfully!")
            except Exception as e:
                st.error(f"Error processing camera image: {e}")
    
    def show_image_analysis(self):
        """Show image and face detection analysis"""
        st.subheader("Image Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Input Image:**")
            st.image(st.session_state.current_image, caption="Selected Image", width=400)
        
        with col2:
            st.write("**Face Detection Analysis:**")
            
            # Analyze image for faces
            with st.spinner("Detecting faces..."):
                faces = self.face_detector.detect_faces(st.session_state.current_image)
            
            if faces:
                st.success(f"**{len(faces)} face(s) detected**")
                
                # Show face detection visualization
                annotated = self.face_detector.draw_face_boxes(st.session_state.current_image, faces)
                st.image(annotated, caption=f"Faces Detected", width=400)
                
                if len(faces) > 1:
                    st.info("Multiple faces detected. Will use the largest face for emotion analysis.")
                else:
                    st.info("Ready for emotion prediction!")
            else:
                st.warning("**No faces detected**")
                st.info("Try adjusting lighting or camera angle for better face detection.")
    
    def run_prediction(self, models, confidence_threshold, comparison_mode):
        """Run emotion prediction"""
        try:
            # Get the best face for prediction
            face_info = self.face_detector.get_largest_face(st.session_state.current_image)
            
            if not face_info:
                st.error("No face detected for analysis")
                return
            
            if not self.face_detector.is_valid_face(face_info, min_area=500):
                st.error("Face too small or unclear for reliable prediction")
                return
            
            with st.spinner("Analyzing emotions..."):
                if comparison_mode and len(models) > 1:
                    # Multi-model prediction
                    results = self.run_multi_model_prediction(models, face_info['face_image'], confidence_threshold)
                    
                    if results:
                        st.session_state.prediction_results = {
                            'type': 'multi_model',
                            'results': results,
                            'face_info': face_info,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        }
                        
                        # Add to history (use best model result)
                        best_result = max(results.items(), key=lambda x: x[1]['confidence'])
                        self.add_to_history(best_result[1], best_result[0])
                    else:
                        st.error("All model predictions failed")
                        
                else:
                    # Single model prediction
                    model_name = list(models.keys())[0]
                    model = models[model_name]
                    
                    result = self.image_processor.predict_emotion(
                        model, face_info['face_image'], confidence_threshold
                    )
                    
                    if result:
                        st.session_state.prediction_results = {
                            'type': 'single_model',
                            'result': result,
                            'model_name': model_name,
                            'face_info': face_info,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        }
                        
                        # Add to history
                        self.add_to_history(result, model_name)
                        
                    else:
                        st.error("Emotion prediction failed")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            logger.error(f"Prediction error: {e}")
    
    def add_to_history(self, result, model_name):
        """Add prediction result to history"""
        history_entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'predicted_emotion': result['predicted_emotion'],
            'confidence': result['confidence'],
            'inference_time_ms': result['inference_time_ms'],
            'model': model_name
        }
        st.session_state.emotion_history.append(history_entry)
    
    def display_results(self):
        """Display prediction results and analytics"""
        results = st.session_state.prediction_results
        
        st.subheader("Prediction Results")
        st.write(f"*Analysis completed at {results['timestamp']}*")
        
        if results['type'] == 'multi_model':
            self.display_multi_model_results(results)
        else:
            self.display_single_model_results(results)
        
        # Face analysis section
        self.display_face_analysis(results)
        
        # Session history
        if st.session_state.emotion_history:
            self.display_session_analytics()
    
    def display_single_model_results(self, results):
        """Display single model prediction results"""
        result = results['result']
        model_name = results['model_name']
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Predicted Emotion", 
                result['predicted_emotion'],
                help="The most likely emotion detected"
            )
        with col2:
            st.metric(
                "Confidence", 
                f"{result['confidence']:.1%}",
                help="How confident the model is in this prediction"
            )
        with col3:
            st.metric(
                "Processing Time", 
                f"{result['inference_time_ms']:.1f}ms",
                help="Time taken for analysis"
            )
        
        # Detailed probability chart
        st.write(f"**Detailed Analysis using {model_name}:**")
        self.render_prediction_charts(result)
    
    def display_multi_model_results(self, results):
        """Display clean multi-model comparison results"""
        model_results = results['results']
        
        # Best result summary
        best_model = max(model_results.items(), key=lambda x: x[1]['confidence'])
        st.info(f"**Best Result:** {best_model[1]['predicted_emotion']} "
               f"({best_model[1]['confidence']:.1%}) by {best_model[0]}")
        
        # Clean model comparison visualization
        self.render_multi_model_comparison_charts(model_results)
        
        # Simple comparison table
        st.write("**Model Comparison Results:**")
        comparison_data = []
        for model_name, result in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Predicted Emotion': result['predicted_emotion'],
                'Confidence': f"{result['confidence']:.1%}",
                'Processing Time': f"{result['inference_time_ms']:.1f}ms",
                'Status': "Above Threshold" if result['above_threshold'] else "Below Threshold"
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    def display_face_analysis(self, results):
        """Display clean face analysis section"""
        st.subheader("Face Analysis")
        
        face_info = results['face_info']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Analyzed Face Region:**")
            st.image(face_info['face_image'], caption="Face used for emotion detection", width=250)
            
        with col2:
            st.write("**Detection Details:**")
            st.write(f"• Face area: {face_info['area']:,} pixels")
            st.write(f"• Detection confidence: {face_info['confidence']:.2f}")
            st.write(f"• Face center: ({face_info['center'][0]}, {face_info['center'][1]})")
            
            # Show face location on original image
            annotated = self.face_detector.draw_face_boxes(
                st.session_state.current_image, 
                [face_info['bbox']]
            )
            st.image(annotated, caption="Face location in original image", width=300)
    
    def display_session_analytics(self):
        """Display session history and analytics"""
        st.subheader("Session Analytics")
        
        history_df = pd.DataFrame(st.session_state.emotion_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Emotion frequency distribution
            st.write("**Emotion Distribution:**")
            emotion_counts = history_df['predicted_emotion'].value_counts()
            fig = go.Figure([go.Pie(labels=emotion_counts.index, values=emotion_counts.values)])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average confidence by emotion
            st.write("**Average Confidence by Emotion:**")
            avg_conf = history_df.groupby('predicted_emotion')['confidence'].mean().sort_values(ascending=False)
            fig2 = go.Figure([go.Bar(x=avg_conf.index, y=avg_conf.values)])
            fig2.update_layout(height=300, yaxis=dict(tickformat='.0%'))
            st.plotly_chart(fig2, use_container_width=True)
        
        # Recent predictions table
        st.write("**Recent Predictions:**")
        display_cols = ['timestamp', 'predicted_emotion', 'confidence', 'inference_time_ms']
        if 'model' in history_df.columns:
            display_cols.insert(1, 'model')
        
        recent_df = history_df[display_cols].tail(5).copy()
        recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.1%}")
        recent_df['inference_time_ms'] = recent_df['inference_time_ms'].apply(lambda x: f"{x:.1f}ms")
        
        st.dataframe(recent_df, use_container_width=True)
        
        # Clear history button
        if st.button("Clear History", help="Clear all session data"):
            st.session_state.emotion_history = []
            st.success("History cleared!")
            st.experimental_rerun()


def main():
    """Application entry point"""
    try:
        app = FERApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()