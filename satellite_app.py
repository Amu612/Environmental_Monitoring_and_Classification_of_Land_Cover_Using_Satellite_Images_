import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Page configuration
st.set_page_config(
    page_title="EcoVision - Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: #f0f0f0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .upload-section h3 {
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Header
st.markdown("""
<div class="main-header">
    <h1>🛰️ EcoVision</h1>
    <p>Advanced Satellite Image Classification & Environmental Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🚀 Navigation")
    
    page = st.selectbox(
        "Choose a section:",
        ["🏠 Home", "📊 Analytics", "ℹ️ About", "📈 Model Info"]
    )
    
    st.markdown("---")
    
    # Model loading section
    st.markdown("## 🤖 Model Management")
    
    model_file = st.file_uploader(
        "Upload your trained model (.h5)",
        type=['h5'],
        help="Upload the Modelenv.v1.h5 file or similar trained model"
    )
    
    if model_file is not None:
        try:
            # Save uploaded file temporarily
            with open("temp_model.h5", "wb") as f:
                f.write(model_file.read())
            
            # Load model
            st.session_state.model = load_model("temp_model.h5")
            st.success("✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    # Quick stats
    if st.session_state.prediction_history:
        st.markdown("## 📊 Quick Stats")
        total_predictions = len(st.session_state.prediction_history)
        st.metric("Total Predictions", total_predictions)
        
        # Most common prediction
        predictions = [p['prediction'] for p in st.session_state.prediction_history]
        most_common = max(set(predictions), key=predictions.count)
        st.metric("Most Common Class", most_common)

# Main content based on selected page
if page == "🏠 Home":
    # Main prediction interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h3>📸 Upload Satellite Image</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a satellite image for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Satellite Image", use_column_width=True)
            
            # Prediction button
            if st.button("🔮 Analyze Image", key="predict_btn"):
                if st.session_state.model is not None:
                    with st.spinner("🔄 Analyzing image..."):
                        try:
                            # Preprocess image
                            img_resized = img.resize((255, 255))
                            img_array = image.img_to_array(img_resized)
                            img_array = np.expand_dims(img_array, axis=0)
                            img_array = img_array / 255.0
                            
                            # Make prediction
                            predictions = st.session_state.model.predict(img_array)
                            class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
                            predicted_class = class_names[np.argmax(predictions)]
                            confidence = float(np.max(predictions))
                            
                            # Store prediction
                            st.session_state.prediction_history.append({
                                'prediction': predicted_class,
                                'confidence': confidence,
                                'timestamp': pd.Timestamp.now()
                            })
                            
                            # Display prediction
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h2>🎯 Prediction Result</h2>
                                <h1>{predicted_class}</h1>
                                <p>Confidence: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence chart
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=class_names,
                                    y=predictions[0],
                                    marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe']
                                )
                            ])
                            fig.update_layout(
                                title="Prediction Confidence Scores",
                                xaxis_title="Land Cover Types",
                                yaxis_title="Confidence Score",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                else:
                    st.error("❌ Please upload a model first!")
    
    with col2:
        st.markdown("## 🌍 Land Cover Types")
        
        # Information cards for each class
        classes_info = {
            "☁️ Cloudy": "Areas covered by clouds in satellite imagery",
            "🏜️ Desert": "Arid regions with sparse vegetation",
            "🌱 Green Area": "Forests, grasslands, and vegetated regions",
            "🌊 Water": "Rivers, lakes, oceans, and water bodies"
        }
        
        for class_name, description in classes_info.items():
            st.markdown(f"""
            <div class="feature-card">
                <h4>{class_name}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Analytics":
    st.markdown("## 📈 Prediction Analytics")
    
    if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            class_counts = df['prediction'].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Prediction Distribution",
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence over time
            fig = px.line(
                df,
                x='timestamp',
                y='confidence',
                title="Prediction Confidence Over Time",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions table
        st.markdown("## 📋 Recent Predictions")
        recent_df = df.tail(10).sort_values('timestamp', ascending=False)
        st.dataframe(recent_df, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.experimental_rerun()
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>📊 No Analytics Data Yet</h3>
            <p>Upload and analyze some images to see analytics!</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "ℹ️ About":
    st.markdown("## 🌟 About EcoVision")
    
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Mission</h3>
        <p>EcoVision is an advanced satellite image classification system designed to monitor and analyze environmental changes using cutting-edge deep learning technology.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>🔧 Technology Stack</h3>
        <ul>
            <li><strong>Deep Learning:</strong> TensorFlow/Keras CNN Architecture</li>
            <li><strong>Frontend:</strong> Streamlit with Custom CSS</li>
            <li><strong>Visualization:</strong> Plotly for Interactive Charts</li>
            <li><strong>Image Processing:</strong> PIL and OpenCV</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>🌍 Applications</h3>
        <ul>
            <li>Environmental monitoring and conservation</li>
            <li>Urban planning and development</li>
            <li>Climate change research</li>
            <li>Agricultural assessment</li>
            <li>Water resource management</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "📈 Model Info":
    st.markdown("## 🤖 Model Information")
    
    if st.session_state.model is not None:
        # Model summary
        st.markdown("### 📋 Model Architecture")
        
        # Create a summary of the model
        model_info = {
            "Total Parameters": st.session_state.model.count_params(),
            "Input Shape": str(st.session_state.model.input_shape),
            "Output Classes": st.session_state.model.output_shape[-1],
            "Optimizer": "Adam",
            "Loss Function": "Categorical Crossentropy"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for key, value in list(model_info.items())[:3]:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>{key}</h4>
                    <p>{value}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for key, value in list(model_info.items())[3:]:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>{key}</h4>
                    <p>{value}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Model layers
        st.markdown("### 🏗️ Model Layers")
        layers_info = []
        for i, layer in enumerate(st.session_state.model.layers):
            layers_info.append({
                "Layer": i + 1,
                "Type": layer.__class__.__name__,
                "Output Shape": str(layer.output_shape),
                "Parameters": layer.count_params()
            })
        
        layers_df = pd.DataFrame(layers_info)
        st.dataframe(layers_df, use_container_width=True)
        
    else:
        st.markdown("""
        <div class="info-box">
            <h3>🤖 No Model Loaded</h3>
            <p>Please upload a model file to view detailed information!</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🛰️ EcoVision - Advanced Satellite Image Classification | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)