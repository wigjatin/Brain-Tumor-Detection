import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('model.h5')
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon=":brain:",
    layout="centered"
)

st.markdown("""
<style>
    .header {
        color: #1E90FF;
        text-align: center;
        font-size: 2.5rem !important;
    }
    .section-title {
        color: #4169E1;
        border-bottom: 2px solid #1E90FF;
        padding-bottom: 0.3rem;
    }
    .result-box {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .no-tumor {
        background-color: #E6F4EA;
        border: 2px solid #34A853;
    }
    .has-tumor {
        background-color: #FCE8E6;
        border: 2px solid #EA4335;
    }
    .confidence {
        background-color: #E8F0FE;
        border: 2px solid #4285F4;
    }
    .github-badge {
        display: inline-block;
        background: #24292e;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        text-decoration: none;
    }
    .interpretation-box {
        background-color: #F8F9FA;
        border-left: 4px solid #4285F4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_image(image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_tumor(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    return class_labels[predicted_class_index], confidence

st.markdown('<h1 class="header">Brain Tumor Detection</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    Upload an MRI scan to detect potential brain tumors. The model can identify:
    <ul style="text-align: left; display: inline-block;">
        <li><span style="color: #EA4335; font-weight: bold;">Glioma</span> tumors</li>
        <li><span style="color: #FBBC05; font-weight: bold;">Meningioma</span> tumors</li>
        <li><span style="color: #4285F4; font-weight: bold;">Pituitary</span> tumors</li>
        <li><span style="color: #34A853; font-weight: bold;">No tumor</span> cases</li>
    </ul>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an MRI scan...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", width=None)
    
    with st.spinner("🔍 Analyzing the MRI scan..."):
        tumor_type, confidence = predict_tumor(image)
        
    st.markdown('<h3 class="section-title">Analysis Results</h3>', unsafe_allow_html=True)
    
    if tumor_type == 'notumor':
        st.markdown('<div class="result-box no-tumor">'
                    '<h3>No Tumor Detected!</h3>'
                    '<p>Your MRI scan shows no signs of brain tumors.</p>'
                    '</div>', unsafe_allow_html=True)
        st.balloons()
    else:
        tumor_colors = {
            'glioma': '#EA4335',
            'meningioma': '#FBBC05',
            'pituitary': '#4285F4'
        }
        st.markdown(f'<div class="result-box has-tumor">'
                    f'<h3>⚠️ Potential Tumor Detected: <span style="color: {tumor_colors.get(tumor_type, "#000")}">{tumor_type.capitalize()}</span></h3>'
                    f'<p>Please consult a medical professional for further evaluation.</p>'
                    '</div>', unsafe_allow_html=True)
    
    confidence_color = "#34A853" if confidence > 0.9 else "#FBBC05" if confidence > 0.7 else "#EA4335"
    st.markdown('<div class="result-box confidence">'
                f'<h3>Confidence Level: <span style="color: {confidence_color}">{confidence*100:.2f}%</span></h3>'
                f'<div style="background: #E0E0E0; border-radius: 5px; height: 20px; margin: 10px 0;">'
                f'<div style="background: {confidence_color}; width: {confidence*100}%; height: 100%; border-radius: 5px;"></div>'
                '</div>'
                '</div>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="section-title">Interpretation Guide</h3>', unsafe_allow_html=True)
    
    interpretation = {
        'glioma': """
            <div class="interpretation-box">
                <h4>Glioma Tumor</h4>
                <ul>
                    <li>Most common type of primary brain tumor</li>
                    <li>Arises from glial cells that support nerve cells</li>
                    <li>Can range from low-grade (slow-growing) to high-grade (aggressive)</li>
                    <li>Treatment typically involves surgery, radiation, and chemotherapy</li>
                </ul>
            </div>
            """,
        'meningioma': """
            <div class="interpretation-box">
                <h4>Meningioma Tumor</h4>
                <ul>
                    <li>Typically benign and slow-growing</li>
                    <li>Develops from meninges (protective membranes around brain)</li>
                    <li>More common in women and older adults</li>
                    <li>Treatment may include observation, surgery, or radiation</li>
                </ul>
            </div>
            """,
        'pituitary': """
            <div class="interpretation-box">
                <h4>Pituitary Tumor</h4>
                <ul>
                    <li>Affects pituitary gland at base of brain</li>
                    <li>Often causes hormone imbalances</li>
                    <li>Mostly benign (adenomas)</li>
                    <li>Treatment options include medication, surgery, or radiation</li>
                </ul>
            </div>
            """,
        'notumor': """
            <div class="interpretation-box">
                <h4>No Tumor Detected</h4>
                <ul>
                    <li>No signs of glioma, meningioma, or pituitary tumors</li>
                    <li>Continue regular medical check-ups as recommended</li>
                    <li>Maintain a healthy lifestyle for brain health</li>
                    <li>Consult a doctor if you experience neurological symptoms</li>
                </ul>
            </div>
            """
    }
    
    st.markdown(interpretation[tumor_type], unsafe_allow_html=True)
    
    st.markdown('<h3 class="section-title">Important Notice</h3>', unsafe_allow_html=True)
    st.error("""
    ⚠️ **Medical Disclaimer:**  
    This AI tool is for research and educational purposes only. It should **NOT** be used as a substitute for professional medical diagnosis. 
    Always consult with a qualified healthcare provider for medical advice and treatment decisions.
    """)

st.sidebar.markdown('<h2 class="section-title">How to Use</h2>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background-color: #E8F0FE; padding: 1rem; border-radius: 10px;">
    <ol>
        <li><b>Upload</b> a brain MRI scan (JPG/PNG format)</li>
        <li><b>Wait</b> for the AI model to analyze the image</li>
        <li><b>Review</b> the detection results</li>
        <li><b>Consult</b> a medical professional for diagnosis</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<h2 class="section-title">About This Tool</h2>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background-color: #E6F4EA; padding: 1rem; border-radius: 10px;">
    <p>This AI-powered tool uses deep learning to detect brain tumors in MRI scans:</p>
    <ul>
        <li><b>Model:</b> VGG16 Transfer Learning</li>
        <li><b>Training Data:</b> 2870 brain MRI scans</li>
        <li><b>Classes:</b> Glioma, Meningioma, Pituitary, No Tumor</li>
        <li><b>Accuracy:</b> ~98% on test data</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<h2 class="section-title">Source Code</h2>', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="text-align: center; margin-top: 1rem;">
    <a href="https://github.com/wigjatin/Brain-Tumor-Detection" target="_blank" class="github-badge">
        <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub Repository">
    </a>
    <p style="margin-top: 0.5rem; font-size: 0.9rem;">Explore the code and dataset</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
""", unsafe_allow_html=True)