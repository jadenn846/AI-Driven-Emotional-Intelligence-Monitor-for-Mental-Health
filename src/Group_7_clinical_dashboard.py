import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from collections import deque
import time


# ============================================================
# 0. IMPROVED CNN ARCHITECTURE (FROM SCRATCH)
# ============================================================
class ImprovedCNNEmotion(nn.Module):
    def __init__(self, num_classes=7, dropout_head=0.5):
        super().__init__()

        def conv_block(in_ch, out_ch, p_drop=0.0):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if p_drop > 0:
                layers.append(nn.Dropout2d(p_drop))
            layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1,   32, p_drop=0.05),
            conv_block(32,  64, p_drop=0.10),
            conv_block(64, 128, p_drop=0.15),
            conv_block(128, 256, p_drop=0.20),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================
# 1. SETTINGS & RESOURCE LOADING
# ============================================================
st.set_page_config(page_title="AI Mental Health Monitor", layout="wide")

# Ensure this path points to your FER2013 model weights
MODEL_PATH = r"C:/data/emotion_recognition/emotion_model_cnn_improved.pth"
# FER2013 standard label order
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise' ]

@st.cache_resource
def load_resources():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = ImprovedCNNEmotion(num_classes=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return face_cascade, model

face_net, emotion_net = load_resources()

# Pre-processing (FER2013 typically uses 0-1 scaling)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# ============================================================
# 2. SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# ============================================================
# 3. UI LAYOUT
# ============================================================
st.title("🧠 Mental Health Real-Time Analytics (FER2013)")
st.markdown("### Clinical Monitoring Dashboard")

with st.sidebar:
    st.header("⚙️ Calibration Panel")
    st.info("Adjust sensitivity to compensate for lighting or individual baseline differences.")
    
    # Sliders to help with your 'Yellow Light' and 'Muscle Tension' issues
    sad_boost = st.slider("Sadness Sensitivity", 1.0, 5.0, 2.5)
    anger_boost = st.slider("Anger Sensitivity", 1.0, 5.0, 2.0)
    neutral_suppress = st.slider("Neutrality Suppression", 0.0, 1.0, 0.3)
    
    st.divider()
    start_btn = st.button("▶️ Start 10-Min Session", use_container_width=True)
    stop_btn = st.button("⏹️ End & Generate Report", use_container_width=True)
    if st.button("🗑️ Reset All Data", use_container_width=True):
        st.session_state.history = []
        st.session_state.timestamps = []
        st.rerun()

col_video, col_metrics = st.columns([2, 1])
webcam_placeholder = col_video.empty()
chart_placeholder = col_metrics.empty()

# ============================================================
# 4. PROCESSING LOOP
# ============================================================
if start_btn:
    cap = cv2.VideoCapture(0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Vital for yellow light
    prob_buffer = deque(maxlen=8) # Smoothing to prevent "flickering"
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break

        current_elapsed = time.time() - start_time
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_net.detectMultiScale(gray_frame, 1.1, 7, minSize=(60, 60))
        
        current_probs = np.zeros(7)
        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # ENHANCEMENT: Fix lighting contrast
                face_roi = clahe.apply(face_roi)
                pil_img = Image.fromarray(face_roi)
                tensor = transform(pil_img).unsqueeze(0)
                
                with torch.no_grad():
                    output = emotion_net(tensor)
                    probs = torch.softmax(output, dim=1).numpy()[0]
                    
                    # --- DYNAMIC CLINICAL LOGIC ---
                    probs[4] *= (1.0 - neutral_suppress)  # Neutral
                    probs[5] *= sad_boost                 # Sad
                    probs[0] *= anger_boost               # Angry
                                        
                    # Re-normalize
                    probs = np.clip(probs, 0, 1)
                    probs /= np.sum(probs)
                    
                    prob_buffer.append(probs)
                    current_probs = np.mean(prob_buffer, axis=0)
                    
                    st.session_state.history.append(current_probs)
                    st.session_state.timestamps.append(current_elapsed)

                label = EMOTIONS[np.argmax(current_probs)]
                # Visual Alerts
                color = (0, 0, 255) if label in ['Angry', 'Sad', 'Fear', 'Disgust'] else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label}: {np.max(current_probs):.1%}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        webcam_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        chart_placeholder.bar_chart(pd.DataFrame(current_probs, index=EMOTIONS, columns=["Intensity"]))

    cap.release()
    st.success("Session finished. Analysis report ready.")

# ============================================================
# 5. CLINICAL ASSESSMENT REPORT
# ============================================================
if not start_btn and len(st.session_state.history) > 0:
    st.divider()
    st.header("📊 Therapeutic Analytics Report")
    
    df = pd.DataFrame(st.session_state.history, columns=EMOTIONS)
    df['Time'] = st.session_state.timestamps
    avg_emotions = df[EMOTIONS].mean()
    
    # Clinical Grouping
    distress_score = avg_emotions[['Angry', 'Disgust', 'Fear', 'Sad']].sum()
    stability_score = avg_emotions[['Happy', 'Neutral']].sum()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Emotional Profile")
        st.bar_chart(avg_emotions)
    with c2:
        st.subheader("Longitudinal Session Arc")
        st.line_chart(df.set_index('Time')[['Sad', 'Angry', 'Happy', 'Neutral', 'Surprise']])

    # --- FINAL CLINICAL SUMMARY ---
    st.subheader("🩺 Therapeutic Assessment")
    
    if avg_emotions['Surprise'] > 0.20:
        st.info(f"**Insight:** Significant 'Cognitive Shift' detected (Surprise: {avg_emotions['Surprise']:.1%}).")

    if distress_score > stability_score:
        st.warning(f"**Assessment:** High Negative Affect Detected ({distress_score:.1%}).")
        st.write("The patient exhibited frequent indicators of emotional imbalance. Sustained distress detected.")
    else:
        st.success(f"**Assessment:** High Emotional Stability Detected ({stability_score:.1%}).")
        st.write("The patient demonstrated consistent emotional regulation throughout the session.")