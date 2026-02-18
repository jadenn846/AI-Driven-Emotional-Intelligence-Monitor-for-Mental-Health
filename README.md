# AI-Driven Emotional Intelligence Monitor for Mental Health

## 🏥 Clinical Problem
Therapists and mental health professionals often rely on subjective observation to gauge a patient's emotional state during clinical sessions. To provide objective, quantitative data, this project aims to develop a tool that monitors and logs patient emotional fluctuations in real-time.

**Objective:** Develop a deep learning computer vision system to detect 7 distinct emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) and visualize them for therapists during a 10-minute clinical session.

---

## 🧠 Analytical Approach

### 1. Data Engineering & Preprocessing
* **Dataset:** Utilized the **FER2013** dataset (Facial Expression Recognition), consisting of 25,000+ grayscale images (48x48 pixels).
* **Challenges:** Identified severe class imbalance (e.g., "Disgust" was only 1.5% of data).
* **Strategy:**
    * **Stratified Splitting:** Ensured training/validation sets had identical emotion distributions to prevent bias.
    * **Augmentation:** Applied Random Affine transformations (rotation, scaling, translation) to simulate real-world patient head movements.
    * **Class Balancing:** Implemented `WeightedRandomSampler` to force the model to pay attention to rare micro-expressions like "Disgust" and "Fear."

### 2. Model Architecture: "Improved CNN"
We developed a custom Convolutional Neural Network (CNN) designed for clinical precision:
* **Hierarchical Feature Extraction:** 4 stages of double-convolution blocks to capture complex facial landmarks (e.g., brow furrows, lip corners).
* **Global Average Pooling (GAP):** Replaced traditional flattening to reduce overfitting and focus on semantic features.
* **Regularization:** Used Batch Normalization and specialized Dropout schedules (light dropout in early layers, heavy dropout in the classifier) to ensure the model generalizes to new patients.

---

## 📊 Key Results & Findings

We compared a Baseline CNN against our Improved Architecture:

| Metric | Baseline CNN | Improved CNN |
| :--- | :--- | :--- |
| **Test Accuracy** | 63.7% | **65.9%** |
| **Stability** | Moderate | **High (Smoother Convergence)** |
| **Generalization** | Prone to Overfitting | **Robust (Low Variance)** |

* **Clinical Relevance:** The Improved CNN successfully distinguished between "Neutral" (resting face) and "Sad" (micro-expression), a critical distinction for diagnosing depressive symptoms.
* **Validation:** Achieved ~66% accuracy on unseen test data, significantly outperforming random chance (14.3%).

---

## 💻 Clinical Dashboard (Streamlit App)

This project includes a fully functional **Streamlit Dashboard** for real-time inference.

* **Real-Time Tracking:** Processes live webcam feed to detect emotions frame-by-frame.
* **Clinical Calibration:** Allows therapists to adjust sensitivity sliders (e.g., "Sadness Boost" or "Neutral Suppression") to account for different patient baselines.
* **Session Analytics:** Generates a post-session report with:
    * **Distress Score:** Aggregate of negative emotions (Anger + Fear + Sadness).
    * **Stability Score:** Measure of emotional regulation over time.
    * **Longitudinal Graph:** A time-series plot of the entire 10-minute session.

---

## 🛠️ How to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/jadenn846/AI-Driven-Emotional-Intelligence-Monitor-for-Mental-Health
cd AI-Emotion-Monitor
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Training Notebook (Optional)
```bash
If you want to retrain the model from scratch:
Open Group_7_AI_Emotion_Detection_DL.ipynb in Jupyter Notebook or Google Colab to train the model. This will save the weights as emotion_model_cnn_improved.pth.
```
### 4. Launch the Clinical Dashboard
```bash
streamlit run Group_7_clinical_dashboard.py
```
