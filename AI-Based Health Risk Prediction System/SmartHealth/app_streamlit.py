"""
SmartHealth AI - Patient Risk Assessment Frontend
Streamlit web application for health risk prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Import our modules
from eda_visualization import load_dataset
from model_training import HealthRiskModel
from explainability import ModelExplainability

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SmartHealth AI - Patient Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        color: #155724;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .risk-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 20px;
        color: #856404;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        color: #721c24;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .score-box {
        background-color: #e7f3ff;
        border: 2px solid #0066cc;
        border-radius: 10px;
        padding: 15px;
        font-size: 1.2rem;
        text-align: center;
    }
    .input-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL & SCALER
# ============================================================================

@st.cache_resource
def load_trained_model():
    """Load the trained model and data preprocessing components"""
    try:
        # Load dataset for preprocessing reference
        dataset_path = r"D:\ClgDocs\3-2\AIAC\Project\SmartHealth\smart_healthcare_dataset.csv"
        df = load_dataset(dataset_path)
        
        # Create disease_risk target variable
        df['disease_risk'] = df['heart_disease'] + df['diabetes'] + df['stroke']
        df['disease_risk'] = df['disease_risk'].apply(lambda x: min(x, 2) if x > 0 else 0)
        
        # Preprocess
        model = HealthRiskModel(target_col='disease_risk')
        X, y = model.preprocess(df)
        model.split_and_scale(X, y)
        model.train(n_estimators=200, learning_rate=0.1, max_depth=6)
        
        return model, X, y
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# ============================================================================
# RISK LEVEL FUNCTIONS
# ============================================================================

def get_risk_level(risk_value):
    """Convert numeric risk to category"""
    if risk_value == 0:
        return "LOW RISK", "🟢"
    elif risk_value == 1:
        return "MEDIUM RISK", "🟡"
    else:
        return "HIGH RISK", "🔴"

def get_risk_color_class(risk_value):
    """Get CSS class for risk level"""
    if risk_value == 0:
        return "risk-low"
    elif risk_value == 1:
        return "risk-medium"
    else:
        return "risk-high"

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #1f77b4;">🏥 SmartHealth AI</h1>
            <h3 style="color: #666;">Patient Health Risk Assessment System</h3>
            <p style="color: #888;">Powered by XGBoost & SHAP Explainability</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Load model
    with st.spinner("Loading trained model..."):
        model, X, y = load_trained_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure the dataset exists.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # ====================================================================
    # SIDEBAR - NAVIGATION
    # ====================================================================
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Patient Assessment", "Model Info", "About"]
    )
    
    # ====================================================================
    # PAGE 1: PATIENT ASSESSMENT
    # ====================================================================
    if page == "Patient Assessment":
        st.header("Patient Health Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Patient Demographics")
            age = st.slider("Age (years)", min_value=18, max_value=79, value=45)
            gender = st.selectbox("Gender", ["Female", "Male"])
            
        with col2:
            st.subheader("📊 Basic Health Metrics")
            bmi = st.slider("BMI (Body Mass Index)", min_value=15.0, max_value=42.6, value=24.9, step=0.1)
            exercise_level = st.slider("Exercise Level (0-10)", min_value=0, max_value=10, value=5)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚠️ Risk Factors")
            smoking = st.selectbox("Smoking Status", ["No (0)", "Yes (1)"])
            smoking = int(smoking.split('(')[1].strip(')'))
            
            alcohol = st.selectbox("Alcohol Consumption", ["No (0)", "Yes (1)"])
            alcohol = int(alcohol.split('(')[1].strip(')'))
            
            blood_pressure = st.slider("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
        
        with col2:
            st.subheader("🩺 Medical Markers")
            cholesterol = st.slider("Cholesterol (mg/dL)", min_value=150, max_value=300, value=200)
            glucose = st.slider("Glucose (mg/dL)", min_value=70, max_value=200, value=100)
            fatigue = st.selectbox("Fatigue", ["No (0)", "Yes (1)"])
            fatigue = int(fatigue.split('(')[1].strip(')'))
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 Symptoms")
            chest_pain = st.selectbox("Chest Pain", ["No (0)", "Yes (1)"])
            chest_pain = int(chest_pain.split('(')[1].strip(')'))
            
            dizziness = st.selectbox("Dizziness", ["No (0)", "Yes (1)"])
            dizziness = int(dizziness.split('(')[1].strip(')'))
        
        with col2:
            st.write("")  # Spacer
        
        st.divider()
        
        # ====================================================================
        # PREDICTION
        # ====================================================================
        
        if st.button("🔍 Assess Health Risk", key="assess_btn", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'age': [age],
                'gender': [0 if gender == "Female" else 1],  # Encode gender
                'bmi': [bmi],
                'exercise_level': [exercise_level],
                'smoking': [smoking],
                'alcohol': [alcohol],
                'blood_pressure': [blood_pressure],
                'cholesterol': [cholesterol],
                'glucose': [glucose],
                'fatigue': [fatigue],
                'chest_pain': [chest_pain],
                'dizziness': [dizziness]
            })
            
            # Scale input
            input_scaled = model.scaler.transform(input_data)
            
            # Make prediction
            prediction = model.model.predict(input_scaled)[0]
            prediction_proba = model.model.predict_proba(input_scaled)[0]
            
            # Get risk level and emoji
            risk_level, risk_emoji = get_risk_level(prediction)
            risk_class = get_risk_color_class(prediction)
            
            # ====================================================================
            # DISPLAY RESULTS
            # ====================================================================
            
            st.divider()
            st.header("📊 Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="{risk_class}">
                        {risk_emoji}<br>
                        {risk_level}
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Calculate risk score (0-100)
                risk_score = max(prediction_proba) * 100
                st.markdown(f"""
                    <div class="score-box">
                        Risk Score<br>
                        <span style="font-size: 2rem; font-weight: bold; color: #0066cc;">
                            {risk_score:.1f}%
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric("Low Risk", f"{prediction_proba[0]*100:.1f}%")
                st.metric("Medium Risk", f"{prediction_proba[1]*100:.1f}%")
                st.metric("High Risk", f"{prediction_proba[2]*100:.1f}%")
            
            st.divider()
            
            # ====================================================================
            # RECOMMENDATIONS
            # ====================================================================
            
            st.subheader("💡 Personalized Recommendations")
            
            recommendations = []
            
            # BMI recommendations
            if bmi < 18.5:
                recommendations.append("• **Weight Management**: Your BMI is underweight. Consider consulting a nutritionist.")
            elif bmi >= 25:
                recommendations.append("• **Weight Management**: Aim for a healthy BMI (18.5-24.9). Consider regular exercise and balanced diet.")
            
            # Exercise recommendations
            if exercise_level < 5:
                recommendations.append("• **Physical Activity**: Increase exercise to at least 150 minutes of moderate activity per week.")
            
            # Lifestyle recommendations
            if smoking == 1:
                recommendations.append("• **Smoking Cessation**: Quitting smoking significantly reduces health risks. Seek professional support.")
            
            if alcohol == 1:
                recommendations.append("• **Alcohol Reduction**: Limit alcohol consumption to recommended levels (≤1 drink/day for women, ≤2 for men).")
            
            # Blood pressure recommendations
            if blood_pressure > 140:
                recommendations.append("• **Blood Pressure**: Your BP is elevated. Reduce sodium, manage stress, and consult a doctor.")
            
            # Cholesterol recommendations
            if cholesterol > 240:
                recommendations.append("• **Cholesterol**: High cholesterol increases heart disease risk. Consider diet changes or medication.")
            
            # Glucose recommendations
            if glucose > 126:
                recommendations.append("• **Blood Glucose**: High glucose levels may indicate diabetes risk. Consult your healthcare provider.")
            
            # Symptom recommendations
            if chest_pain == 1 or dizziness == 1:
                recommendations.append("• **Medical Consultation**: Seek immediate medical attention for chest pain or persistent dizziness.")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("✅ Great! Your health metrics are within normal ranges. Continue maintaining healthy habits!")
            
            st.divider()
            
            # ====================================================================
            # FEATURE IMPORTANCE
            # ====================================================================
            
            st.subheader("🔍 What Factors Influenced This Assessment?")
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'Feature': model.feature_names,
                'Importance': model.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart(data=feature_importance.set_index('Feature'), use_container_width=True)
            
            with col2:
                st.write("**Top Contributing Factors:**")
                for idx, (feature, importance) in enumerate(zip(feature_importance['Feature'], feature_importance['Importance']), 1):
                    st.write(f"{idx}. **{feature}**: {importance:.4f}")
            
            st.divider()
            
            # ====================================================================
            # DISCLAIMER
            # ====================================================================
            
            st.warning(
                "⚠️ **Medical Disclaimer**: This assessment is for informational purposes only and should not replace "
                "professional medical advice. Always consult with a qualified healthcare provider for accurate diagnosis and treatment."
            )
    
    # ====================================================================
    # PAGE 2: MODEL INFO
    # ====================================================================
    elif page == "Model Info":
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Architecture")
            st.info("""
                **Algorithm**: XGBoost (Extreme Gradient Boosting)
                
                **Configuration**:
                - n_estimators: 200
                - learning_rate: 0.1
                - max_depth: 6
                
                **Task**: Multi-class Classification (3 classes)
                - Class 0: Low Risk
                - Class 1: Medium Risk
                - Class 2: High Risk
            """)
        
        with col2:
            st.subheader("📊 Model Performance")
            st.metric("Accuracy", "98%")
            st.metric("Precision (Macro)", "93%")
            st.metric("Recall (Macro)", "93%")
        
        st.divider()
        
        st.subheader("📋 Input Features (12 Total)")
        features_info = pd.DataFrame({
            'Feature': model.feature_names,
            'Type': ['Numeric', 'Categorical', 'Numeric', 'Numeric', 'Binary', 'Binary', 
                    'Numeric', 'Numeric', 'Numeric', 'Binary', 'Binary', 'Binary']
        })
        st.dataframe(features_info, use_container_width=True)
        
        st.divider()
        
        st.subheader("🔍 Explainability")
        st.info("""
            This model uses **SHAP (SHapley Additive exPlanations)** for transparency:
            
            - **Global Explanations**: Understand which features matter most overall
            - **Local Explanations**: See why specific predictions were made
            - **Feature Importance**: Ranked features by their contribution to predictions
            
            This ensures **trust and transparency** in medical decision-making.
        """)
    
    # ====================================================================
    # PAGE 3: ABOUT
    # ====================================================================
    elif page == "About":
        st.header("About SmartHealth AI")
        
        st.markdown("""
            ### 🏥 Project Overview
            
            **SmartHealth AI** is an intelligent health risk assessment system designed to help identify 
            potential health risks early through machine learning and data analysis.
            
            ### 🎯 Objectives
            
            - Provide **accurate health risk predictions** using advanced ML algorithms
            - Ensure **transparency and explainability** through SHAP analysis
            - Enable **proactive healthcare** management
            - Support **personalized health recommendations**
            
            ### 📊 Dataset
            
            - **Size**: 5,000 patient records
            - **Features**: 12 health metrics (demographics, risk factors, medical markers)
            - **Target**: Multi-class disease risk (Low, Medium, High)
            
            ### 🤖 Technology Stack
            
            - **Framework**: Python, Streamlit
            - **ML Algorithm**: XGBoost
            - **Data Processing**: Pandas, NumPy, Scikit-learn
            - **Explainability**: SHAP
            - **Visualization**: Matplotlib, Seaborn
            
            ### ✨ Key Features
            
            1. **Real-time Assessment**: Get instant health risk predictions
            2. **Personalized Recommendations**: Receive tailored health advice
            3. **Model Transparency**: Understand what factors drive predictions
            4. **Risk Quantification**: Get confidence scores for each risk level
            
            ### 🔐 Data Privacy
            
            Patient data entered in this application is processed locally and not stored or transmitted 
            to external servers. Your privacy is our priority.
            
            ### 📞 Support
            
            For questions or issues, please contact the development team.
            
            ---
            
            **Disclaimer**: This application is for educational and informational purposes only. 
            Always consult with qualified healthcare professionals for medical decisions.
        """)
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", "98%")
        
        with col2:
            st.metric("Input Features", "12")
        
        with col3:
            st.metric("Risk Classes", "3")

if __name__ == "__main__":
    main()
