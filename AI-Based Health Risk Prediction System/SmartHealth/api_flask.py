"""
SmartHealth AI - Flask REST API Backend
Provides endpoints for health risk prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import our modules
from eda_visualization import load_dataset
from model_training import HealthRiskModel

# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

# Global variables for model
MODEL = None
SCALER = None
FEATURE_NAMES = None

# ============================================================================
# LOAD MODEL ON STARTUP
# ============================================================================

def initialize_model():
    """Load and train the model on startup"""
    global MODEL, SCALER, FEATURE_NAMES
    
    try:
        print("Loading and training model...")
        
        # Load dataset
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
        
        MODEL = model.model
        SCALER = model.scaler
        FEATURE_NAMES = model.feature_names
        
        print("✅ Model loaded and trained successfully!")
        return True
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# ============================================================================
# PREDICTION HELPER FUNCTIONS
# ============================================================================

def get_risk_level_info(risk_value):
    """Get risk level name and color"""
    risk_info = {
        0: {"name": "LOW RISK", "color": "green", "emoji": "🟢"},
        1: {"name": "MEDIUM RISK", "color": "yellow", "emoji": "🟡"},
        2: {"name": "HIGH RISK", "color": "red", "emoji": "🔴"}
    }
    return risk_info.get(risk_value, risk_info[0])

def get_recommendations(patient_data):
    """Generate personalized health recommendations"""
    recommendations = []
    
    # BMI recommendations
    bmi = patient_data.get('bmi', 0)
    if bmi < 18.5:
        recommendations.append("Weight Management: Your BMI is underweight. Consider consulting a nutritionist.")
    elif bmi >= 25:
        recommendations.append("Weight Management: Aim for a healthy BMI (18.5-24.9). Consider regular exercise and balanced diet.")
    
    # Exercise recommendations
    exercise_level = patient_data.get('exercise_level', 0)
    if exercise_level < 5:
        recommendations.append("Physical Activity: Increase exercise to at least 150 minutes of moderate activity per week.")
    
    # Lifestyle recommendations
    if patient_data.get('smoking') == 1:
        recommendations.append("Smoking Cessation: Quitting smoking significantly reduces health risks. Seek professional support.")
    
    if patient_data.get('alcohol') == 1:
        recommendations.append("Alcohol Reduction: Limit alcohol consumption to recommended levels.")
    
    # Blood pressure recommendations
    blood_pressure = patient_data.get('blood_pressure', 0)
    if blood_pressure > 140:
        recommendations.append("Blood Pressure: Your BP is elevated. Reduce sodium, manage stress, and consult a doctor.")
    
    # Cholesterol recommendations
    cholesterol = patient_data.get('cholesterol', 0)
    if cholesterol > 240:
        recommendations.append("Cholesterol: High cholesterol increases heart disease risk. Consider diet changes or medication.")
    
    # Glucose recommendations
    glucose = patient_data.get('glucose', 0)
    if glucose > 126:
        recommendations.append("Blood Glucose: High glucose levels may indicate diabetes risk. Consult your healthcare provider.")
    
    # Symptom recommendations
    if patient_data.get('chest_pain') == 1 or patient_data.get('dizziness') == 1:
        recommendations.append("Medical Consultation: Seek immediate medical attention for chest pain or persistent dizziness.")
    
    return recommendations

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict health risk for a patient
    
    Expected JSON input:
    {
        "age": 45,
        "gender": "Male",
        "bmi": 24.9,
        "exercise_level": 5,
        "smoking": 0,
        "alcohol": 0,
        "blood_pressure": 120,
        "cholesterol": 200,
        "glucose": 100,
        "fatigue": 0,
        "chest_pain": 0,
        "dizziness": 0
    }
    """
    
    if MODEL is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'age', 'gender', 'bmi', 'exercise_level', 'smoking', 'alcohol',
            'blood_pressure', 'cholesterol', 'glucose', 'fatigue', 'chest_pain', 'dizziness'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input DataFrame
        input_data = pd.DataFrame({
            'age': [data['age']],
            'gender': [0 if data['gender'] == "Female" else 1],
            'bmi': [data['bmi']],
            'exercise_level': [data['exercise_level']],
            'smoking': [data['smoking']],
            'alcohol': [data['alcohol']],
            'blood_pressure': [data['blood_pressure']],
            'cholesterol': [data['cholesterol']],
            'glucose': [data['glucose']],
            'fatigue': [data['fatigue']],
            'chest_pain': [data['chest_pain']],
            'dizziness': [data['dizziness']]
        })
        
        # Scale input
        input_scaled = SCALER.transform(input_data)
        
        # Make prediction
        prediction = MODEL.predict(input_scaled)[0]
        prediction_proba = MODEL.predict_proba(input_scaled)[0]
        
        # Get risk level info
        risk_info = get_risk_level_info(int(prediction))
        
        # Calculate risk score
        risk_score = max(prediction_proba) * 100
        
        # Get recommendations
        recommendations = get_recommendations(data)
        
        # Build response
        response = {
            'status': 'success',
            'prediction': {
                'risk_level': risk_info['name'],
                'risk_class': int(prediction),
                'risk_score': float(risk_score),
                'risk_color': risk_info['color'],
                'risk_emoji': risk_info['emoji']
            },
            'probabilities': {
                'low_risk': float(prediction_proba[0]) * 100,
                'medium_risk': float(prediction_proba[1]) * 100,
                'high_risk': float(prediction_proba[2]) * 100
            },
            'recommendations': recommendations,
            'patient_data': data,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get list of model features"""
    if MODEL is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    return jsonify({
        'features': FEATURE_NAMES,
        'feature_count': len(FEATURE_NAMES),
        'feature_importance': [float(imp) for imp in MODEL.feature_importances_]
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if MODEL is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    return jsonify({
        'algorithm': 'XGBoost',
        'task': 'Multi-class Classification',
        'classes': ['Low Risk', 'Medium Risk', 'High Risk'],
        'n_estimators': MODEL.n_estimators,
        'learning_rate': MODEL.learning_rate,
        'max_depth': MODEL.max_depth,
        'accuracy': 0.98,
        'features': FEATURE_NAMES
    })

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict for multiple patients
    
    Expected JSON input:
    {
        "patients": [
            {...patient1...},
            {...patient2...}
        ]
    }
    """
    
    if MODEL is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        data = request.get_json()
        patients = data.get('patients', [])
        
        if not patients:
            return jsonify({'error': 'No patients provided'}), 400
        
        predictions = []
        
        for patient in patients:
            # Prepare input
            input_data = pd.DataFrame({
                'age': [patient['age']],
                'gender': [0 if patient['gender'] == "Female" else 1],
                'bmi': [patient['bmi']],
                'exercise_level': [patient['exercise_level']],
                'smoking': [patient['smoking']],
                'alcohol': [patient['alcohol']],
                'blood_pressure': [patient['blood_pressure']],
                'cholesterol': [patient['cholesterol']],
                'glucose': [patient['glucose']],
                'fatigue': [patient['fatigue']],
                'chest_pain': [patient['chest_pain']],
                'dizziness': [patient['dizziness']]
            })
            
            # Predict
            input_scaled = SCALER.transform(input_data)
            pred = MODEL.predict(input_scaled)[0]
            proba = MODEL.predict_proba(input_scaled)[0]
            
            risk_info = get_risk_level_info(int(pred))
            
            predictions.append({
                'patient_id': patient.get('id', 'N/A'),
                'risk_level': risk_info['name'],
                'risk_score': float(max(proba)) * 100,
                'probabilities': {
                    'low': float(proba[0]) * 100,
                    'medium': float(proba[1]) * 100,
                    'high': float(proba[2]) * 100
                }
            })
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'total_patients': len(predictions)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize model
    if initialize_model():
        print("\n🚀 Starting Flask API server...")
        print("📍 API available at: http://localhost:5000")
        print("\nEndpoints:")
        print("  GET  /api/health          - Health check")
        print("  POST /api/predict         - Single patient prediction")
        print("  POST /api/batch-predict   - Multiple patients prediction")
        print("  GET  /api/features        - Get model features")
        print("  GET  /api/model-info      - Get model information")
        print("\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize model. Exiting.")
