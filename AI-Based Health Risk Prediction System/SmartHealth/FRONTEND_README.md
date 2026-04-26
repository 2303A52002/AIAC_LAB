# SmartHealth AI - Frontend Applications

This directory contains two frontend options for the SmartHealth AI patient health risk assessment system.

## 🎯 Overview

Both frontends provide the same core functionality:
- **Patient Input**: Collect 12 health metrics
- **Risk Prediction**: Generate health risk assessment (Low/Medium/High)
- **Risk Scoring**: Display confidence scores
- **Recommendations**: Provide personalized health recommendations
- **Transparency**: Explain model decisions

---

## Option 1: Streamlit Application (Recommended for Quick Setup)

Streamlit is the easiest and fastest way to get a web interface running.

### ✅ Advantages
- Single command to launch
- Interactive sliders and inputs
- Built-in visualization support
- No additional server needed
- Responsive design out-of-the-box

### 🚀 Installation & Running

#### 1. Install Streamlit
```bash
pip install streamlit>=1.30.0
```

#### 2. Run the Application
```bash
streamlit run app_streamlit.py
```

This will automatically open your browser to: `http://localhost:8501`

### 📖 Usage

1. **Navigate to "Patient Assessment"** tab
2. **Fill in patient information**:
   - Demographics (age, gender)
   - Health metrics (BMI, exercise level)
   - Risk factors (smoking, alcohol)
   - Medical markers (blood pressure, cholesterol, glucose)
   - Symptoms (fatigue, chest pain, dizziness)
3. **Click "Assess Health Risk"** button
4. **View Results**:
   - Risk level (Low/Medium/High with color coding)
   - Risk score (0-100%)
   - Probability breakdown
   - Personalized recommendations
   - Feature importance analysis

### 📱 Features

- **Patient Assessment**: Real-time health risk evaluation
- **Model Info**: Technical details about the model
- **About**: Project background and features

### 🎨 User Interface

Streamlit automatically provides:
- Responsive layout
- Mobile-friendly interface
- Interactive sliders and selectors
- Real-time value display
- Error handling and messages

---

## Option 2: Flask REST API + HTML Frontend

For more control and a traditional web interface, use the Flask backend with HTML/CSS/JavaScript.

### ✅ Advantages
- RESTful API for multiple clients
- Traditional web interface
- Batch prediction support
- Easy to customize HTML/CSS
- Can integrate with other systems

### 🚀 Installation & Running

#### 1. Install Flask
```bash
pip install flask>=3.0.0 flask-cors>=4.0.0
```

#### 2. Start Flask API Server
```bash
python api_flask.py
```

You'll see:
```
🚀 Starting Flask API server...
📍 API available at: http://localhost:5000
```

#### 3. Open HTML Frontend
Open `index.html` in your web browser:
```bash
# On Windows
start index.html

# On macOS
open index.html

# On Linux
xdg-open index.html
```

Or open in VS Code Live Server:
- Right-click `index.html` → "Open with Live Server"

Frontend will be available at: `http://localhost:5500` (or similar)

### 📖 Usage

1. **Fill in Patient Information** using the form
2. **Real-time Value Display** - Sliders show live values
3. **Click "Assess Health Risk"** button
4. **View Instant Results**:
   - Color-coded risk level
   - Risk percentage score
   - Probability distribution
   - Personalized health recommendations
5. **Navigate Tabs**:
   - Patient Assessment
   - About (Project information)
   - FAQ (Frequently Asked Questions)

### 🎨 User Interface Features

- **Modern Design**: Gradient backgrounds, smooth animations
- **Responsive Layout**: Works on desktop, tablet, mobile
- **Real-time Sliders**: See values update instantly
- **Color-Coded Results**:
  - 🟢 Green for Low Risk
  - 🟡 Yellow for Medium Risk
  - 🔴 Red for High Risk
- **Smooth Transitions**: Professional animations
- **Error Handling**: Clear error messages

### 🔌 API Endpoints

If using Flask backend:

```bash
# Health Check
GET /api/health

# Single Patient Prediction
POST /api/predict
Content-Type: application/json
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

# Batch Predictions (Multiple Patients)
POST /api/batch-predict
Content-Type: application/json
{
  "patients": [
    { ...patient1... },
    { ...patient2... }
  ]
}

# Get Model Features
GET /api/features

# Get Model Information
GET /api/model-info
```

---

## 📊 Input Fields Explained

### Demographics
- **Age** (18-79): Patient's age in years
- **Gender** (M/F): Biological sex

### Health Metrics
- **BMI** (15-42.6): Body Mass Index
- **Exercise Level** (0-10): Physical activity frequency/intensity

### Risk Factors
- **Smoking** (Yes/No): Current smoking status
- **Alcohol** (Yes/No): Alcohol consumption

### Medical Markers
- **Blood Pressure** (80-200): Systolic BP in mmHg
- **Cholesterol** (150-300): Total cholesterol in mg/dL
- **Glucose** (70-200): Fasting glucose in mg/dL

### Symptoms
- **Fatigue** (Yes/No): Unusual tiredness
- **Chest Pain** (Yes/No): Chest discomfort
- **Dizziness** (Yes/No): Vertigo/lightheadedness

---

## 📈 Output & Results

### Risk Level Classification
- **Low Risk (0)**: ✅ Minimal disease indicators
- **Medium Risk (1)**: ⚠️ Some concerning factors
- **High Risk (2)**: 🚨 Multiple risk indicators

### Risk Score (0-100%)
Represents the confidence of the highest probability class.

### Probability Distribution
Shows likelihood for each risk category:
- Low Risk: %
- Medium Risk: %
- High Risk: %

### Personalized Recommendations
Based on patient metrics, the system provides:
- Weight management advice
- Exercise recommendations
- Lifestyle modification suggestions
- Medical consultation recommendations

---

## 🔧 Troubleshooting

### Flask Server Issues

**Error: "Connection refused"**
- Make sure Flask server is running: `python api_flask.py`
- Check that port 5000 is available

**Error: "Model not initialized"**
- Ensure dataset exists at: `D:\ClgDocs\3-2\AIAC\Project\SmartHealth\smart_healthcare_dataset.csv`
- Check that all dependencies are installed

### Streamlit Issues

**Error: "No module named 'streamlit'"**
- Install: `pip install streamlit`

**Error: "Port already in use"**
- Kill process: `lsof -i :8501` then `kill -9 <PID>`
- Or use different port: `streamlit run app_streamlit.py --server.port 8502`

### CORS Issues with Flask + HTML

If you get CORS errors:
- Make sure Flask is running
- Check that CORS is enabled (should be by default)
- Try accessing from same host (localhost)

---

## 🚀 Deployment

### Streamlit Cloud
```bash
# Requirements: GitHub account
# Push code to GitHub
# Go to share.streamlit.io
# Select your repository
# Deploy automatically
```

### Flask to Production
```bash
# Use production server (Gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_flask.py

# Use in Docker container
# Or deploy to cloud (AWS, Heroku, etc.)
```

---

## 📝 Notes

- All predictions are performed locally - no data is sent to external servers
- The model achieves 98% accuracy on test data
- Results should be verified by healthcare professionals
- Always consult qualified medical practitioners for diagnosis

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Check that the dataset file exists

---

**Choose Streamlit for simplicity, Flask for customization!** 🎉
