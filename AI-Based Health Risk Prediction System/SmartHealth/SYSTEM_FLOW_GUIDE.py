"""
SmartHealth AI - Complete System Architecture & Data Flow
"""

SYSTEM_ARCHITECTURE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    SmartHealth AI - System Architecture                    ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│  USER INTERFACE LAYER                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │   STREAMLIT APP      │  │  HTML + JavaScript   │  │   REST API   │  │
│  │  app_streamlit.py    │  │   index.html         │  │  (Swagger)   │  │
│  │                      │  │                      │  │              │  │
│  │  • Interactive UI    │  │  • Modern Design     │  │  • JSON I/O  │  │
│  │  • Real-time Input   │  │  • Responsive        │  │  • Batch     │  │
│  │  • Instant Results   │  │  • Animations        │  │  • 5 Endpoints
│  └──────────────────────┘  └──────────────────────┘  └──────────────┘  │
│          ↓                           ↓                        ↓           │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
┌───────────────────┴──────────────────┐   ┌───────────┴──────────────────┐
│  API LAYER (Flask Backend)           │   │  Direct Python Import         │
├──────────────────────────────────────┤   ├────────────────────────────────┤
│  api_flask.py                        │   │  model_training.py             │
│  • Single prediction endpoint        │   │  eda_visualization.py          │
│  • Batch prediction endpoint         │   │  explainability.py             │
│  • Model info endpoint               │   │                                │
│  • Features endpoint                 │   │  (Used by Streamlit directly)  │
│  • CORS enabled                      │   │                                │
└──────────────────────────────────────┘   └────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ↓

┌─────────────────────────────────────────────────────────────────────────┐
│  ML MODEL LAYER                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  PREPROCESSING (model_training.py)                              │  │
│  │  • LabelEncoder - Encode categorical (gender)                   │  │
│  │  • StandardScaler - Scale numerical features                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                            ↓                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  MODEL INFERENCE                                                │  │
│  │  • XGBoost Classifier                                           │  │
│  │  • Input: 12 scaled features                                    │  │
│  │  • Output: Risk class (0/1/2) + Probabilities                   │  │
│  │  • Accuracy: 98%                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                            ↓                                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  EXPLAINABILITY (explainability.py)                             │  │
│  │  • SHAP TreeExplainer - Model interpretability                  │  │
│  │  • Global feature importance                                    │  │
│  │  • Individual prediction explanations                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                      ↓

┌─────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  • smart_healthcare_dataset.csv - 5,000 training records               │
│  • Model weights - stored in XGBoost object                            │
│  • Scaler parameters - stored in StandardScaler object                 │
│  • Feature encoders - stored in LabelEncoder objects                   │
└─────────────────────────────────────────────────────────────────────────┘
"""

PATIENT_INPUT_FLOW = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    Patient Input → Risk Assessment Flow                    ║
╚════════════════════════════════════════════════════════════════════════════╝

STEP 1: PATIENT DATA ENTRY
═════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────┐
│ User fills in patient information:                           │
│                                                              │
│ Demographics:         Health Metrics:    Risk Factors:       │
│ • Age: 45            • BMI: 28.5        • Smoking: Yes      │
│ • Gender: Male       • Exercise: 3      • Alcohol: Yes      │
│                                                              │
│ Medical Markers:      Symptoms:                              │
│ • BP: 150           • Fatigue: Yes                          │
│ • Cholesterol: 280  • Chest Pain: No                        │
│ • Glucose: 130      • Dizziness: No                         │
│                                                              │
│ (12 Total Metrics)                                          │
└──────────────────────────────────────────────────────────────┘
                            ↓

STEP 2: DATA PREPARATION
═════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────┐
│ Convert to DataFrame:                                        │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ age  gender  bmi  exercise  smoking  alcohol  ...     │  │
│ │ 45   1       28.5 3         1        1        ...     │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Encode Categorical:                                         │
│ • gender: "Male" → 1                                        │
│ • smoking: "Yes" → 1                                        │
│ • alcohol: "Yes" → 1                                        │
└──────────────────────────────────────────────────────────────┘
                            ↓

STEP 3: FEATURE SCALING
═════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────┐
│ StandardScaler: (value - mean) / std                         │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ Original Values        Scaled Values                  │  │
│ │ age: 45       →        -0.156                         │  │
│ │ bmi: 28.5     →         0.722                         │  │
│ │ bp: 150       →         0.581                         │  │
│ │ cholesterol: 280 →      1.289                         │  │
│ │ ...                     ...                            │  │
│ └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                            ↓

STEP 4: MODEL INFERENCE
═════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────┐
│ XGBoost Model Input: [Scaled Features]                       │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ XGBoost Decision Trees (200 estimators)               │  │
│ │ ├─ Tree 1: Split on age, then bmi                     │  │
│ │ ├─ Tree 2: Split on cholesterol, then glucose         │  │
│ │ ├─ Tree 3: Split on bp, then smoking                  │  │
│ │ ├─ ...                                                │  │
│ │ └─ Tree 200: Final aggregation                        │  │
│ │                                                        │  │
│ │ Prediction: class=2 (HIGH RISK)                       │  │
│ │ Probabilities:                                        │  │
│ │   • Low Risk: 0.05                                    │  │
│ │   • Medium Risk: 0.10                                 │  │
│ │   • High Risk: 0.85 ← Maximum                         │  │
│ └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                            ↓

STEP 5: EXPLAINABILITY ANALYSIS
═════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────┐
│ SHAP TreeExplainer - Why HIGH RISK?                          │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ Top Contributing Features:                             │  │
│ │                                                        │  │
│ │ 1. age (0.87)           ┃████████████████            │  │
│ │    Contributes strongly to high risk                  │  │
│ │                                                        │  │
│ │ 2. blood_pressure (0.31)┃███████                      │  │
│ │    Elevated BP increases risk                         │  │
│ │                                                        │  │
│ │ 3. cholesterol (0.24)   ┃█████                        │  │
│ │    High cholesterol is concerning                     │  │
│ │                                                        │  │
│ │ 4. glucose (0.18)       ┃████                         │  │
│ │    Elevated glucose contributes                       │  │
│ │                                                        │  │
│ │ 5. smoking (0.12)       ┃███                          │  │
│ │    Smoking habit increases risk                       │  │
│ │                                                        │  │
│ └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                            ↓

STEP 6: RECOMMENDATION GENERATION
═════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────┐
│ Logic-Based Recommendations:                                 │
│                                                              │
│ IF blood_pressure > 140 THEN                                 │
│   → "Blood Pressure: Your BP is elevated. Reduce sodium,    │
│      manage stress, and consult a doctor."                   │
│                                                              │
│ IF cholesterol > 240 THEN                                    │
│   → "Cholesterol: High cholesterol increases heart disease   │
│      risk. Consider diet changes or medication."             │
│                                                              │
│ IF glucose > 126 THEN                                        │
│   → "Blood Glucose: High glucose levels may indicate         │
│      diabetes risk. Consult your healthcare provider."       │
│                                                              │
│ IF smoking == 1 THEN                                         │
│   → "Smoking Cessation: Quitting smoking significantly       │
│      reduces health risks. Seek professional support."       │
│                                                              │
│ IF alcohol == 1 THEN                                         │
│   → "Alcohol Reduction: Limit alcohol consumption to         │
│      recommended levels."                                    │
│                                                              │
│ (Multiple recommendations can be generated)                  │
└──────────────────────────────────────────────────────────────┘
                            ↓

STEP 7: RESULTS DISPLAY
═════════════════════════════════════════════════════════════════════════════
┌──────────────────────────────────────────────────────────────┐
│ ┌────────────────────────────────────────────────────────┐  │
│ │           📊 ASSESSMENT RESULTS                        │  │
│ ├────────────────────────────────────────────────────────┤  │
│ │                                                        │  │
│ │  🔴 HIGH RISK                                         │  │
│ │  Risk Score: 85%                                      │  │
│ │                                                        │  │
│ │  Probabilities:                                       │  │
│ │  • 🟢 Low Risk: 5%                                    │  │
│ │  • 🟡 Medium Risk: 10%                                │  │
│ │  • 🔴 High Risk: 85%                                  │  │
│ │                                                        │  │
│ │  💡 Personalized Recommendations:                     │  │
│ │  • Blood Pressure Management                          │  │
│ │  • Cholesterol Reduction                              │  │
│ │  • Glucose Control                                    │  │
│ │  • Smoking Cessation                                  │  │
│ │  • Alcohol Reduction                                  │  │
│ │                                                        │  │
│ │  🔍 Model Explainability:                             │  │
│ │  • Age: 0.87 (Top factor)                             │  │
│ │  • Blood Pressure: 0.31                               │  │
│ │  • Cholesterol: 0.24                                  │  │
│ │                                                        │  │
│ │  ⚠️ Medical Disclaimer:                                │  │
│ │  Consult healthcare professionals                     │  │
│ └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
"""

MODEL_DECISION_TREE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    XGBoost Decision Tree Example                           ║
╚════════════════════════════════════════════════════════════════════════════╝

Input: [age=45, gender=1, bmi=28.5, exercise_level=3, ...]

                         ┌──────────────────────┐
                         │   Root Node          │
                         │   age <= 48          │
                         └──┬──────────────────┬┘
                    YES /    │                   │    \ NO
                    /        │                   │      \
          ┌─────────────┐    │                   │    ┌──────────┐
          │  age <= 48  │    │                   │    │ age > 48 │
          │  (our case) │    │                   │    └──────────┘
          └──────┬──────┘    │                   │
                 │           │                   │
                 ▼           │                   │
        ┌─────────────────┐  │                   │
        │  bmi > 26       │  │                   │
        │  (28.5 > 26)    │  │                   │
        │  YES - our path │  │                   │
        └────┬────────────┘  │                   │
             │               │                   │
             ▼               │                   │
      ┌───────────────┐      │                   │
      │ cholesterol   │      │                   │
      │ > 240         │      │                   │
      │ YES-our case  │      │                   │
      └────┬──────────┘      │                   │
           │                 │                   │
           ▼                 │                   │
    ┌─────────────────┐      │                   │
    │ glucose > 126   │      │                   │
    │ YES-our case    │      │                   │
    └────┬────────────┘      │                   │
         │                   │                   │
         ▼                   │                   │
  ┌───────────────────┐      │                   │
  │ smoking == 1      │      │                   │
  │ YES-our case      │      │                   │
  └────┬──────────────┘      │                   │
       │                     │                   │
       ▼                     │                   │
┌─────────────────────────┐  │                   │
│ LEAF NODE: HIGH RISK    │  │                   │
│ Class = 2               │  │                   │
│ Confidence: 0.85        │  │                   │
│ Samples: 247            │  │                   │
│ Value: [5, 8, 234]      │  │                   │
└─────────────────────────┘  │                   │
                             │                   │
                    (Other paths lead to different leaves)

Each tree makes a small contribution to the final prediction.
With 200 trees, the final probability is the average of all tree predictions.
"""

FEATURE_IMPORTANCE_EXAMPLE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    Feature Importance Analysis                             ║
╚════════════════════════════════════════════════════════════════════════════╝

GLOBAL IMPORTANCE (All Predictions)
═════════════════════════════════════════════════════════════════════════════

Feature                 Importance   Bar Chart
────────────────────────────────────────────────────────────────────────────
age                     0.7619       ████████████████████████████ (76.19%)
bmi                     0.0516       ██ (5.16%)
glucose                 0.0343       █ (3.43%)
smoking                 0.0322       █ (3.22%)
chest_pain              0.0260       (2.60%)
alcohol                 0.0194       (1.94%)
cholesterol             0.0166       (1.66%)
blood_pressure          0.0158       (1.58%)
fatigue                 0.0146       (1.46%)
exercise_level          0.0109       (1.09%)
dizziness               0.0104       (1.04%)
gender                  0.0064       (0.64%)
────────────────────────────────────────────────────────────────────────────

KEY INSIGHTS:
• Age is by far the most important feature (76% of importance)
• BMI is the second most important (5%)
• All other features have <5% individual importance
• This shows age and BMI together drive most predictions
• Other factors provide supporting evidence


LOCAL IMPORTANCE (Individual Prediction)
═════════════════════════════════════════════════════════════════════════════

For our patient (age 45, various metrics):

Feature                 SHAP Value   Impact
────────────────────────────────────────────────────────────────────────────
age (45)                +0.32        Increases risk
blood_pressure (150)    +0.18        Increases risk
cholesterol (280)       +0.12        Increases risk
glucose (130)           +0.08        Increases risk
smoking (1)             +0.06        Increases risk
bmi (28.5)              +0.05        Increases risk
────────────────────────────────────────────────────────────────────────────
                        +0.81        Total increase to base (HIGH RISK)

BASE VALUE (Average prediction): 0.25 (Low Risk)
+ SHAP VALUES:           +0.81
= FINAL PREDICTION:      1.06 → Class 2 (HIGH RISK)

This shows exactly which factors pushed this patient toward high risk.
"""

if __name__ == "__main__":
    print(SYSTEM_ARCHITECTURE)
    print("\n")
    print(PATIENT_INPUT_FLOW)
    print("\n")
    print(MODEL_DECISION_TREE)
    print("\n")
    print(FEATURE_IMPORTANCE_EXAMPLE)
