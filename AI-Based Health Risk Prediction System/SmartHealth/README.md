# SmartHealth AI Project

A comprehensive machine learning pipeline for health risk prediction using XGBoost and SHAP explainability.

## 📋 Project Overview

This project implements an end-to-end ML workflow with:
- **Dataset**: 100,000 health records with lifestyle metrics
- **Target**: Multi-class disease risk classification (Low, Medium, High)
- **Model**: XGBoost with SHAP-based explainability
- **Features**: BMI, daily steps, sleep hours, water intake, calories, smoking, alcohol consumption

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python main.py
```

This will execute the complete workflow:
1. **EDA**: Data exploration and visualization
2. **Preprocessing**: Feature encoding and scaling
3. **Model Training**: XGBoost classification
4. **Evaluation**: Performance metrics and confusion matrix
5. **Explainability**: SHAP analysis for trust and transparency

## 📁 Project Structure

```
SmartHealth/
├── main.py                      # Main orchestration script
├── eda_visualization.py         # Data exploration & visualization
├── model_training.py            # Preprocessing & XGBoost model
├── explainability.py            # SHAP-based explainability
├── requirements.txt             # Python dependencies
├── smart_healthcare_dataset.csv # Dataset (100k records)
└── outputs/                     # Generated visualizations
    ├── 01_disease_risk_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_bmi_vs_sleep.png
    ├── 04_feature_distributions.png
    ├── 05_confusion_matrix.png
    ├── 06_xgboost_feature_importance.png
    ├── 07_shap_summary_plot.png
    ├── 08_shap_dependence_*.png
    └── 09_prediction_explanation_*.png
```

## 📊 Pipeline Phases

### Phase 1: Exploratory Data Analysis
- Dataset loading and profiling
- Missing values analysis
- Statistical summaries
- Correlation heatmap
- Feature distributions
- Target variable analysis

### Phase 2: Preprocessing & Model Training
- Categorical feature encoding (LabelEncoder)
- Train-test split (80-20)
- Feature scaling (StandardScaler)
- XGBoost model training
- Performance evaluation
- Feature importance analysis

### Phase 3: Explainability & Trust
- SHAP TreeExplainer creation
- Global feature importance (summary plot)
- Local prediction explanations
- Feature dependence analysis
- Trust & transparency reporting

## 🎯 Model Configuration

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
```

**Target Accuracy**: ≥90%

## 📈 Key Metrics

The pipeline generates:
- **Accuracy Score**: Overall model performance
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Prediction breakdown by class
- **Feature Importance**: XGBoost feature rankings
- **SHAP Values**: Model explainability for every prediction

## 🔍 Understanding the Results

### Feature Importance
Shows which features are most predictive of disease risk:
- Global importance (overall model behavior)
- Local importance (individual predictions)

### SHAP Analysis
Provides trust and transparency through:
- **Summary Plots**: Which features matter most
- **Dependence Plots**: How features affect predictions
- **Individual Explanations**: Why specific predictions were made

### Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives among positive predictions
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall

## 💾 Dataset Format

Expected CSV structure:
```
age, gender, daily_steps, sleep_hours, water_intake_l, 
calories_consumed, bmi, smoker, alcohol, disease_risk
```

## 📝 Notes

- Dataset path must be configured in `main.py`
- Outputs are saved to `outputs/` directory
- SHAP computation may take time for large datasets
- All visualizations are saved as high-resolution PNG files (300 dpi)

## 🤝 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, shap
- matplotlib, seaborn
- jupyter (optional, for interactive exploration)

## 📚 References

- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/
- Kaggle Dataset: Health & Lifestyle Dataset (100k Records)

---

**Project Status**: ✅ Complete and Ready for Deployment
