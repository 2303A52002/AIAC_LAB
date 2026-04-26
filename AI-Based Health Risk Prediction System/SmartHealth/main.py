"""
"Create a main.py file that orchestrates the complete SmartHealth AI pipeline:
 1. Load the health dataset from CSV
 2. Create disease_risk target variable by combining heart_disease, diabetes, stroke
 3. Run EDA visualization on the dataset
 4. Train an XGBoost model with 200 estimators for risk classification
 5. Generate SHAP explainability analysis
 6. Save all outputs to outputs/ directory
 
 Make it production-ready with clear progress indicators and error handling."

PURPOSE:
────────
Orchestrates the complete end-to-end ML workflow:
• EDA → Preprocessing → Model Training → Explainability Analysis0129
• Loads 5,000 patient health records
• Creates 12-feature ML model
• Generates 7 visualization outputs
• Provides SHAP model interpretability

COMPLETE END-TO-END WORKFLOW:
EDA → Preprocessing → Model Training → Explainability
"""

import sys
import os
from pathlib import Path

# Import modules
from eda_visualization import (
    load_dataset, plot_correlation_heatmap, plot_bmi_sleep_scatter,
    plot_feature_distributions, plot_disease_risk_distribution, generate_eda_report
)
from model_training import HealthRiskModel
from explainability import ModelExplainability

# Configuration
DATASET_PATH = r"D:\ClgDocs\3-2\AIAC\Project\SmartHealth\smart_healthcare_dataset.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    """Main pipeline execution"""
    
    print("\n" + "="*70)
    print("🏥 SMARTHEALTH AI PROJECT - COMPLETE PIPELINE")
    print("="*70)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*70)
    
    # ============================================================================
    # PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)
    # ============================================================================
    print("\n\n🔍 PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("-" * 70)
    
    # Load dataset
    df = load_dataset(DATASET_PATH)
    
    # Create disease_risk target variable from disease indicators
    # Combine heart_disease, diabetes, and stroke into multi-class target
    print("\n[DATA PREPROCESSING] Creating disease_risk target variable...")
    df['disease_risk'] = df['heart_disease'] + df['diabetes'] + df['stroke']
    # Map to 0: Low (no diseases), 1: Medium (1 disease), 2: High (2+ diseases)
    df['disease_risk'] = df['disease_risk'].apply(lambda x: min(x, 2) if x > 0 else 0)
    print(f"✓ disease_risk created with distribution:\n{df['disease_risk'].value_counts().sort_index()}")
    
    # Generate EDA report
    generate_eda_report(df)
    
    # Visualizations
    plot_disease_risk_distribution(
        df, 
        save_path=str(OUTPUT_DIR / "01_disease_risk_distribution.png")
    )
    
    plot_correlation_heatmap(
        df, 
        save_path=str(OUTPUT_DIR / "02_correlation_heatmap.png")
    )
    
    plot_bmi_sleep_scatter(
        df, 
        target_col='disease_risk',
        save_path=str(OUTPUT_DIR / "03_bmi_vs_cholesterol.png")
    )
    
    lifestyle_features = ['blood_pressure', 'cholesterol', 'glucose', 'bmi']
    plot_feature_distributions(
        df, 
        lifestyle_features,
        save_path=str(OUTPUT_DIR / "04_feature_distributions.png")
    )
    
    # ============================================================================
    # PHASE 2: PREPROCESSING & MODEL TRAINING
    # ============================================================================
    print("\n\n🤖 PHASE 2: PREPROCESSING & MODEL TRAINING")
    print("-" * 70)
    
    # Initialize model
    model = HealthRiskModel(target_col='disease_risk', test_size=0.2, random_state=42)
    
    # Preprocessing
    X, y = model.preprocess(df)
    
    # Split and scale
    model.split_and_scale(X, y)
    
    # Train model
    model.train(n_estimators=200, learning_rate=0.1, max_depth=6)
    
    # Evaluate
    accuracy = model.evaluate()
    
    # Visualizations
    model.plot_confusion_matrix(
        save_path=str(OUTPUT_DIR / "05_confusion_matrix.png")
    )
    
    model.plot_feature_importance(
        top_n=15,
        save_path=str(OUTPUT_DIR / "06_xgboost_feature_importance.png")
    )
    
    # ============================================================================
    # PHASE 3: EXPLAINABILITY & TRUST (SHAP)
    # ============================================================================
    print("\n\n🔍 PHASE 3: EXPLAINABILITY & TRUST (SHAP)")
    print("-" * 70)
    
    # Initialize explainability module
    xai = ModelExplainability(
        model=model.model,
        X_test_scaled=model.X_test_scaled,
        X_test=model.X_test,
        feature_names=model.feature_names
    )
    
    # Create SHAP explainer
    xai.create_explainer()
    
    # SHAP visualizations
    xai.plot_summary(save_path=str(OUTPUT_DIR / "07_shap_summary_plot.png"))
    
    # Trust & transparency report
    trust_report = xai.generate_trust_report(accuracy * 100)
    
    # ============================================================================
    # SUMMARY & RESULTS
    # ============================================================================
    print("\n\n" + "="*70)
    print("✅ PROJECT COMPLETION SUMMARY")
    print("="*70)
    
    model_summary = model.get_model_summary()
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {model_summary['accuracy']:.2f}%")
    print(f"   Training Samples: {model_summary['train_size']}")
    print(f"   Test Samples: {model_summary['test_size']}")
    
    print(f"\n🎯 Model Configuration:")
    print(f"   Algorithm: XGBoost")
    print(f"   n_estimators: 200")
    print(f"   learning_rate: 0.1")
    print(f"   max_depth: 6")
    
    print(f"\n📈 Features Used: {model_summary['n_features']}")
    for feature in model_summary['features']:
        print(f"   • {feature}")
    
    print(f"\n🔍 Explainability:")
    print(f"   Method: SHAP (SHapley Additive exPlanations)")
    print(f"   Trust Level: {trust_report['trust_level']}")
    
    print(f"\n📁 Output Files Generated:")
    output_files = sorted(OUTPUT_DIR.glob("*.png"))
    for i, file in enumerate(output_files, 1):
        print(f"   {i}. {file.name}")
    
    print(f"\n💾 Outputs saved to: {OUTPUT_DIR.absolute()}")
    
    print("\n" + "="*70)
    print("🚀 Pipeline execution completed successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print(f"\n❌ Error: Dataset not found at {DATASET_PATH}")
        print("Please ensure the dataset CSV file exists at the specified path.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
