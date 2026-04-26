"""
Create an exploratory data analysis module for health risk prediction.
Include functions to:
1. Load the health dataset (CSV format with 5000 records)
2. Plot correlation heatmap for numeric features
3. Create scatter plot comparing BMI vs cholesterol
4. Generate feature distribution histograms
5. Visualize target variable (disease_risk) distribution
6. Generate comprehensive EDA report with statistics

Features to analyze: age, gender, BMI, exercise_level, smoking, alcohol,
blood_pressure, cholesterol, glucose, fatigue, chest_pain, dizziness

Output: PNG visualizations saved to outputs/ directory
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(dataset_path):
    """Load the health and lifestyle dataset"""
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nDataset info:\n{df.info()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    return df

def plot_correlation_heatmap(df, save_path=None):
    """📊 Visualization: Correlation Heatmap"""
    print("\n[EDA] Creating correlation heatmap...")
    plt.figure(figsize=(12, 8))
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', 
                cbar_kws={'label': 'Correlation'}, square=True)
    plt.title("Feature Correlation with Health Risk", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_bmi_sleep_scatter(df, target_col='disease_risk', save_path=None):
    """📊 Distribution: BMI vs Cholesterol"""
    print("\n[EDA] Creating BMI vs Cholesterol scatter plot...")
    plt.figure(figsize=(10, 6))
    
    # Sample data if too large
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    sns.scatterplot(data=sample_df, x='bmi', y='cholesterol', hue=target_col, 
                    palette='Set1', s=80, alpha=0.6)
    plt.title("Impact of BMI and Cholesterol on Health Risk Level", fontsize=14, fontweight='bold')
    plt.xlabel("BMI", fontsize=12)
    plt.ylabel("Cholesterol", fontsize=12)
    plt.legend(title="Risk Level", loc='best')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_feature_distributions(df, features, save_path=None):
    """📊 Distribution plots for key lifestyle metrics"""
    print("\n[EDA] Creating feature distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features[:4]):
        if feature in df.columns:
            sns.histplot(data=df, x=feature, bins=30, kde=True, ax=axes[idx], color='steelblue')
            axes[idx].set_title(f"Distribution of {feature.capitalize()}", fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def plot_disease_risk_distribution(df, target_col='disease_risk', save_path=None):
    """📊 Target variable distribution"""
    print("\n[EDA] Creating disease risk distribution...")
    plt.figure(figsize=(10, 6))
    
    risk_counts = df[target_col].value_counts().sort_index()
    colors = ['green', 'orange', 'red'][:len(risk_counts)]
    
    sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=colors)
    plt.title(f"Distribution of {target_col.replace('_', ' ').title()}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Risk Level", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Add value labels on bars
    for i, v in enumerate(risk_counts.values):
        plt.text(i, v + 100, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def generate_eda_report(df):
    """Generate comprehensive EDA report"""
    print("\n" + "="*60)
    print("📋 EXPLORATORY DATA ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📌 Dataset Dimensions: {df.shape[0]} records × {df.shape[1]} features")
    print(f"\n📌 Missing Values:\n{df.isnull().sum()}")
    print(f"\n📌 Data Types:\n{df.dtypes}")
    print(f"\n📌 Numeric Columns Summary:\n{df.describe()}")
    
    return df
