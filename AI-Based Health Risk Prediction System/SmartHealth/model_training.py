"""
🤖 Model Training & Preprocessing Module
Handles data preprocessing and XGBoost model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class HealthRiskModel:
    """XGBoost model for health risk classification"""
    
    def __init__(self, target_col='disease_risk', test_size=0.2, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        
        # Components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
        # Data
        self.X_train = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
    
    def preprocess(self, df):
        """Preprocessing: encode categorical features and prepare data"""
        print("\n" + "="*60)
        print("🔧 DATA PREPROCESSING")
        print("="*60)
        
        df_processed = df.copy()
        
        # Identify categorical columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical features
        for col in categorical_cols:
            if col != self.target_col:  # Don't encode target yet
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                print(f"✓ Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Separate features and target
        # Remove non-feature columns (id, target, and individual disease columns)
        drop_cols = [self.target_col, 'heart_disease', 'diabetes', 'stroke', 'health_risk_score']
        drop_cols = [col for col in drop_cols if col in df_processed.columns]
        
        X = df_processed.drop(drop_cols, axis=1)
        y = df_processed[self.target_col]
        
        self.feature_names = X.columns.tolist()
        print(f"\n✓ Features selected: {len(X.columns)}")
        print(f"  Features: {self.feature_names}")
        print(f"✓ Target: {self.target_col}")
        print(f"  Classes: {sorted(y.unique())}")
        
        return X, y
    
    def split_and_scale(self, X, y):
        """Split data and scale numerical features"""
        print("\n" + "="*60)
        print("📊 DATA SPLITTING & SCALING")
        print("="*60)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Training set: {self.X_train.shape}")
        print(f"✓ Test set: {self.X_test.shape}")
        print(f"✓ Train-Test split: {(1-self.test_size)*100:.0f}%-{self.test_size*100:.0f}%")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n✓ Scaling applied (StandardScaler)")
        print(f"  Mean: {self.scaler.mean_}")
        print(f"  Std: {self.scaler.scale_}")
    
    def train(self, n_estimators=200, learning_rate=0.1, max_depth=6):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("🚀 MODEL TRAINING (XGBoost)")
        print("="*60)
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            verbosity=1
        )
        
        print(f"Model Configuration:")
        print(f"  - n_estimators: {n_estimators}")
        print(f"  - learning_rate: {learning_rate}")
        print(f"  - max_depth: {max_depth}")
        
        self.model.fit(self.X_train_scaled, self.y_train)
        print("\n✓ Model training completed!")
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*60)
        print("📈 MODEL EVALUATION")
        print("="*60)
        
        self.y_pred = self.model.predict(self.X_test_scaled)
        
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
        print(f"   Target: ≥90%")
        
        if accuracy >= 0.90:
            print("   ✅ Target achieved!")
        else:
            print("   ⚠️  Below target - consider hyperparameter tuning")
        
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        return accuracy
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix"""
        if self.y_pred is None:
            print("Model not evaluated yet!")
            return
        
        print("\n[EVALUATION] Creating confusion matrix...")
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        plt.title("Confusion Matrix - Health Risk Classification", fontsize=14, fontweight='bold')
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """Plot feature importance from XGBoost"""
        print(f"\n[EVALUATION] Creating feature importance plot (top {top_n})...")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f"Top {top_n} Feature Importance (XGBoost)", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
        
        print(f"\n📊 Feature Importance Ranking:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    def get_model_summary(self):
        """Get model summary"""
        return {
            'accuracy': accuracy_score(self.y_test, self.y_pred) * 100,
            'features': self.feature_names,
            'n_features': len(self.feature_names),
            'classes': sorted(self.y_test.unique()),
            'train_size': self.X_train.shape[0],
            'test_size': self.X_test.shape[0]
        }
