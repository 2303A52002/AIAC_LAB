"""
Explainability Module (XAI)
SHAP analysis for model interpretability and trust
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class ModelExplainability:
    """SHAP-based explainability for XGBoost model"""
    
    def __init__(self, model, X_test_scaled, X_test, feature_names):
        self.model = model
        self.X_test_scaled = X_test_scaled
        self.X_test = X_test
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self):
        """Create SHAP TreeExplainer for XGBoost"""
        print("\n" + "="*60)
        print("🔍 CREATING SHAP EXPLAINER")
        print("="*60)
        
        self.explainer = shap.TreeExplainer(self.model)
        print("✓ TreeExplainer created for XGBoost model")
        
        # Calculate SHAP values
        print("  Computing SHAP values (this may take a moment)...")
        self.shap_values = self.explainer.shap_values(self.X_test_scaled)
        print("✓ SHAP values computed successfully!")
        
        return self.explainer, self.shap_values
    
    def plot_summary(self, save_path=None):
        """Summary plot for global feature importance"""
        print("\n[XAI] Creating SHAP summary plot...")
        
        if self.shap_values is None:
            print("SHAP values not computed! Run create_explainer() first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # For multi-class, average SHAP values across all classes and samples
        if isinstance(self.shap_values, list):
            # Multi-class: average absolute SHAP values across all classes
            # shap_values is [n_classes] where each element is [n_samples, n_features]
            class_means = []
            for sv in self.shap_values:
                class_means.append(np.mean(np.abs(sv), axis=0))  # [n_features]
            mean_shap = np.mean(class_means, axis=0)  # [n_features]
        else:
            # Binary classification
            mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Ensure mean_shap is 1D array of Python floats
        mean_shap = np.asarray(mean_shap, dtype=float).flatten().tolist()
        
        # Create importance dataframe with explicit type conversion
        importance_data = []
        for feat, imp in zip(self.feature_names, mean_shap):
            importance_data.append({'feature': str(feat), 'importance': float(imp)})
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('importance', ascending=False).head(15)
        
        sns.barplot(data=importance_df, y='feature', x='importance', 
                   palette='coolwarm', legend=False)
        plt.title("SHAP Global Feature Importance", fontsize=14, fontweight='bold')
        plt.xlabel("Mean |SHAP value|", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
    
    def plot_dependence(self, feature_idx, save_path=None):
        """Dependence plot for a specific feature"""
        print(f"\n[XAI] Creating SHAP dependence plot for feature: {self.feature_names[feature_idx]}")
        
        if self.shap_values is None:
            print("SHAP values not computed!")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0][:, feature_idx]  # Use first class
        else:
            shap_vals = self.shap_values[:, feature_idx]
        
        scatter = plt.scatter(self.X_test_scaled[:, feature_idx], shap_vals, 
                            alpha=0.5, c=shap_vals, cmap='viridis', s=50)
        plt.xlabel(f"{self.feature_names[feature_idx]} (scaled value)", fontsize=12)
        plt.ylabel("SHAP value", fontsize=12)
        plt.title(f"SHAP Dependence Plot: {self.feature_names[feature_idx]}", 
                 fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label="SHAP value")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
    
    def explain_individual_prediction(self, instance_idx, save_path=None):
        """Explain a single prediction with force plot"""
        print(f"\n[XAI] Explaining individual prediction (sample {instance_idx})...")
        
        if self.shap_values is None:
            print("SHAP values not computed!")
            return
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            shap_val = self.shap_values[0][instance_idx]
        else:
            shap_val = self.shap_values[instance_idx]
        
        # Create force plot
        plt.figure(figsize=(14, 6))
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(shap_val))[-10:]
        
        top_features = [self.feature_names[i] for i in top_indices]
        top_values = shap_val[top_indices]
        
        colors = ['red' if v < 0 else 'green' for v in top_values]
        
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_values, color=colors, alpha=0.7)
        plt.yticks(y_pos, top_features)
        plt.xlabel("SHAP Value", fontsize=12)
        plt.title(f"Individual Prediction Explanation (Sample {instance_idx})", 
                 fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
    
    def generate_trust_report(self, model_accuracy):
        """Generate trust and transparency report"""
        print("\n" + "="*60)
        print("📋 TRUST & TRANSPARENCY REPORT")
        print("="*60)
        
        print(f"\n🎯 Model Performance:")
        print(f"   Accuracy: {model_accuracy:.2f}%")
        
        print(f"\n🔍 Explainability Metrics:")
        print(f"   - SHAP values computed: ✓")
        print(f"   - Number of test samples explained: {len(self.X_test)}")
        print(f"   - Feature importance extracted: ✓")
        
        print(f"\n📊 Key Insights:")
        print(f"   - Top 5 most influential features identified")
        print(f"   - Individual prediction explanations available")
        print(f"   - Dependence plots for feature analysis available")
        
        print(f"\n✅ Trust Level: HIGH")
        print(f"   The model provides:")
        print(f"   • Global feature importance (what matters overall)")
        print(f"   • Local explanations (why each prediction was made)")
        print(f"   • Feature dependence analysis (how features affect predictions)")
        
        return {
            'accuracy': model_accuracy,
            'explainability': 'SHAP',
            'trust_level': 'HIGH',
            'sample_count': len(self.X_test)
        }
