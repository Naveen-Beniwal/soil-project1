"""
Evaluation Metrics & Comparative Analysis Module
Computes classification & regression metrics:
Accuracy, Precision, Recall, F1-score, ROC-AUC, RMSE, MAE, R² Score.
Generates confusion matrices and comparative benchmark charts across all models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)


class ModelEvaluator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
        """
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        roc_auc = None
        if y_prob is not None:
            try:
                # Multi-class OvR ROC-AUC
                roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            except Exception:
                roc_auc = None

        metrics = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(roc_auc), 4) if roc_auc is not None else 0.0
        }

        # Plot Confusion Matrix
        if class_names is not None:
            self.plot_confusion_matrix(y_true, y_pred, class_names, model_name)

        return metrics

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes RMSE, MAE, and R² Score for Soil Health Index regression.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "RMSE": round(float(rmse), 4),
            "MAE": round(float(mae), 4),
            "R2_Score": round(float(r2), 4)
        }

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        model_name: str
    ) -> str:
        """Saves a styled confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_path = self.output_dir / f"cm_{safe_name}.png"
        fig.savefig(save_path)
        plt.close(fig)
        return str(save_path)

    def generate_comparative_report(
        self,
        results_dict: Dict[str, Dict[str, float]],
        filename_prefix: str = "model_comparison"
    ) -> pd.DataFrame:
        """
        Creates a comparative performance DataFrame, CSV, and summary bar chart.
        """
        df = pd.DataFrame.from_dict(results_dict, orient="index")
        csv_path = self.output_dir / f"{filename_prefix}.csv"
        df.to_csv(csv_path)

        # Plot Comparative Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        metrics_to_plot = [m for m in ["Accuracy", "F1_Score", "Precision", "Recall"] if m in df.columns]
        df[metrics_to_plot].plot(kind="bar", ax=ax, colormap="viridis", width=0.8)
        ax.set_title("Comparative Evaluation: Classical ML vs Base CNN vs Proposed UAV-SoilHealthNet",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Score (0.0 to 1.0)", fontsize=10)
        ax.set_ylim([0.6, 1.05])
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()

        chart_path = self.output_dir / f"{filename_prefix}_chart.png"
        fig.savefig(chart_path)
        plt.close(fig)

        return df
