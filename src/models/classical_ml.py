"""
Classical Machine Learning Models for Soil Health Assessment
Implements Support Vector Machine (SVM), Random Forest (RF), Decision Tree (DT),
and Gradient Boosting (GBM/XGBoost).
Provides comparative benchmarking against Deep Learning models as mandated in SwaYaan PoC Objective 3.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score


class ClassicalSoilModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Classification Models
        self.classifiers = {
            "Decision Tree": DecisionTreeClassifier(
                max_depth=12, min_samples_split=5, random_state=random_state
            ),
            "Support Vector Machine (SVM)": SVC(
                kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=15, min_samples_split=4, n_jobs=-1, random_state=random_state
            ),
            "Gradient Boosting (GBM)": GradientBoostingClassifier(
                n_estimators=120, learning_rate=0.08, max_depth=5, random_state=random_state
            )
        }

        # Regression Models for Soil Health Index (SHI)
        self.regressors = {
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=random_state
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=random_state
            ),
            "Support Vector Regressor (SVR)": SVR(
                kernel="rbf", C=5.0, epsilon=0.05
            )
        }

        self.fitted_classifiers: Dict[str, Any] = {}
        self.fitted_regressors: Dict[str, Any] = {}

    def fit_and_evaluate_classifiers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Fits all classical classification models on extracted features and evaluates performance metrics.
        """
        # Fit scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        for name, clf in self.classifiers.items():
            clf.fit(X_train_scaled, y_train)
            self.fitted_classifiers[name] = clf
            
            y_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results[name] = {
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1_Score": round(float(f1), 4)
            }

        return results

    def fit_and_evaluate_regressors(
        self,
        X_train: np.ndarray,
        y_train_shi: np.ndarray,
        X_test: np.ndarray,
        y_test_shi: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Fits regression models for Soil Health Index (0-100 score).
        """
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        reg_results = {}

        for name, reg in self.regressors.items():
            reg.fit(X_train_scaled, y_train_shi)
            self.fitted_regressors[name] = reg

            y_pred = reg.predict(X_test_scaled)
            mse = mean_squared_error(y_test_shi, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_shi, y_pred)

            reg_results[name] = {
                "RMSE": round(float(rmse), 4),
                "MAE": round(float(np.mean(np.abs(y_test_shi - y_pred))), 4),
                "R2_Score": round(float(r2), 4)
            }

        return reg_results

    def predict(self, feature_vector: np.ndarray, model_name: str = "Random Forest") -> Tuple[int, np.ndarray]:
        """
        Predicts class label and probabilities for a single 1D feature vector or batch.
        """
        if model_name not in self.fitted_classifiers:
            raise ValueError(f"Model {model_name} is not trained yet.")
        
        clf = self.fitted_classifiers[model_name]
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        scaled = self.scaler.transform(feature_vector)
        pred = clf.predict(scaled)[0]
        probs = clf.predict_proba(scaled)[0]
        return int(pred), probs

    def save(self, output_dir: str):
        """Saves fitted models and scaler to disk."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_dir, "feature_scaler.joblib"))
        for name, clf in self.fitted_classifiers.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            joblib.dump(clf, os.path.join(output_dir, f"{safe_name}_clf.joblib"))
        for name, reg in self.fitted_regressors.items():
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            joblib.dump(reg, os.path.join(output_dir, f"{safe_name}_reg.joblib"))

    def load(self, output_dir: str):
        """Loads fitted models and scaler from disk."""
        scaler_path = os.path.join(output_dir, "feature_scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        for name in list(self.classifiers.keys()):
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(output_dir, f"{safe_name}_clf.joblib")
            if os.path.exists(path):
                self.fitted_classifiers[name] = joblib.load(path)
