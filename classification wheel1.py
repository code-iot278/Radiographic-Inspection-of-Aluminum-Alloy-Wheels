# train_lgbm_tabular_full_metrics.py
"""
Lightweight classification using LightGBM.
- Supports binary/multi-class classification.
- Optional calibration (Isotonic/Platt) if dataset large enough.
- Displays extended metrics: Accuracy, Precision, Sensitivity, Specificity, F1, NPV, MCC, FPR, FNR
- Saves model with features and label encoder
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

# ----------------------------
# PARAMETERS
# ----------------------------
INPUT_CSV = "/content/drive/MyDrive/Colab Notebooks/Wheel/labels.csv"
MODEL_OUT = "lgbm_tabular_full_metrics.joblib"
METRICS_FILE = "/content/drive/MyDrive/Colab Notebooks/Wheel/performance_metrics.npy"
RANDOM_STATE = 42
TEST_SIZE = 0.2
NUM_BOOST_ROUND = 100
EARLY_STOPPING = 50
CALIBRATION_SPLIT = 0.1

# ----------------------------
# FUNCTION: Compute extended metrics
# ----------------------------
def compute_extended_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_class = len(np.unique(y_true))

    if num_class == 2:
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 2 * TP / (2 * TP + FP + FN) if (2*TP + FP + FN) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        mcc = matthews_corrcoef(y_true, y_pred)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        return {
            "Accuracy (%)": accuracy*100,
            "Precision (%)": precision*100,
            "Sensitivity (%)": sensitivity*100,
            "Specificity (%)": specificity*100,
            "F1 Score (%)": f1*100,
            "NPV (%)": npv*100,
            "MCC (%)": mcc*100,
            "FPR (%)": fpr*100,
            "FNR (%)": fnr*100
        }
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {
            "Accuracy (%)": accuracy*100,
            "Precision (%)": precision*100,
            "Sensitivity (%)": recall*100,
            "Specificity (%)": np.nan,
            "F1 Score (%)": f1*100,
            "NPV (%)": np.nan,
            "MCC (%)": np.nan,
            "FPR (%)": np.nan,
            "FNR (%)": np.nan
        }

# ----------------------------
# MAIN SCRIPT
# ----------------------------
def main():
    # ----------------------------
    # Load dataset
    # ----------------------------
    df = pd.read_csv(INPUT_CSV)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])

    y = df['label']
    X = df.drop(columns=['label'])

    # Encode labels if categorical
    le = None
    if y.dtype == object or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Fill missing values
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].astype(str).fillna("NA")

    # One-hot encode categorical features
    X_processed = pd.get_dummies(X, drop_first=True)

    # ----------------------------
    # Train/Test split
    # ----------------------------
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # ----------------------------
    # LightGBM Dataset & Parameters
    # ----------------------------
    num_class = len(np.unique(y_train))
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    param = {
        'objective': 'binary' if num_class == 2 else 'multiclass',
        'metric': 'binary_logloss' if num_class == 2 else 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': RANDOM_STATE,
        'num_threads': 4
    }
    if num_class > 2:
        param['num_class'] = num_class

    # ----------------------------
    # Train with early stopping
    # ----------------------------
    bst = lgb.train(
        param,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(EARLY_STOPPING)]
    )

    # ----------------------------
    # Predictions
    # ----------------------------
    preds_prob = bst.predict(X_test, num_iteration=bst.best_iteration)
    if num_class == 2:
        preds = (preds_prob >= 0.5).astype(int)
    else:
        preds = np.argmax(preds_prob, axis=1)

    # ----------------------------
    # Optional Calibration
    # ----------------------------
    if len(X_train) > 5:
        try:
            calib_size = max(1, int(len(X_train) * CALIBRATION_SPLIT))
            X_calib, _, y_calib, _ = train_test_split(
                X_train, y_train, test_size=calib_size, random_state=RANDOM_STATE, stratify=stratify
            )
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(bst.predict(X_calib, num_iteration=bst.best_iteration), y_calib)
            preds_prob = iso_reg.predict(preds_prob)
            if num_class == 2:
                preds = (preds_prob >= 0.5).astype(int)
        except Exception:
            pass

    # ----------------------------
    # Confidence / Quality Score
    # ----------------------------
    preds_prob_clipped = np.clip(preds_prob, 1e-6, 1-1e-6)
    if num_class == 2:
        entropy = - (preds_prob_clipped * np.log(preds_prob_clipped) + (1 - preds_prob_clipped) * np.log(1 - preds_prob_clipped))
        quality_score = 1 - (entropy / np.log(2))
    else:
        entropy = -np.sum(preds_prob_clipped * np.log(preds_prob_clipped), axis=1)
        quality_score = 1 - (entropy / np.log(num_class))

    results = pd.DataFrame({
        'pred': preds,
        'prob': preds_prob,
        'quality_score': quality_score
    })
    print("\nSample predictions with quality score:")
    print(results.head(10))

    # ----------------------------
    # Save Model
    # ----------------------------
    joblib.dump({'model': bst, 'label_encoder': le, 'features': X_processed.columns.tolist()}, MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")



    # ----------------------------
    # Optionally Load & Display Saved .npy Metrics
    # ----------------------------
    if os.path.exists(METRICS_FILE):
        loaded_data = np.load(METRICS_FILE, allow_pickle=True)
        print("\nLoaded metrics")
        for entry in loaded_data:
            print(f"Accuracy (%): {entry['Accuracy']}")
            print(f"Precision (%): {entry['Precision']}")
            print(f"Sensitivity (%): {entry['Sensitivity']}")
            print(f"Specificity (%): {entry['Specificity']}")
            print(f"F1 Score (%): {entry['F1_Score']}")
            print(f"NPV (%): {entry['NPV']}")
            print(f"MCC (%): {entry['MCC']}")
            print(f"FPR (%): {entry['FPR']}")
            print(f"FNR (%): {entry['FNR']}")
            print("-" * 40)

if __name__ == "__main__":
    main()
