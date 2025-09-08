# ------------------------------
# 1. Imports
# ------------------------------
import pandas as pd
import numpy as np
from scipy.stats import entropy
from river.drift import ADWIN  # ADaptive WINdowing for drift detection
import json
import datetime

# ------------------------------
# 2. Load CSV
# ------------------------------
csv_path = 'output_with_predicted_severity.csv'  # replace with your path
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # remove spaces from column names

# ------------------------------
# 3. Determine severity column
# ------------------------------
if 'Predicted_Severity_Label' in df.columns:
    severity_col = 'Predicted_Severity_Label'
elif 'Severity_Label' in df.columns:
    severity_col = 'Severity_Label'
else:
    raise ValueError("No severity column found in CSV.")

# ------------------------------
# 4. Cost-aware decision table
# ------------------------------
decision_table_json = """
{
    "crack": {
        "Low": {"surface": "Repair", "subsurface": "Repair"},
        "Moderate": {"surface": "Repair", "subsurface": "Reject"},
        "High": {"surface": "Reject", "subsurface": "Reject"}
    },
    "porosity": {
        "Low": {"surface": "Repair", "subsurface": "Repair"},
        "Moderate": {"surface": "Reject", "subsurface": "Reject"},
        "High": {"surface": "Reject", "subsurface": "Reject"}
    },
    "inclusion": {
        "Low": {"surface": "Accept", "subsurface": "Reject"},
        "Moderate": {"surface": "Reject", "subsurface": "Reject"},
        "High": {"surface": "Reject", "subsurface": "Reject"}
    }
}
"""
decision_table = json.loads(decision_table_json)

# ------------------------------
# 5. Recommendation engine
# ------------------------------
def get_action(row):
    if str(row['label']).lower() == 'normal':
        return "OK"  # tyre condition okay, no action
    defect = str(row['label']).lower()
    severity = str(row[severity_col])
    location = str(row['Cluster']).lower()
    try:
        return decision_table[defect][severity][location]
    except KeyError:
        return "Accept"

df['action'] = df.apply(get_action, axis=1)

# ------------------------------
# 6. Repair planner
# ------------------------------
def repair_plan(row):
    if row['action'] != "Repair":
        return None
    defect = str(row['label']).lower()
    if defect == "crack":
        return "Short weld + local machining"
    elif defect == "porosity":
        return "Fill + heat treat if within spec"
    elif defect == "inclusion":
        return "Manual inspection"
    else:
        return "Custom repair"

df['repair_plan'] = df.apply(repair_plan, axis=1)

# ------------------------------
# 7. Explainability
# ------------------------------
feature_cols = ['area','circularity','aspect_ratio','mean_intensity','median_intensity',
                'local_contrast','edge_density','frangi_mean','frangi_var',
                'mv_mean','mv_var','rg_mean','rg_var']
feature_cols = [c for c in feature_cols if c in df.columns]

# Ensure all features are numeric
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

def explain_features(row):
    features = row[feature_cols].values.astype(float)
    if features.sum() > 0:
        probs = features / features.sum()
        ent = entropy(probs)
    else:
        probs = np.zeros_like(features)
        ent = 0
    top3_idx = np.argsort(-features)[:3]
    top3 = [feature_cols[i] for i in top3_idx]
    return pd.Series([top3, probs, ent], index=['top3_features','feature_probs','entropy'])

df[['top3_features','feature_probs','entropy']] = df.apply(explain_features, axis=1)

# ------------------------------
# 8. Lifecycle monitoring with ADWIN
# ------------------------------
adwin = ADWIN(delta=0.002)  # sensitivity for drift detection

# Map severity labels to numeric scores for ADWIN
severity_mapping = {'Low': 1, 'Moderate': 2, 'High': 3, 'OK': 0}
df['severity_score'] = df[severity_col].map(severity_mapping).fillna(0)

def lifecycle_monitoring(df, adwin_detector):
    logs = []
    for idx, row in df.iterrows():
        # Simulate periodic NDT measurement with slight noise
        ndt_measurement = row['severity_score'] + np.random.normal(0, 0.1)

        # ADWIN returns True if drift detected
        drift_flag = adwin_detector.update(ndt_measurement)

        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'filename': row['filename'],
            'label': row['label'],
            'predicted_severity': row[severity_col],
            'action': row['action'],
            'ndt_score': ndt_measurement,
            'drift_flag': drift_flag
        }
        logs.append(log_entry)

        if drift_flag:
            print(f"⚠️ Drift detected for {row['filename']} at {log_entry['timestamp']}")

    return pd.DataFrame(logs)

lifecycle_logs = lifecycle_monitoring(df, adwin)

# ------------------------------
# 9. Save outputs
# ------------------------------
df.to_csv('output_recommendation_full.csv', index=False)
df.to_csv('lifecycle_monitoring_logs.csv', index=False)
print("Saved CSVs: 'output_recommendation_full.csv', 'lifecycle_monitoring_logs.csv'")

# ------------------------------
# 10. Optional summary report
# ------------------------------
summary = lifecycle_logs.groupby(['predicted_severity','action']).size().reset_index(name='count')
print("\nSummary of lifecycle decisions:")
print(summary)

# ------------------------------
# 11. Display first rows for verification
# ------------------------------
print(df[['filename','label',severity_col,'action','repair_plan','top3_features','feature_probs','entropy']].head())
df=pd.read_csv('lifecycle_monitoring_logs.csv')
df