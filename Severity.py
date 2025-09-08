import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

# ------------------------------
# 1. Load CSV
# ------------------------------
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Wheel/labels.csv')
label_col = 'label'

# ------------------------------
# 2. Map labels to severity
# ------------------------------
def map_severity(value):
    value = str(value).lower()
    if value == 'normal':
        return 'Low'
    elif value == 'defect':
        return 'High'
    else:
        return 'Moderate'

df['Severity_Label'] = df[label_col].apply(map_severity)

# ------------------------------
# 3. Encode severity labels with all classes
# ------------------------------
le = LabelEncoder()
le.fit(['Low', 'Moderate', 'High'])  # Ensure all three classes exist
df['Severity_Code'] = le.transform(df['Severity_Label'])

# ------------------------------
# 4. Prepare features
# ------------------------------
X = df.drop(columns=[label_col, 'Severity_Label', 'Severity_Code'])
y = df['Severity_Code']

# Encode categorical/object columns automatically
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# ------------------------------
# 5. Add synthetic rows if a class is missing in y_train
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Find missing classes in training
missing_classes = set(range(3)) - set(y_train)
for cls in missing_classes:
    # Add one synthetic row (all zeros)
    X_train = pd.concat([X_train, pd.DataFrame([np.zeros(X_train.shape[1])], columns=X_train.columns)], ignore_index=True)
    y_train = pd.concat([y_train, pd.Series([cls])], ignore_index=True)

# ------------------------------
# 6. Train LightGBM
# ------------------------------
model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)

# ------------------------------
# 7. Predict for all rows
# ------------------------------
pred_labels = model.predict(X)
df['Predicted_Severity_Code'] = pred_labels
df['Predicted_Severity_Label'] = le.inverse_transform(pred_labels)

# ------------------------------
# 8. Save output
# ------------------------------
output_path = 'output_with_predicted_severity.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved CSV with predicted severity: {output_path}")

# ------------------------------
# 9. Display final result
# ------------------------------
print(df[['Severity_Label', 'Predicted_Severity_Label']].head())
df.head()