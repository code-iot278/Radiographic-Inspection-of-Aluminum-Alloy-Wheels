import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import random

# Load your input CSV
input_path = "/content/drive/MyDrive/Colab Notebooks/Wheel/features_lightweight.csv"
df = pd.read_csv(input_path)
# --------------------------
# Step 2: Select only numeric feature columns
# --------------------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
X = df[numeric_cols].values

# --------------------------
# Step 3: Run DBSCAN
# --------------------------
db = DBSCAN(eps=3, min_samples=5).fit(X)
df['Cluster'] = db.labels_

# Add random labels (Defect / Normal) to all rows
df["label"] = [random.choice(["Defect", "Normal"]) for _ in range(len(df))]

# Save updated CSV with all input columns + label
output_path = "/content/drive/MyDrive/Colab Notebooks/Wheel/labels.csv"
df.to_csv(output_path, index=False)

print(f"CSV file saved at: {output_path}")
df.head()