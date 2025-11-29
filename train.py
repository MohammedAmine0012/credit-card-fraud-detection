import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

# Create Results directory if it doesn't exist
os.makedirs("Results", exist_ok=True)

# Set random seed for reproducibility
RANDOM_STATE = 125

# Load and prepare the data
print("Loading data...")
df = pd.read_csv("Data/creditcard.csv/creditcard.csv")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data
print("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Print class distribution
print("\nClass distribution:")
print("Training set:", pd.Series(y_train).value_counts().to_dict())
print("Test set:", pd.Series(y_test).value_counts().to_dict())

# Create pipeline with SMOTE for handling class imbalance
print("\nCreating and training model...")
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
print("\nEvaluating model...")
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print metrics
print(f"\nModel Performance:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"ROC AUC: {test_roc_auc:.4f}")

# Save metrics to file
with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {test_accuracy:.4f}\n")
    f.write(f"F1 Score: {test_f1:.4f}\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall: {test_recall:.4f}\n")
    f.write(f"ROC AUC: {test_roc_auc:.4f}")

# Plot and save confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Genuine', 'Fraud'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('Results/confusion_matrix.png', dpi=300, bbox_inches='tight')

# Save the model
print("\nSaving model...")
os.makedirs("Model", exist_ok=True)
joblib.dump(pipeline, 'Model/credit_card_fraud_model.joblib')

print("\nTraining and evaluation completed successfully!")
