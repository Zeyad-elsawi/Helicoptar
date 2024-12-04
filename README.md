### **README for Anomaly Detection using PCA and Isolation Forest**

This project aims to detect anomalies in a high-dimensional dataset using dimensionality reduction (PCA) and Isolation Forest. Below is a detailed explanation of the process and the code provided.

---

### **1. Project Overview**
- The dataset consists of numerical features representing certain characteristics (e.g., sensor readings, system logs).
- The goal is to identify anomalies (outliers) in the data.
- Techniques used:
  - **StandardScaler**: To normalize data.
  - **Principal Component Analysis (PCA)**: To reduce dimensionality for visualization and computational efficiency.
  - **Isolation Forest**: A machine learning model specialized for anomaly detection.

---

### **2. Requirements**
#### Libraries
- **NumPy**: For numerical operations.
- **Pandas**: For handling datasets.
- **Matplotlib**: For data visualization.
- **scikit-learn**:
  - `StandardScaler` for scaling features.
  - `PCA` for dimensionality reduction.
  - `IsolationForest` for anomaly detection.
- **TensorFlow** (optional): For deeper anomaly detection models (not used explicitly here).

---

### **3. Data Preparation**
- **Input Datasets**:
  - `train_df.csv`: Training dataset without labels.
  - `validation_df.csv`: Validation dataset with a column `anomaly` indicating ground truth (1 = anomaly, 0 = normal).
  
- **Steps**:
  1. Load the datasets using `pandas.read_csv()`.
  2. Normalize features using `StandardScaler` to ensure each feature has a mean of 0 and variance of 1.
  3. Reduce dimensions using PCA, retaining 2 components for visualization.

---

### **4. Anomaly Detection**
- **Isolation Forest**:
  - Detects anomalies by isolating points in the feature space.
  - Configured with:
    - `contamination=0.1`: Proportion of expected anomalies in the data.
    - `random_state=42`: Ensures reproducibility.
  - Training:
    - Model is fit on the reduced training data (`train_reduced`).
  - Prediction:
    - Predictions are made on the validation set.
    - Conversion: `1` for normal, `-1` for anomalies → `0` for normal, `1` for anomalies.

---

### **5. Visualization**
- Scatter plot of PCA-reduced data:
  - **Blue points**: Normal predictions.
  - **Red points**: Predicted anomalies.
- Helpful for understanding the separation in the reduced feature space.

---

### **6. Evaluation**
- **Metrics**:
  - **Accuracy**: Proportion of correct predictions.
  - **Precision** and **Recall**: (Optional) Evaluate anomaly detection effectiveness.
  
---

### **7. Key Code Components**
#### Data Loading
```python
train = pd.read_csv('train_df.csv')
validation = pd.read_csv('validation_df.csv')
validation_labels = validation['anomaly']
```

#### Data Scaling and PCA
```python
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
validation_scaled = scaler.transform(validation.drop(columns=['anomaly']))

pca = PCA(n_components=2)
train_reduced = pca.fit_transform(train_scaled)
validation_reduced = pca.transform(validation_scaled)
```

#### Isolation Forest
```python
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
isolation_forest.fit(train_reduced)

validation_predictions = isolation_forest.predict(validation_reduced)
validation_predictions = np.where(validation_predictions == 1, 0, 1)
```

#### Visualization
```python
plt.scatter(validation_reduced[:, 0], validation_reduced[:, 1], c='blue', label='Normal Predictions')
plt.scatter(validation_reduced[index, 0], validation_reduced[index, 1], c='red', label='Anomalies')
plt.title("PCA Reduced Data with Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

#### Evaluation
```python
accuracy = accuracy_score(validation_labels, validation_predictions)
print(f'Accuracy on Validation Set: {accuracy * 100:.2f}%')
```

---

### **8. Outputs**
- **Scatter Plot**: Visualization of anomalies vs. normal predictions.
- **Accuracy Score**: Measures model performance on the validation set.

---

### **9. Limitations**
- PCA reduces dimensionality, which may cause loss of information.
- Isolation Forest assumes a certain contamination rate, which may not match the true data distribution.

---

