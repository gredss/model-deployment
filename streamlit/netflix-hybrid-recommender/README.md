# Obesity Level Prediction Based on Lifestyle Habits

## 📊 Dataset Description

The dataset comprises **1,056 samples** with **17 input features** and **1 target variable** (`NObeyesdad`). The target variable represents one of seven obesity classes:

- `Insufficient_Weight`
- `Normal_Weight`
- `Overweight_Level_I`
- `Overweight_Level_II`
- `Obesity_Type_I`
- `Obesity_Type_II`
- `Obesity_Type_III`

### Input Features

| Feature | Description |
|--------|-------------|
| `Gender` | Biological sex (male/female) |
| `Age` | Age (in years) |
| `Height` | Height (in meters) |
| `Weight` | Weight (in kilograms) |
| `family_history_with_overweight` | Family history of overweight (yes/no) |
| `FAVC` | Frequent consumption of high-calorie food (yes/no) |
| `FCVC` | Frequency of vegetable consumption (ordinal: 1–3) |
| `NCP` | Number of main meals per day |
| `CAEC` | Eating habits outside of main meals |
| `SMOKE` | Smoking habit (yes/no) |
| `CH2O` | Daily water intake (ordinal: 1–3) |
| `SCC` | Caloric intake monitoring (yes/no) |
| `FAF` | Weekly physical activity (ordinal: 0–3) |
| `TUE` | Daily screen time (ordinal: 0–3) |
| `CALC` | Alcohol consumption |
| `MTRANS` | Primary mode of transportation |
| `NObeyesdad` | Target obesity level class |

---

## ⚙️ Methodology

### 1. **Exploratory Data Analysis**
- Overview of data types, missing values, and outlier detection
- Histograms for numerical distributions
- Target class distribution visualization

### 2. **Data Cleaning**
- Extracted numeric values from string-based `Age` entries (e.g., "20 years")
- Standardized invalid entries and ensured proper data types

### 3. **Dataset Splitting**
- Train-test split (80:20) using stratified sampling based on the target variable

### 4. **Preprocessing Pipeline**
Feature preprocessing was handled using a `ColumnTransformer`:
- **Numerical features**: Median imputation + Standard Scaling
- **Ordinal features**: Mode imputation + Ordinal Encoding
- **Categorical features**: Mode imputation + One-Hot Encoding

### 5. **Model Training**
- Classifier used: `RandomForestClassifier`
- Integrated into a complete pipeline combining preprocessing and classification

### 6. **Evaluation Metrics**
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **93.0%** |
| Macro F1-score | **0.93** |
| Weighted F1-score | **0.93** |

- The model achieved **perfect precision and recall (1.00)** for the `Insufficient_Weight` class.
- It demonstrated strong generalization performance across high-obesity classes (`Obesity_Type_II`, `Obesity_Type_III`).
- Slightly lower F1-scores were observed for `Overweight_Level_II` and `Normal_Weight`, though still above 0.86.

These results indicate that the model is highly effective in classifying individuals into the correct obesity category based on lifestyle and demographic data. It is suitable for integration into clinical decision support systems or digital health platforms focused on obesity risk assessment.

---

## 💾 Model Saving

The trained pipeline is exported for deployment:

```python
import joblib
joblib.dump(model, 'obesity_model.pkl')
````

---

## 🚀 API Deployment using FastAPI

### `app.py` – FastAPI Inference Script

### Run the API

```bash
uvicorn app:app --reload
```

### Sample cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "Gender": "Male",
  "Age": 21,
  "Height": 1.75,
  "Weight": 85,
  "family_history_with_overweight": "yes",
  "FAVC": "yes",
  "FCVC": 2.0,
  "NCP": 3.0,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2.0,
  "SCC": "yes",
  "FAF": 1.0,
  "TUE": 1.0,
  "CALC": "Sometimes",
  "MTRANS": "Public_Transportation"
}'
```

---

## 📦 Requirements

```txt
fastapi
uvicorn
scikit-learn==1.6.1   
pandas
joblib
streamlit
requests
matplotlib
```

---

## 📚 License

This project is for academic and educational purposes only.
