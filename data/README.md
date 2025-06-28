# Data Directory

This directory contains the telecom customer churn datasets used for the analysis.

## 📁 Dataset Files

### Training Data
- **File:** `churn-bigml-80.csv`
- **Size:** 2,666 records
- **Purpose:** Model training and cross-validation
- **Usage:** 80% of the original dataset

### Testing Data  
- **File:** `churn-bigml-20.csv`
- **Size:** 667 records
- **Purpose:** Final model evaluation and performance assessment
- **Usage:** 20% of the original dataset (held-out test set)

## 📊 Dataset Information

### Source
- **Provider:** BigML
- **Domain:** Telecommunications
- **Type:** Customer behavior and churn prediction
- **Split:** Pre-split into train/test to ensure consistent evaluation

### Features (20 total)
| Feature | Type | Description |
|---------|------|-------------|
| State | Categorical | US state (50 unique values) |
| Area code | Categorical | Telephone area code |
| International plan | Binary | Yes/No international calling plan |
| Voice mail plan | Binary | Yes/No voice mail service |
| Number vmail messages | Numerical | Count of voice mail messages |
| Total day minutes | Numerical | Total minutes of day calls |
| Total day calls | Numerical | Total number of day calls |
| Total day charge | Numerical | Total charge for day calls |
| Total eve minutes | Numerical | Total minutes of evening calls |
| Total eve calls | Numerical | Total number of evening calls |
| Total eve charge | Numerical | Total charge for evening calls |
| Total night minutes | Numerical | Total minutes of night calls |
| Total night calls | Numerical | Total number of night calls |
| Total night charge | Numerical | Total charge for night calls |
| Total intl minutes | Numerical | Total minutes of international calls |
| Total intl calls | Numerical | Total number of international calls |
| Total intl charge | Numerical | Total charge for international calls |
| Customer service calls | Numerical | Number of customer service calls |
| **Churn** | **Binary** | **Target variable: True/False** |

### Data Quality
- ✅ **No missing values** in either dataset
- ✅ **Consistent schema** between train and test sets
- ✅ **Balanced geographical distribution** across US states
- ⚠️ **Class imbalance:** ~14.5% churn rate (minority class)

## 🔍 Key Statistics

### Target Variable Distribution
- **No Churn (False):** ~85.5%
- **Churn (True):** ~14.5%
- **Challenge:** Imbalanced classification problem

### Feature Characteristics
- **Numerical features:** 16 (usage patterns, charges, counts)
- **Categorical features:** 2 (geographical)
- **Binary features:** 2 (plan subscriptions)
- **Highly correlated:** Minutes and charges (expected business relationship)

## 🚀 Usage in Analysis

### Preprocessing Steps Applied
1. **Binary encoding** of Yes/No and True/False values
2. **One-hot encoding** of categorical features (State, Area code)
3. **Standard scaling** of numerical features
4. **Correlation analysis** and redundant feature removal
5. **Feature engineering** resulting in 67 final features

### Model Training Strategy
- **Training set:** Used for model fitting and hyperparameter tuning
- **Cross-validation:** 5-fold CV within training set for model selection
- **Test set:** Final evaluation only (no data leakage)
- **Evaluation focus:** F1-score due to class imbalance

## 📋 Data Ethics & Privacy

- **Anonymized data:** No personally identifiable information
- **Academic use:** Dataset used for educational/research purposes
- **Public availability:** Standard benchmark dataset in ML community
- **No sensitive attributes:** No protected class information included

## 🔄 Reproducing Analysis

To load and explore the data:

```python
import pandas as pd

# Load datasets
train_df = pd.read_csv('data/churn-bigml-80.csv')
test_df = pd.read_csv('data/churn-bigml-20.csv')

# Basic exploration
print(f"Training shape: {train_df.shape}")
print(f"Testing shape: {test_df.shape}")
print(f"Churn rate: {train_df['Churn'].mean():.2%}")
```

For complete preprocessing pipeline, see `src/preprocessing.py` or the main analysis notebook. 