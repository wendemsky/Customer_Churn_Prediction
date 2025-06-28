# 📊 Customer Segmentation & Churn Prediction
### Advanced Data Mining with Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **🎯 Objective:** Comprehensive analysis of telecom customer behavior using advanced data mining techniques to identify customer segments and predict churn with 82%+ F1-score accuracy.

## 🚀 Project Highlights

- **🏆 Champion Model:** XGBoost achieving **F1-Score: 0.821** with balanced precision (0.91) and recall (0.75)
- **📈 Performance Optimization:** Reduced overfitting by 60% through systematic hyperparameter tuning
- **🎯 Customer Segmentation:** K-Means clustering revealed 4% churn rate differential between usage-based segments
- **💡 Business Impact:** Delivered 5 actionable retention strategies based on feature importance analysis

## 📋 Table of Contents
- [🔍 Project Overview](#-project-overview)
- [🛠️ Technical Architecture](#️-technical-architecture)
- [📊 Results & Performance](#-results--performance)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🔧 Dependencies](#-dependencies)
- [📈 Key Insights](#-key-insights)
- [🤝 Contributing](#-contributing)

## 🔍 Project Overview

This project demonstrates advanced data science capabilities through a comprehensive analysis of telecom customer churn, implementing both **unsupervised learning** for customer segmentation and **supervised learning** for churn prediction.

### 🎯 Key Objectives
1. **Customer Segmentation:** Identify distinct customer groups using K-Means and DBSCAN clustering
2. **Churn Prediction:** Build high-performance classification models with ensemble methods
3. **Business Intelligence:** Extract actionable insights for customer retention strategies

### 📊 Dataset Information
- **Source:** Telecom Customer Churn Dataset (BigML)
- **Size:** 3,333 total records (2,666 training, 667 testing)
- **Features:** 19 predictive features + 1 target variable
- **Challenge:** Class imbalance (14.5% churn rate)

## 🛠️ Technical Architecture

### 🔧 Data Engineering Pipeline
```python
# Robust preprocessing with sklearn Pipeline
├── Binary Encoding (Yes/No → 1/0)
├── One-Hot Encoding (Categorical features)
├── Feature Scaling (StandardScaler)
└── Feature Selection (Correlation analysis)
```

### 🤖 Machine Learning Models
| Algorithm | Type | Best F1-Score | Key Strengths |
|-----------|------|---------------|---------------|
| **XGBoost** | Gradient Boosting | **0.821** | Ensemble power, regularization |
| **Decision Tree** | Tree-based | **0.816** | Interpretability, feature importance |
| **Random Forest** | Ensemble | 0.653 | Robust to overfitting |
| **Logistic Regression** | Linear | 0.340 | Baseline, fast training |
| **K-NN** | Instance-based | 0.381 | Non-parametric approach |

### 🎯 Optimization Strategy
- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
- **Evaluation Metric:** F1-score (optimal for imbalanced classes)
- **Overfitting Control:** Regularization parameters and complexity constraints
- **Feature Engineering:** 67 engineered features from 20 original variables

## 📊 Results & Performance

### 🏆 Model Performance Comparison
![Model Performance](https://via.placeholder.com/600x300/4CAF50/FFFFFF?text=Model+Performance+Chart)

### 📈 Key Metrics (Champion Model - XGBoost)
```
🎯 F1-Score:     0.821
🎯 Precision:    0.91
🎯 Recall:       0.75
🎯 Accuracy:     91.2%
📉 Overfitting:  Reduced by 60%
```

### 🔍 Feature Importance Rankings
1. **Total day minutes** - Primary usage indicator
2. **Customer service calls** - Service quality proxy  
3. **International plan** - Plan selection impact
4. **Total evening minutes** - Secondary usage pattern
5. **International usage** - Global communication needs
6. **Voicemail engagement** - Service utilization

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### 🎮 Run the Analysis
```bash
# Open the main notebook
jupyter notebook v2.ipynb

# Or run the streamlined version
jupyter notebook main.ipynb
```

## 📁 Project Structure
```
📦 customer-churn-prediction/
├── 📊 v2.ipynb                    # Main analysis notebook
├── 📊 main.ipynb                  # Streamlined version
├── 📄 README.md                   # Project documentation
├── 📄 PROJECT_DESCRIPTION.md      # Detailed technical description
├── 📄 requirements.txt            # Python dependencies
├── 📄 LICENSE                     # MIT License
├── 📂 data/
│   ├── 📊 churn-bigml-80.csv     # Training dataset
│   └── 📊 churn-bigml-20.csv     # Testing dataset
├── 📂 results/
│   ├── 📈 model_performance.png   # Performance visualizations
│   ├── 📊 feature_importance.png  # Feature analysis charts
│   └── 📋 model_comparison.csv    # Detailed results
└── 📂 src/
    ├── 🔧 preprocessing.py        # Data preprocessing utilities
    ├── 🤖 models.py               # Model implementations
    └── 📊 visualization.py        # Plotting functions
```

## 🔧 Dependencies

### Core Libraries
```python
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.0.0    # Machine learning
xgboost>=1.5.0         # Gradient boosting
matplotlib>=3.5.0      # Plotting
seaborn>=0.11.0        # Statistical visualization
jupyter>=1.0.0         # Notebook environment
```

## 📈 Key Insights & Business Impact

### 🎯 Customer Segmentation Findings
- **High Usage Cluster:** 210 daily minutes, 16.52% churn rate
- **Moderate Usage Cluster:** 148 daily minutes, 12.52% churn rate
- **Insight:** Usage volume directly correlates with churn risk

### 💡 Actionable Recommendations
1. **🚨 Proactive Monitoring:** Alert system for high-usage customers
2. **🛠️ Service Quality:** Priority support for multiple service calls
3. **📱 Plan Optimization:** Review international plan value proposition
4. **💬 Engagement:** Promote voicemail and value-added services
5. **🎯 Retention Focus:** Current experience over historical tenure

### 📊 Business Value
- **Risk Stratification:** Clear customer segmentation with measurable churn differentials
- **Cost Optimization:** Targeted retention efforts based on predictive modeling
- **Revenue Protection:** Early identification of at-risk high-value customers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🐛 Issues & Suggestions
- Found a bug? [Open an issue](https://github.com/yourusername/customer-churn-prediction/issues)
- Have an idea? [Start a discussion](https://github.com/yourusername/customer-churn-prediction/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Course:** CS5228 Knowledge Discovery and Data Mining
- **Dataset:** BigML Telecom Customer Churn Dataset
- **Institution:** National University of Singapore

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/customer-churn-prediction.svg?style=social&label=Star)](https://github.com/yourusername/customer-churn-prediction)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/customer-churn-prediction.svg?style=social&label=Fork)](https://github.com/yourusername/customer-churn-prediction/fork)

</div> 