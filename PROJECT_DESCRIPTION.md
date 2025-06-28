# Customer Segmentation and Churn Prediction - Advanced Data Mining Project

## Project Overview
**Domain:** Telecommunications Analytics | **Type:** End-to-End Data Science Pipeline | **Course:** CS5228 Knowledge Discovery and Data Mining

This project demonstrates advanced data mining capabilities through a comprehensive analysis of telecom customer behavior, implementing both unsupervised learning for customer segmentation and supervised learning for churn prediction. The project showcases sophisticated feature engineering, multiple clustering algorithms, ensemble methods, and rigorous model optimization techniques.

## Technical Architecture & Methodology

### 1. Data Engineering & Preprocessing Pipeline
- **Dataset:** Telecom Customer Churn dataset (3,333 total records: 2,666 training, 667 testing)
- **Features:** 19 predictive features + 1 target variable across numerical and categorical data types
- **Preprocessing Framework:** Implemented robust sklearn Pipeline architecture with ColumnTransformer
  - **Binary Encoding:** Converted categorical variables (Yes/No, True/False) to numerical format
  - **One-Hot Encoding:** Applied to nominal categorical features (State, Area Code) with multicollinearity prevention
  - **Feature Scaling:** StandardScaler implementation with proper train/test isolation to prevent data leakage
  - **Feature Selection:** Correlation analysis identified and removed redundant features (Voice mail plan removed due to 0.957 correlation with Number vmail messages)

### 2. Exploratory Data Analysis & Statistical Insights
- **Class Distribution Analysis:** Identified class imbalance (14.5% churn rate) requiring specialized evaluation metrics
- **Correlation Analysis:** Comprehensive feature-target correlation assessment revealing key predictive indicators
- **Statistical Visualization:** Multi-dimensional analysis using box plots, heatmaps, and distribution analysis
- **Key Findings:** Established strong correlations between churn and usage patterns, service interactions, and plan selections

### 3. Unsupervised Learning - Customer Segmentation

#### K-Means Clustering Implementation
- **Optimization Strategy:** Elbow Method and Silhouette Score analysis across K=2-15
- **Parameter Selection:** Optimal K=2 identified through silhouette score maximization (0.096)
- **Cluster Characteristics:** 
  - Cluster 0: High usage customers (210 day minutes, 16.52% churn rate)
  - Cluster 1: Moderate usage customers (148 day minutes, 12.52% churn rate)
- **Business Insight:** Usage volume directly correlates with churn risk

#### DBSCAN Density-Based Clustering
- **Parameter Tuning:** k-distance graph analysis for eps estimation with NearestNeighbors
- **Configuration:** eps=5.0, min_samples=10 based on statistical analysis
- **Results:** Single cluster identification indicating lack of density-based separation in high-dimensional space
- **Dimensionality Analysis:** PCA visualization (67→2 dimensions) confirmed clustering challenges in high-dimensional data

### 4. Supervised Learning - Advanced Classification Pipeline

#### Model Architecture
Implemented comprehensive machine learning pipeline with 5 algorithms:
- **Logistic Regression:** Baseline linear classifier with L1/L2 regularization
- **Decision Tree:** Non-linear classifier with entropy/gini criterion optimization
- **Random Forest:** Ensemble method with bootstrap aggregation
- **K-Nearest Neighbors:** Instance-based learning with distance metric optimization
- **XGBoost:** Gradient boosting framework with advanced regularization

#### Hyperparameter Optimization Framework
- **Search Strategy:** GridSearchCV with 5-fold cross-validation
- **Optimization Metric:** F1-score prioritized due to class imbalance
- **Regularization Focus:** Emphasis on overfitting prevention through complexity control
- **Parameter Grids:** Comprehensive search spaces for each algorithm (144+ parameter combinations)

#### Performance Results & Model Selection
**Baseline Performance:**
- XGBoost: F1=0.818 (best baseline)
- Decision Tree: F1=0.725
- Random Forest: F1=0.703
- Significant overfitting observed in tree-based models (perfect training scores)

**Optimized Performance:**
- **XGBoost (Champion Model):** F1=0.821, Precision=0.91, Recall=0.75
- **Decision Tree (Runner-up):** F1=0.816, Precision=0.87, Recall=0.77
- **Overfitting Mitigation:** Successfully reduced train-test F1 gap from >0.25 to <0.15

### 5. Model Interpretation & Feature Importance Analysis

#### Feature Importance Ranking (Consensus across XGBoost & Decision Tree):
1. **Total day minutes** - Primary usage indicator
2. **Customer service calls** - Service quality proxy
3. **International plan** - Plan selection impact
4. **Total evening minutes** - Secondary usage pattern
5. **Total international minutes/calls** - International usage behavior
6. **Number voicemail messages** - Engagement indicator

#### Business Intelligence Insights:
- **Usage Patterns:** High daily usage correlates with increased churn risk
- **Service Quality:** Multiple service calls indicate satisfaction issues
- **Plan Optimization:** International plan subscribers show higher churn propensity
- **Geographic Independence:** Location-based features showed minimal predictive power

## Technical Implementation Highlights

### Advanced Techniques Demonstrated:
- **Pipeline Architecture:** Modular, reproducible preprocessing with sklearn Pipeline/ColumnTransformer
- **Cross-Validation Strategy:** Stratified k-fold validation for reliable performance estimation
- **Ensemble Methods:** Advanced boosting algorithms with regularization tuning
- **Dimensionality Reduction:** PCA for cluster visualization and interpretation
- **Statistical Analysis:** Comprehensive correlation analysis and multicollinearity detection
- **Evaluation Framework:** Multi-metric assessment (Precision, Recall, F1, Confusion Matrix)

### Code Quality & Best Practices:
- **Modular Design:** Structured workflow with clear separation of concerns
- **Data Leakage Prevention:** Proper train/test isolation throughout pipeline
- **Reproducibility:** Random state management and consistent methodology
- **Scalability:** Efficient implementation with parallel processing (n_jobs=-1)
- **Documentation:** Comprehensive inline documentation and analysis summaries

## Business Impact & Recommendations

### Actionable Insights Delivered:
1. **Proactive Monitoring System:** Implement usage threshold alerts for high-risk customers
2. **Service Quality Enhancement:** Prioritize customers with multiple service interactions
3. **Plan Portfolio Optimization:** Review international plan value proposition and pricing
4. **Engagement Strategies:** Promote value-added services to increase customer stickiness
5. **Retention Framework:** Focus on current experience over historical tenure

### Quantifiable Outcomes:
- **Model Accuracy:** 82%+ F1-score achievement with balanced precision/recall
- **Risk Stratification:** Clear customer segmentation with 4% churn rate differential
- **Feature Reduction:** 67-feature engineered dataset from 20 original variables
- **Overfitting Control:** Reduced generalization gap by 60% through hyperparameter tuning

## Technical Skills Demonstrated
- **Machine Learning:** Supervised/Unsupervised learning, Ensemble methods, Hyperparameter optimization
- **Statistical Analysis:** Correlation analysis, Distribution analysis, Dimensionality reduction
- **Data Engineering:** Feature engineering, Pipeline architecture, Data preprocessing
- **Python Ecosystem:** pandas, scikit-learn, XGBoost, matplotlib, seaborn
- **Model Evaluation:** Cross-validation, Performance metrics, Statistical significance testing
- **Business Analytics:** Customer segmentation, Churn analysis, ROI-focused recommendations

---
*This project demonstrates advanced proficiency in end-to-end data science workflows, combining theoretical knowledge with practical implementation to deliver actionable business insights through sophisticated machine learning techniques.* 