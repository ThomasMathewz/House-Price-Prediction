---
# HOUSE PRICE PREDICTION
---
## 1. Introduction

This report presents a complete modelling pipeline developed for predicting residential house prices using structured tabular data.
The project begins with foundational preprocessing techniques, progresses through multiple machine learning models, introduces advanced feature engineering, and finally evaluates a modern deep learning method (TabNet).
The objective of this study is to develop a robust, accurate, and generalisable predictive model suitable for real-world applications in real estate analytics.

## 2. Dataset Overview

The dataset contains numerical and categorical attributes related to house characteristics, including:

* Structural details
* Construction materials
* Lot and land measurements
* Quality ratings
* Neighborhood categories
* Sale dates and sale price

The data includes missing values, mixed data types, and multicollinearity.
These characteristics make the dataset appropriate for a multi-stage modelling pipeline involving both classical machine learning algorithms and deep learning architectures.

## 3. Phase 1: Basic Preprocessing and Initial Models

The first phase focused on preparing the dataset using widely adopted preprocessing techniques:

__Preprocessing Steps__

* Median imputation for numerical features
* Mode imputation for categorical features
* Standard scaling for continuous variables
* One-hot encoding for categorical variables
* Construction of a ColumnTransformer to ensure pipeline consistency
* Train-test split for proper evaluation


__Models Evaluated__

* Linear Regression
* Ridge Regression
* Lasso Regression
* Random Forest Regression
* Gradient Boosting Regression
* XGBoost
* LightGBM
* CatBoost
* Stacking Ensemble

__Findings__

Tree-based boosting models (Gradient Boosting, XGBoost, LightGBM, CatBoost) substantially outperformed linear models, reflecting the dataset's nonlinear feature interactions.
The Stacking Ensemble performed competitively but did not consistently outperform the top boosting models.

## 4. Phase 2: Advanced Preprocessing and Feature Engineering

To improve predictive performance, advanced preprocessing and engineered features were introduced:

__Key Enhancements__

* Removal of highly correlated variables (to reduce multicollinearity)
* Engineering domain-informed features including:
  * Age of the house at time of sale
  * Recency of renovation
  * Living area per room
  * Total functional bathrooms
  * Interaction terms between key variables
* Improved handling of missing values based on domain relevance
* Consistent encoding and scaling across all phases

__Impact__

These enhancements improved data quality, reduced noise, and increased the ability of models to learn meaningful relationships.

## 5. Hyperparameter Tuning and Advanced Optimization

RandomizedSearchCV was used to optimise several high-performing models.
The following hyperparameters were tuned:

* Learning rate
* Number of estimators
* Maximum tree depth
* Regularisation strengths (L1, L2)
* Subsample ratios
* Column sample frequencies

__Outcome__

Hyperparameter tuning reduced overfitting and improved Cross-Validation RMSE across most boosting models.
Optimised versions of XGBoost, Gradient Boosting, LightGBM, and CatBoost performed significantly better than their default configurations.

## 6. Deep Learning Approach – TabNet

To examine the effect of deep learning on structured data, TabNet was incorporated.

__TabNet Pipeline__

* Preparing dense float32 arrays from preprocessed outputs
* Unsupervised pretraining using TabNetPretrainer
* Supervised fine-tuning using TabNetRegressor

__Findings__

Although TabNet was able to extract useful representations and perform competitively, it did not outperform the optimised boosting models.
This result aligns with current research showing that boosting algorithms still generally excel over deep learning in small-to-medium sized tabular datasets.

## 7. Model Evaluation and Comparison

__Model performance was compared using:__

* Cross-validation RMSE
* Holdout test RMSE
* R² scores
* Error distribution plots

__Best Performing Models__

* XGBoost
* Gradient Boosting
* CatBoost
* Stacking Ensemble
* LightGBM

__Summary__

XGBoost consistently achieved the best combination of cross-validation RMSE, holdout RMSE, and robustness across preprocessing variations.
Tree-based models clearly outperformed linear and deep learning models.

## 8. Final Model Recommendation

Based on experimental results, the recommended model for deployment is:

__XGBoost__

__Reasons for Recommendation__

* Lowest holdout RMSE among all models
* Strong generalisation ability
* Robust to nonlinear feature interactions
* High performance even with minimal preprocessing variation
* Fast inference and suitable for real-time use
* Consistent performance across multiple data splits and feature sets

The Stacking Ensemble showed competitive results but did not consistently surpass XGBoost, and it introduced additional complexity.
TabNet did not achieve comparable accuracy and therefore is not recommended for deployment.

## 9. Conclusion

This study successfully developed an end-to-end modelling pipeline for predicting house prices using structured tabular data.
It demonstrated the importance of:

* Rigorous preprocessing
* Domain-informed feature engineering
* Proper hyperparameter optimisation
* Comprehensive model comparison

Boosting algorithms proved superior to both linear models and deep learning approaches, with XGBoost emerging as the most accurate and dependable model.

The methodology and findings from this project provide a strong foundation for predictive modelling in real estate valuation, financial analytics, and related fields.
