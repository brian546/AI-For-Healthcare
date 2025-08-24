# AI for Healthcare: Binary Classifier Model

## Overview
This project develops a binary classifier model to detect a specific health condition using medical data. The dataset includes 800 patient records with 70 medical indicators (features) and a binary target label. The data is split into:
- Training set: 600 samples
- Validation set: 100 samples (for hyperparameter tuning)
- Testing set: 100 samples (for final evaluation)

The model is trained exclusively on the training set, validated on the validation set, and evaluated on the testing set to ensure unbiased performance assessment.

## Features
- **Binary Classification**: Predicts the presence or absence of a health condition.
- **Interpretability**: Uses SHAP (SHapley Additive exPlanations) to analyze feature importance and their impact on predictions.
- **Visualization**: Includes a SHAP summary plot showing the distribution of SHAP values for the top 10 features, highlighting how each feature influences the model's output.

## Requirements
The notebook uses various Python libraries for data processing, model training, and visualization. Key dependencies include:
- `numpy`
- `pandas`
- `matplotlib`
- `shap` (for model interpretability)
- `sklearn` 

Install dependencies using:
```
pip install numpy pandas matplotlib shap scikit-learn
```

## Usage
1. **Run the Notebook**:
   - Open `code.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to load data, train the model, and generate visualizations.

2. **Key Outputs**:
   - SHAP Summary Plot: Displays the top 10 features' SHAP values on the validation set. Features are ranked by importance, with dots representing individual data points colored by feature value (red for high, blue for low). Positive SHAP values push predictions towards the positive class (e.g., presence of the condition).

## Analysis of Code Results
The primary output in the notebook is a SHAP summary plot (dot type) for the top 10 features based on validation data (`valid_X`). 

<img width="380" alt="SHAP" src="https://github.com/brian546/AI-For-Healthcare/blob/main/direction_shap_value_nn.png">

### Key Insights from SHAP Plot:
- **Feature Importance**: Features are ordered vertically by their mean absolute SHAP value (highest impact at the top). The top features have the strongest influence on model predictions.
- **SHAP Value Distribution**:
  - Horizontal spread indicates the range of impact: Positive values (right) increase the likelihood of the positive class; negative values (left) decrease it.
  - Color gradient: High feature values (red) often correlate with positive or negative impacts, revealing patterns (e.g., high values of a top feature may strongly predict the condition).
- **General Observations** (without named features):
  - The top feature shows a wide spread of SHAP values, suggesting it can both strongly support or oppose the prediction depending on its value.
  - Lower-ranked features have narrower distributions, indicating more consistent but weaker effects.
  - Clusters of red/blue dots highlight non-linear relationships: For some features, high values predominantly push towards one class.

This plot aids in model interpretability, helping identify key medical indicators for the health condition. For domain-specific insights, map anonymous features (e.g., `feature_0` to `feature_69`) to real medical terms.

## Project Structure

- `code.ipynb`: Jupyter notebook with model development, training, and SHAP analysis.
- (Optional) Data files: Assume datasets are loaded within the notebook (e.g., via CSV or built-in loaders).
