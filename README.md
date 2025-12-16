# ML---practise-projects-

# 💻 Laptop Price Prediction – Model Improvement & Regularization

This project focuses on improving a laptop price prediction model using **cross-validation**, **overfitting analysis**, **polynomial regression**, **ridge regression**, and **grid search hyperparameter tuning**.  
The dataset contains various laptop specifications, and the target variable is **Price**.

---

## 📊 Dataset Overview

The dataset includes the following attributes:

- CPU_frequency  
- RAM_GB  
- Storage_GB_SSD  
- CPU_core  
- OS  
- GPU  
- Category  
- **Price** (Target Variable)

---

## 🎯 Project Objectives

1. Improve model performance using **cross-validation**
2. Identify and analyze **overfitting**
3. Apply **Ridge Regression** to reduce model variance
4. Optimize model performance using **GridSearchCV**

---

## 🧩 Task Breakdown

---

### ✅ Task 1: Cross-Validation for Model Improvement

- The dataset is divided into:
  - **x_data** → All independent features
  - **y_data** → Target variable (`Price`)
- Cross-validation is applied to improve model robustness and reduce bias caused by a single train-test split.

**Outcome:**  
More stable and reliable performance evaluation across multiple folds.

---

### ⚠️ Task 2: Overfitting Analysis using Polynomial Regression

- The dataset is split into:
  - **50% Training Data**
  - **50% Testing Data**
- A polynomial regression model is built using only the feature:
  - `CPU_frequency`
- Polynomial degrees from **1 to 5** are evaluated.
- The **R² score** is calculated for each degree to identify overfitting.

**Key Insight:**  
- Lower-degree polynomials may underfit  
- Higher-degree polynomials may overfit  
- R² scores help identify the optimal complexity level

📌 The R² scores for degrees 1–5 are stored in a list for comparison.

---

### 🔒 Task 3: Ridge Regression with Polynomial Features

- Multiple features are used:
  - `CPU_frequency`
  - `RAM_GB`
  - `Storage_GB_SSD`
  - `CPU_core`
  - `OS`
  - `GPU`
  - `Category`
- Polynomial features of **degree = 2** are generated.
- The dataset is split into training and testing sets.
- **Ridge Regression** is applied to control overfitting by penalizing large coefficients.

**Outcome:**  
Improved generalization performance on unseen data.

---

### 🔍 Task 4: Hyperparameter Tuning using Grid Search

- **GridSearchCV** is used to identify the optimal value of **alpha** for Ridge Regression.
- The same set of features from Task 3 is used.
- Multiple alpha values are tested to find the best regularization strength.

**Outcome:**  
Best-performing model configuration with optimized bias–variance tradeoff.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (for analysis and visualization)

---

## 📈 Key Learnings

- Cross-validation improves model reliability
- Polynomial models can easily overfit without regularization
- Ridge Regression helps control variance in complex models
- Grid search ensures optimal hyperparameter selection

--

## 🏠 House Price Prediction using Support Vector Regression (SVR)

### 📌 Overview

This is a **practice machine learning project** aimed at predicting house prices using a **Support Vector Regression (SVR)** model.  
The project uses the **California Housing dataset** from `sklearn.datasets` and explores how factors such as house size, number of rooms, location, median income, and population influence housing prices.

---

### 📊 Dataset

The dataset is sourced from **sklearn.datasets (California Housing)** and contains the following features:

- **MedInc** – Median income of residents in the area  
- **HouseAge** – Average age of houses in the area  
- **AveRooms** – Average number of rooms per household  
- **AveBedrms** – Average number of bedrooms per household  
- **Population** – Population of the area  
- **AveOccup** – Average number of occupants per household  
- **Longitude** – Geographic longitude  
- **Latitude** – Geographic latitude  

🎯 **Target Variable:**  
- **House Price** (Median house value)

---

### 🤖 Model Used

- **Support Vector Regression (SVR)**
- Feature scaling applied to ensure optimal performance
- Hyperparameter tuning to improve prediction accuracy

---

### ⚙️ Project Workflow

1. Load the California Housing dataset  
2. Perform data preprocessing  
   - Feature scaling  
   - Train-test split  
3. Train the SVR model  
4. Evaluate model performance using:
   - **Mean Squared Error (MSE)**
   - **Mean Absolute Error (MAE)**

---

### 📈 Evaluation Metrics

- **Mean Squared Error (MSE):** Measures average squared difference between predicted and actual prices  
- **Mean Absolute Error (MAE):** Measures average absolute difference between predicted and actual prices  

Lower values indicate better model performance.

---

### 🛠️ Tools & Libraries

- Python  
- NumPy  
- Pandas  
- Scikit-learn  

---

### 🚀 Future Enhancements

- Compare SVR with Linear Regression and Tree-based models  
- Perform GridSearchCV for advanced hyperparameter tuning  
- Add data visualization for feature impact analysis  
- Deploy the model using a web interface or API

---
