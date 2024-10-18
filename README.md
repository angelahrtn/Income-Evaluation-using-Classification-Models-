# Income-Evaluation
# Project Overview
This project, conducted as part of a final exam practice in the third-semester Machine Learning course, focuses on predicting income levels using classification techniques. Specifically, the project evaluates whether an individual's income exceeds $50K based on features such as age, education, and occupation. The analysis uses multiple classification models to compare performance and identify the most accurate model for income evaluation.

The dataset used in this project is a well-known income classification dataset. The goal is to develop a predictive model capable of accurately classifying whether an individual's income is below or above $50K. This project uses Python and key machine learning libraries for classification model development, tuning, and evaluation.

# Case Description
The core problem addressed in this project is binary classification: predicting whether an individual's income is above or below $50K per year, based on several personal and demographic factors. The features include:
  - Age
  - Education level
  - Occupation
  - Hours worked per week
  - Marital status
  - Capital gains and losses
  - And other relevant demographic details.
The task is to predict the income category, where:
  - <=50K: Low-income group.
  - >50K: High-income group.

# Objectives
- Build classification models to predict income based on provided features.
- Evaluate and compare multiple models to identify the most accurate and efficient model.
- Fine-tune models to improve performance and minimize errors.
- Use metrics such as accuracy, precision, recall, and F1-score to assess the models' predictive abilities.
- Provide a robust classification solution for the given dataset.

# Project Steps and Features
1. Data Collection & Preprocessing
- The dataset consists of several personal, demographic, and economic factors, loaded and preprocessed using Python.
- Missing data was handled, categorical variables were encoded using techniques like Label Encoding and Frequency Encoding, and feature scaling was applied where necessary.

2. Model Construction
- Seven classification models were implemented, including:
  - Logistic Regression
  - Decision Trees
  - Random Forest (Tuned)
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - Stacking Model: Combines the output of several models to improve predictions.
- Each model was trained and evaluated using key performance metrics such as accuracy, precision, recall, and F1-score.

3. Model Evaluation
- The models were evaluated using various metrics such as:
  - Accuracy: To measure overall correctness.
  - Precision, Recall, and F1-Score: To assess model performance on each class, especially important when dealing with imbalanced data.
  - Confusion Matrix: To visualize where the model predicted correctly or incorrectly.

4. Hyperparameter Tuning
- Grid search was performed to optimize model parameters, aiming to improve model performance by tuning key parameters such as:
  - The number of trees in Random Forest.
  - The regularization strength in Logistic Regression.
  - The learning rate and the number of estimators in Gradient Boosting.
  - Additional parameters were also tuned in other models.

5. Results Interpretation
- Random Forest and Stacking models achieved the highest accuracy on the testing set, with Random Forest providing better recall, particularly for the high-income group (>50K).
- Stacking models combined the predictions from several classifiers to further improve accuracy and recall.
- Precision and recall scores varied across models, but overall, Random Forest and Stacking offered the best performance.
 
# Tools Used
- Python: For data processing, model training, and evaluation.
- Libraries:
  - Pandas: For data manipulation and cleaning.
  - Scikit-learn: For building regression models and evaluation metrics.
  - Matplotlib: For data visualization and EDA.
  - Statistics: For performing statistical computations.

# Challenges
- Handling Imbalanced Data: The dataset had an imbalance between the high-income (>50K) and low-income (<=50K) classes. To address this, metrics like precision and recall were crucial to properly evaluating model performance.
- Model Tuning: Hyperparameter tuning required multiple iterations and computational resources to ensure the best model performance.

# Conclusion
This project successfully applied multiple classification models to predict income based on demographic and economic data. After tuning the models and evaluating their performance, the Random Forest and Stacking models achieved the best results, with an accuracy of 87%. These models provided useful insights into the factors contributing to higher-income predictions.

Further improvements could include exploring additional features, deeper hyperparameter tuning, and testing advanced models such as XGBoost. Nonetheless, this project demonstrates the utility of machine learning classification techniques for income prediction and provides a solid foundation for further work.
