
---
# Heart Disease Final Project Report
---
- Introduction
- Data Preperation
- Data Analysis and Visualization
- Data preprocessing
- Data regularization & Model Selection
- Hyperparameter tuning
- Model Evaluation
- Challenges and closure
---


# 1. Introduction
Heart disease is a significant worldwide health concern and one of the top causes of death.  Early detection may greatly improve treatment results.

The heart disease dataset contains data for 1000 patients where information such as cholesterol, chest pain type, Fasting BS etc.. all play a role and are factors that can lead to Cardiovascular diseases (CVDs).

This is a report for our project that aims to utilize machine learning models and algorithms to develop a predictive model that can accuractely determine whether a person has heart disease or not based on early indicators.

---
# 2. Data Preperation
To start off a machine learning project we firstly need to import some libraries we will keep using throughout the code.

- ***Pandas*** and ***Numpy*** are libraries for manipulating and filtering the data for better processing.
- ***Matplotlib*** is for visualization of the data and ***Seaborn*** is a library based on Matplotlib for visual appeal.
- ***Scikit-learn*** provides us with the models and metrics for developing our predictive model.

After importing the libraries we are going to use, we went ahead and checked for any null values in the dataset that can interfere with our predictions.

We then followed this up with dividing the dataset into **X and Y  variables** where **X** considers all information on the patient except for the patient_id, and **Y** is the target variable for heart disease.

The data is now ready for visualization and analysis.
---

---
# 3. Data Analysis and Visualization
>**Class Imbalance and Visualization**

To visualize the dataset, we calculated the percentage of absence and presence of heart disease in the data set and found that there was an imbalance in the class. We found that 54% had presence of heart disease and 46% had absence.

To prevent bias, we use a technique called Oversampling using RandomOverSampler, which allows us to represent the classes equally for better predictions.

We then re-visualized the classes and it was evident that the classes are now equal with 50% for each class.

To find the best features to use for our model, we look for the features that have strong correlation between each other. We plot a correlation matrix to display the correlations.
---

---
# 4. Data Preprocessing
>**Data splitting and Normalization**

In order to start building the model we need to divide the dataset into two subsets; **training and testing**.

We use train_test_split on the following variables: X_train, X_test, y_train, y_test. This ensures the proportion of heart disease is preserved.

To improve the efficiency of algorithms used in our model (ex . PCA), we standardize using **StandardScaler** to ensure that the features have a mean of 0 and standard deviation of 1.

>**Feature Selection Using PCA**

Principal Component Analysis (PCA) is used to reduce the features that we have in the dataset. After applying PCA, we visualize from the graph that 9 features are required to reach the 95% predictive power. The reduction of features help eliminate unwanted features that do not help in prediction.

We found that the top 3 features with the highest information were *ST Slope, cholesterol and fasting blood sugar.*

---

---
# 5. Data Regularization & Model Selection

To look for the best model, we decided to test multiple models. These include:
- Logistic Regression (L1 and L2 with and without PCA)
- Random Forest Classifier (with and without PCA)
- Supper Vector Machine (SVM) (with and without PCA)


---


---
# 6. Hyperparameter tuning

To maximize the performance of our best models, we implemented **Hyperparameter tuning** to find the most optimal parameters for the best results. We applied **GridSearchCV** to SVM and RandomForestClassifier to improve the models.

After tuning, we found that the results were:
- Random Forest w PCA with 90%
- SVM with 88%

It is evident that Random Forest with PCA performs better than SVM.
---

---
# 7. Model Evalution

Knowing that Random Forest with PCA was our best model, we evaluated it with a Confusion Matrix and a precision-recall curve.

**Confusion Matrix:**
True Positives --> 99
, True Negatives --> 94
, False Positives --> 14
, False Negatives --> 9

**Precision-Recall:**
The graph suggests that the model works best with moderate threshold. At lower thresholds, recall is 100% and precision is at 50% means 50% of disease predictions are wrong.

Precision --> 88%
Recall --> 92%
F1 Score --> 90%
Accuracy --> 89%

---

---
# 8. Challenges and Conclusion

After checking for class imbalance, null values and oversampled the data set for better testing, we regularized the data and worked with multiple models such as Logistic Regression, Random Forest and SVM.

We incorporated PCA and Hyperparameter tuning to check if these can increase the model's accuracy.

**Findings:**

- PCA revealed ST_Slope , Cholesterol and Fasting Bs as dominant predictors
- Random Forest performed better with PCA achieving 90% accuracy
- SVM performed better without PCA

**Future improvements:**

- Trying different models
- Improve Hyperparameter tuning (maybe using random search)

---
