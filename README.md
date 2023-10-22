# Dataset Description
Kaggle Competition Link: [Canadian Hospital Re-admittance Challenge | Kaggle](https://www.kaggle.com/competitions/canadian-hospital-re-admittance-challenge/overview)

We were provided with a dataset containing records of patients who were admitted to hospitals in Canada. It presents us with demographic as well as diagnostic information for each record corresponding to a visit (encounter).

The aim of the challenge was to predict whether a patient would be *readmitted* within 30 days, after 30 days, or not at all.

*Readmission of a patient generally indicates incorrect diagnosis or prescription, resulting in the risk of side effects and wastage of resources for both the hospital and the patient. Predicting readmission can prevent this.*
### Description of Columns

| Column Name | Description |
| -------- | --------------------------------- |
| enc_id | Unique identifier of an encounter |
| patient_id | Unique identifier of a patient |
| race | Race of the person |
| gender | Gender of the person |
| age | Age of the person grouped in 10-year intervals |
|weight | Weight in pounds |
| admission_type_id | Integer identifier corresponding to 9 distinct values |
| discharge_disposition_id | Integer identifier corresponding to 29 distinct values |
| admission_source_id | Integer identifier corresponding to 21 distinct values |
|time_in_hospital | Integer number of days between admission and discharge |
| payer_code | Integer identifier corresponding to 23 distinct values |
| medical_specialty | Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values |
| num_lab_procedures | Number of lab tests performed during the encounter |
| num_procedures | Number of procedures (other than lab tests) performed during the encounter |
| num_medications | Number of distinct generic names administered during the encounter |
| number_outpatient | Number of outpatient visits of the patient in the year preceding the encounter |
| number_emergency | Number of emergency visits of the patient in the year preceding the encounter |
| number_inpatient | Number of inpatient visits of the patient in the year preceding the encounter |
| diag_1 | The primary diagnosis (coded as first three digits of ICD9) |
| diag_2 | Secondary diagnosis (coded as first three digits of ICD9) |
| diag_3 | Additional secondary diagnosis (coded as first three digits of ICD9) |
| number_diagnoses | Number of diagnoses entered to the system |
| max_glu_serum | Indicates the range of the result or if the test was not taken |
| A1Cresult | Indicates the range of the result or if the test was not taken |
| *Columns corresponding to drug dosage* | describe if there was any change in a given drug's dosage for this encounter. `["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]` |
| change | Indicates if there was a change in diabetic medications (either dosage or generic name) | 
| diabetesMed | Indicates if there was any diabetic medication prescribed |
| readmission_id | Days to inpatient readmission (label) |

## Exploratory Data Analysis
### Tableau Plots, Matplotlib / Seaborn


# Data Processing

### Dropping Columns
| Column Dropped | Reason |
| ----------------- | -------- |
| payer_code | 39.55% null values, does not affect readmission of a patient |
| weight | 96.84% null values |
| max_glu_serum | 94.77% null values |
| A1Cresult | 83.32% null values |
| patient_id | Replaced with a new column to reflect the frequency of patient_id in data |
| Columns corresponding to drug dosage | Dropped after introducing 4 new columns to reflect the count of *Up*, *Down*, *No*, and *Steady* for each encounter_id |
| diag_1, diag_2, diag_3 | Dropped after experimenting with various sub-groupings of these columns and dropping the rest. No significant improvement was observed in validation score. |
### Dropping null rows
We initially considered dropping rows corresponding to null values in *race*, *diag_1*, *diag_2*, and *diag_3* as they had *2.27*%, *0.02*%, *0.34*%, and *1.38*% null values respectively. We observed that this did not improve our validation score, when compared with imputing values.
### Imputing
For all categorical columns, null values were imputed with a new category, denoted by *"0"*.
For all numerical columns, none of them were observed to have null values.

Prior to this, we experimented by imputing with mode, the validation accuracy decreased slightly, prompting us to rethink our strategy.
### Outlier Detection
We tried outlier detection on the numerical columns, ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses']

It was observed that removing these outliers improved the validation score but did not generalize well to unseen data when submitted to Kaggle. We believe this is due to the fact that number of rows in the validation set decreased, leading to a higher validation score.

The definition for upper and lower whiskers is as follows:
```python
Q1 = df_copy[attr].quantile(0.25)   # 1st quartile
Q3 = df_copy[attr].quantile(0.75)   # 3rd quartile
IQR = Q3 - Q1                       # Inter-quartile range
lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR
```
### Grouping / Not Grouping


### Label / OHE / Avg for age

## New Columns Introduced
### Drug change counting
### ! Grouping Numerical values for inpatient / outpatient / emergency

### Frequency for patient_id

# Model Selection and Training

### Logistic Regression, KNN Classifier, and Decision Tree Classifier
In our initial attempt to train a model, we made use of *logistic regression, KNN classifier, and Decision Tree Classifier*. We observed that logistic regression and KNN classifier returned poor results and the decision tree classifier returned considerably better results. After tweaking the hyper parameters for a while, the team decided to switch to an ensemble method, *Random Forest Classifier*.

### Random Forest Classifier

### XGBoost Classifier
### LGBMClassifier - Final Model Used
### CatBoost Classifier
### 2-Model and 3-Model Approaches
Having tried various single model setups , we tried 2 different strategies involving multiple models to improve the prediction accuracy.

The first was creating a 2-model setup. The first model was created to predict whether the `readmission_id` was zero or not. The reason for this choice was that the percentage of records where readmission id was zero is around 10% which is very less. Once a record is classified to have a non-zero `readmission_id`, it is sent to another model which was trained to choose between `readmission_id` 1 or 2. This model was trained only on data which has `readmission_id` 1 or 2. The 2 model setup was an interesting strategy for the problem but the 1 / 2 classifier struggled to classify records correctly, reducing the accuracy of the whole setup. This was the reason why this strategy was dropped.

The 3-model setup trains 3 different models to predict whether a data point has a particular `readmission_id` or not. This way the 3 models predict a particular class with a certain probability. Once the models predict the class with a probability, the `readmission_id` is considered to be the class with maximum probability. This is also a good setup but this idea failed due to the poor accuracy of the classifier for `readmission_id` 1. The class 1 classifier could not find the necessary features to predict class 1 correctly leading to records with `readmission_id` as class 1 to be classified as class 2. This decreased the accuracy of the overall setup *(0.708 on Kaggle)*. Hence we decided to drop both the strategies.
## Hyper parameter Tuning
### Grid Search and Cross Validation
In an attempt to discover the best hyperparameters for some of the above models, we used the `RandomCV` and `GridSearchCV` cross validation methods. The returned hyper parameter values did improve the validation score slightly, but after carefully considering the tradeoff in training time, we decided to use only select hyper parameter values.

**RandomCV**
```python
from sklearn.model_selection import RandomizedSearchCV
# Create the random grid
random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                'max_features': ['log2', 'sqrt', 'none'],
                'max_depth': [int(x) for x in range(1, 13)],
                'min_samples_split': [x for x in range(2, 100, 5)],
                'min_samples_leaf': [x for x in range(3, 15)],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'oob_score': [True, False],
                'class_weight': ['balanced', 'balanced_subsample']}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1, scoring='accuracy')
# Fit the random search model
rf_random.fit(X_train, Y_train)
print(rf_random.best_params_)
```

**GridSearchCV**
```python
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv=3, n_jobs=-1, verbose = 2)

# # Fit the grid search to the data
# grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
best_grid = grid_search.best_estimator_
```
## Validation
In addition to using Grid Search cross validation as described above, we used `accuracy_score` and `f1_score` metrics to gauge the performance of our model.

We also made use of `confusion_matrix` to understand which classes were being wrongly classified, and made strides to improve predictions based on these insights.

Lastly, for some of the models described earlier, we also made use of the `predict_proba` method to interpret the predicted probabilities of each class, for each test point. 
## Final Results
