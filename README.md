# Dataset Description
Kaggle Competition Link: [Canadian Hospital Re-admittance Challenge | Kaggle](https://www.kaggle.com/competitions/canadian-hospital-re-admittance-challenge/overview)

We were provided with a dataset containing records of patients who were admitted to hospitals in Canada. It presents us with demographic as well as diagnostic information for each record corresponding to a visit (encounter).

The aim of the challenge was to predict whether a patient would be *readmitted* within 30 days, after 30 days, or not at all.

*Readmission of a patient generally indicates incorrect diagnosis or prescription, resulting in the risk of side effects and wastage of resources for both the hospital and the patient. Predicting readmission can prevent this.*
### Description of Columns

- enc_id: Unique identifier of an encounter
- patient_id: Unique identifier of a patient
- race: Race of the person
- gender: Gender of the person
- age: Age of the person grouped in 10-year intervals
- weight: Weight in pounds.
- admission_type_id: Integer identifier corresponding to 9 distinct values.
- discharge_disposition_id: Integer identifier corresponding to 29 distinct values.
- admission_source_id: Integer identifier corresponding to 21 distinct values.
- time_in_hospital: Integer number of days between admission and discharge.
- payer_code: Integer identifier corresponding to 23 distinct values.
- medical_specialty: Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct values.
- num_lab_procedures: Number of lab tests performed during the encounter.
- num_procedures: Number of procedures (other than lab tests) performed during the encounter.
- num_medications: Number of distinct generic names administered during the encounter.
- number_outpatient: Number of outpatient visits of the patient in the year preceding the encounter.
- number_emergency: Number of emergency visits of the patient in the year preceding the encounter.
- number_inpatient: Number of inpatient visits of the patient in the year preceding the encounter.
- diag_1: The primary diagnosis (coded as first three digits of ICD9); 848 distinct values.
- diag_2: Secondary diagnosis (coded as first three digits of ICD9); 923 distinct values.
- diag_3: Additional secondary diagnosis (coded as first three digits of ICD9); 954 distinct values.
- number_diagnoses: Number of diagnoses entered to the system.
- max_glu_serum: Indicates the range of the result or if the test was not taken
- A1Cresult: Indicates the range of the result or if the test was not taken.
- *Columns corresponding to drug dosage:* describe if there was any change in a given drug's dosage for this encounter. [---TODO---]
- change: Indicates if there was a change in diabetic medications (either dosage or generic name). 
- diabetesMed: Indicates if there was any diabetic medication prescribed.
- readmission_id: Days to inpatient readmission (label).

## EDA
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

- We experimented by imputing with mode, the validation accuracy decreased slightly, prompting us to rethink our strategy.
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
The given dataset inherently contains categorical columns which can be grouped further. Examples of such columns include discharge
### Label / OHE / Avg for age

## New Columns Introduced
### Drug change counting
### Grouping Numerical values for inpatient / outpatient / emergency

### Frequency for patient_id

# Model Selection and Training

- KNN
- DT
- RF
- XGBoost
- LGBClassifier
- CatBoost
- Logistic Regression
- Give detailed descriptions for 2 model and 3 model approach

## Hyper parameter Tuning


## Validation


## Final Results
