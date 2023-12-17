import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.utils import to_categorical

def fetch_and_process_svm(train_path: str, test_path: str):
    # Loading the data
    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv('./test.csv')

    # Calculating frequency of patient_ids in bothe train and test data.
    train_frequency = train_data['patient_id'].value_counts().to_dict()
    test_frequency = test_data['patient_id'].value_counts().to_dict()
    frequency = {}

    for i in train_frequency:
        frequency[i] = 0
    for i in test_frequency:
        frequency[i] = 0

    for i in train_frequency:
        frequency[i] += train_frequency[i]
    for i in test_frequency:
        frequency[i] += test_frequency[i]

    # Adding a new column to the data for the calculated frequency.
    train_data['frequency_pid'] = train_data['patient_id'].map(frequency)
    test_data['frequency_pid'] = test_data['patient_id'].map(frequency)

    # Splitting data into training and validation sets
    X = train_data.drop('readmission_id', axis=1)
    y = train_data['readmission_id']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identifying categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    return X_train, y_train, X_val, y_val

def fetch_and_process_NN(train_path: str, test_path: str):
    # Loading the data
    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv('./test.csv')

    # Calculating frequency of patient_ids in bothe train and test data.
    train_frequency = train_data['patient_id'].value_counts().to_dict()
    test_frequency = test_data['patient_id'].value_counts().to_dict()
    frequency = {}

    for i in train_frequency:
        frequency[i] = 0
    for i in test_frequency:
        frequency[i] = 0

    for i in train_frequency:
        frequency[i] += train_frequency[i]
    for i in test_frequency:
        frequency[i] += test_frequency[i]

    # Adding a new column to the data for the calculated frequency.
    train_data['frequency_pid'] = train_data['patient_id'].map(frequency)
    test_data['frequency_pid'] = test_data['patient_id'].map(frequency)

    # Assuming 'patient_id' and 'enc_id' are not features for training
    features = train_data.drop(['patient_id', 'enc_id', 'readmission_id'], axis=1)
    labels = to_categorical(train_data['readmission_id'])  # One-hot encoding the labels

    # Handling missing values and encoding categorical variables
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = features.select_dtypes(include=['object']).columns

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

    # Splitting the training data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    test_data_processed = preprocessor.transform(test_data.drop(['patient_id', 'enc_id'], axis=1))

    return X_train, y_train, X_val, y_val, test_data_processed
