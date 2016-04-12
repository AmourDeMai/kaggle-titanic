#!/usr/bin/python

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor


def fill_missing_ages(data):
    """Fill missing age using RandomForestClassifier"""

    # Selete features for training
    age_data = data[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']]

    # Divide passengers into with and without age
    with_age = age_data.loc[age_data['Age'].notnull()]
    without_age = age_data.loc[age_data['Age'].isnull()]
    # X is the features
    X = with_age.iloc[:, 1:]
    Y = with_age.iloc[:, 0]

    # Train RandomForestRegressor
    alg = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
    alg.fit(X, Y)

    # Predict age fill age
    data.loc[data['Age'].isnull(), 'Age'] = alg.predict(without_age.iloc[:, 1:])

    return data, alg


def set_cabin_type(data):
    """Set cabin into 2 types: with or without value """

    data.loc[data['Cabin'].notnull(), 'Cabin'] = 'Yes'
    data.loc[data['Cabin'].isnull(), 'Cabin'] = 'No'

    return data


def extract_features(data):
    """Features extraction"""

    # Set cabin type
    data = set_cabin_type(data)
    # Fill Embarked missing data with S
    data["Embarked"] = data["Embarked"].fillna("S")
    # Fill Fare missing data with the median
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    # Fill missing value of ages
    data, alg = fill_missing_ages(data)

    # Extract dummies of Pclass, Sex, Cabin, Embarked
    pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
    sex = pd.get_dummies(data['Sex'], prefix='Sex')

    cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
    embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')

    # Scale Age
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(train_data['Age'])
    age = scaler.transform(data['Age'], age_scale_param)
    age = pd.DataFrame({'Age': age})
    # Scale Fare
    fare_scale_param = scaler.fit(train_data['Fare'])
    fare = scaler.transform(data['Fare'], fare_scale_param)
    fare = pd.DataFrame({'Fare': fare})

    # Concate features
    features = pd.concat([pclass, sex, cabin, embarked, age, fare], axis=1)

    return features


def train_model(data):
    """Train a model"""

    # Extract train features
    features = extract_features(data)

    # Initialize algorithm class
    alg = LogisticRegression()
    # Set random_state to ensure to get the same splits every time we run this
    kf = KFold(features.shape[0], n_folds=3, shuffle=False, random_state=1)

    predictions = []
    for train, test in kf:
        alg.fit(features.iloc[train, :], train_data['Survived'].iloc[train])
        prediction = alg.predict(features.iloc[test, :])
        predictions.append(prediction)

    return predictions, alg


def evaluation(data, classifier):
    """Evaluate test data"""

    # Extract test features
    features = extract_features(data)
    # Predict test data
    prediction = classifier.predict(features)

    return prediction

# main
if __name__ == '__main__':

    # Load train data
    train_data = pd.read_csv('data/train.csv')
    # CV result, save trained classifier alg
    predictions, alg = train_model(train_data)
    predictions = np.concatenate(predictions, axis=0)
    # Calculate CV accuracy
    error_nb = sum(abs(predictions - pd.Series.as_matrix(train_data.Survived)))
    accuracy = 1 - error_nb / train_data.shape[0]

    # Load test data
    test_data = pd.read_csv('data/test.csv')
    predictions_test = evaluation(test_data, alg)

    print("accuracy for the training data is:", accuracy)
