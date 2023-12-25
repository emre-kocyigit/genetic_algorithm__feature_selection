import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# read csv and create a dataframe
df = pd.read_csv("undersampled_processed_data.csv")

# create X and y
y = df['label']
X = df.drop('label', axis=1)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def get_fitness_value(feature_list):
    # extract non_used features index as a list
    non_used_features = np.where(feature_list == 0)[0].tolist()
    # not to modify the original one, copy it as a new dataframe
    new_X_train = X_train.copy()
    new_X_test = X_test.copy()
    # drop the non-used columns by their indexes
    new_X_train = new_X_train.drop(new_X_train.columns[non_used_features], axis=1)
    new_X_test = new_X_test.drop(new_X_test.columns[non_used_features], axis=1)

    # build and train a ML classifier
    model = LogisticRegression(solver='liblinear', max_iter=100)  # configure the hyper-parameter to increase the speed
    model.fit(new_X_train, y_train)

    # make predictions with test data and calculate classification metrics
    y_pred = model.predict(new_X_test)

    return f1_score(y_test, y_pred)


