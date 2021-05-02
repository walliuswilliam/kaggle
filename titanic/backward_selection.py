import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import sys
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def convert_regressor_output_to_survival_value(output):
  if output < 0.5:
    return 0
  else:
    return 1

def get_accuracy(predictions, actual):
  num_correct = 0
  num_incorrect = 0
  for i in range(len(predictions)):
    if predictions[i] == actual[i]:
      num_correct += 1
    else:
      num_incorrect += 1
  return num_correct / (num_correct + num_incorrect)

def fit_regressor(df, iters=100):
  df_train = df[:500]
  df_test = df[500:]

  arr_train = np.array(df_train)
  arr_test = np.array(df_test)

  y_train = arr_train[:,0]
  y_test = arr_test[:,0]

  x_train = arr_train[:,1:]
  x_test = arr_test[:,1:]

  regressor = LogisticRegression(max_iter=iters, random_state=0)
  regressor.fit(x_train, y_train)

  coef_dict = {}
  feature_columns = df_train.columns[1:]
  feature_coefficients = regressor.coef_

  for i in range(len(feature_columns)):
    column = feature_columns[i]
    coefficient = feature_coefficients[0][i]
    coef_dict[column] = coefficient

  y_test_predictions = regressor.predict(x_test)
  y_train_predictions = regressor.predict(x_train)

  y_test_predictions = [convert_regressor_output_to_survival_value(output) for output in y_test_predictions]
  y_train_predictions = [convert_regressor_output_to_survival_value(output) for output in y_train_predictions]

  return get_accuracy(y_test_predictions, y_test)


df = pd.read_csv('titanic/csv/processed_data.csv')

features = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T', 'Sex * Pclass', 'Sex * Fare', 'Sex * Age', 'Sex * SibSp', 'Sex * SibSp>0', 'Sex * Parch>0', 'Sex * Embarked=C', 'Sex * Embarked=None', 'Sex * Embarked=Q', 'Sex * Embarked=S', 'Sex * CabinType=A', 'Sex * CabinType=B', 'Sex * CabinType=C', 'Sex * CabinType=D', 'Sex * CabinType=E', 'Sex * CabinType=F', 'Sex * CabinType=G', 'Sex * CabinType=None', 'Sex * CabinType=T', 'Pclass * Fare', 'Pclass * Age', 'Pclass * SibSp', 'Pclass * SibSp>0', 'Pclass * Parch>0', 'Pclass * Embarked=C', 'Pclass * Embarked=None', 'Pclass * Embarked=Q', 'Pclass * Embarked=S', 'Pclass * CabinType=A', 'Pclass * CabinType=B', 'Pclass * CabinType=C', 'Pclass * CabinType=D', 'Pclass * CabinType=E', 'Pclass * CabinType=F', 'Pclass * CabinType=G', 'Pclass * CabinType=None', 'Pclass * CabinType=T', 'Fare * Age', 'Fare * SibSp', 'Fare * SibSp>0', 'Fare * Parch>0', 'Fare * Embarked=C', 'Fare * Embarked=None', 'Fare * Embarked=Q', 'Fare * Embarked=S', 'Fare * CabinType=A', 'Fare * CabinType=B', 'Fare * CabinType=C', 'Fare * CabinType=D', 'Fare * CabinType=E', 'Fare * CabinType=F', 'Fare * CabinType=G', 'Fare * CabinType=None', 'Fare * CabinType=T', 'Age * SibSp', 'Age * SibSp>0', 'Age * Parch>0', 'Age * Embarked=C', 'Age * Embarked=None', 'Age * Embarked=Q', 'Age * Embarked=S', 'Age * CabinType=A', 'Age * CabinType=B', 'Age * CabinType=C', 'Age * CabinType=D', 'Age * CabinType=E', 'Age * CabinType=F', 'Age * CabinType=G', 'Age * CabinType=None', 'Age * CabinType=T', 'SibSp * Parch>0', 'SibSp * Embarked=C', 'SibSp * Embarked=None', 'SibSp * Embarked=Q', 'SibSp * Embarked=S', 'SibSp * CabinType=A', 'SibSp * CabinType=B', 'SibSp * CabinType=C', 'SibSp * CabinType=D', 'SibSp * CabinType=E', 'SibSp * CabinType=F', 'SibSp * CabinType=G', 'SibSp * CabinType=None', 'SibSp * CabinType=T', 'SibSp>0 * Parch>0', 'SibSp>0 * Embarked=C', 'SibSp>0 * Embarked=None', 'SibSp>0 * Embarked=Q', 'SibSp>0 * Embarked=S', 'SibSp>0 * CabinType=A', 'SibSp>0 * CabinType=B', 'SibSp>0 * CabinType=C', 'SibSp>0 * CabinType=D', 'SibSp>0 * CabinType=E', 'SibSp>0 * CabinType=F', 'SibSp>0 * CabinType=G', 'SibSp>0 * CabinType=None', 'SibSp>0 * CabinType=T', 'Parch>0 * Embarked=C', 'Parch>0 * Embarked=None', 'Parch>0 * Embarked=Q', 'Parch>0 * Embarked=S', 'Parch>0 * CabinType=A', 'Parch>0 * CabinType=B', 'Parch>0 * CabinType=C', 'Parch>0 * CabinType=D', 'Parch>0 * CabinType=E', 'Parch>0 * CabinType=F', 'Parch>0 * CabinType=G', 'Parch>0 * CabinType=None', 'Parch>0 * CabinType=T', 'Embarked=C * CabinType=A', 'Embarked=C * CabinType=B', 'Embarked=C * CabinType=C', 'Embarked=C * CabinType=D', 'Embarked=C * CabinType=E', 'Embarked=C * CabinType=F', 'Embarked=C * CabinType=G', 'Embarked=C * CabinType=None', 'Embarked=C * CabinType=T', 'Embarked=None * CabinType=A', 'Embarked=None * CabinType=B', 'Embarked=None * CabinType=C', 'Embarked=None * CabinType=D', 'Embarked=None * CabinType=E', 'Embarked=None * CabinType=F', 'Embarked=None * CabinType=G', 'Embarked=None * CabinType=None', 'Embarked=None * CabinType=T', 'Embarked=Q * CabinType=A', 'Embarked=Q * CabinType=B', 'Embarked=Q * CabinType=C', 'Embarked=Q * CabinType=D', 'Embarked=Q * CabinType=E', 'Embarked=Q * CabinType=F', 'Embarked=Q * CabinType=G', 'Embarked=Q * CabinType=None', 'Embarked=Q * CabinType=T', 'Embarked=S * CabinType=A', 'Embarked=S * CabinType=B', 'Embarked=S * CabinType=C', 'Embarked=S * CabinType=D', 'Embarked=S * CabinType=E', 'Embarked=S * CabinType=F', 'Embarked=S * CabinType=G', 'Embarked=S * CabinType=None', 'Embarked=S * CabinType=T']


def backwards_selection():
  temp_df = df.copy()
  del temp_df['id']
  baseline_accuracy = fit_regressor(temp_df)

  for feature in features:
    print('\nfeature', feature)
    print('baseline_accuracy', baseline_accuracy)
    del temp_df[feature]
    new_accuracy = fit_regressor(temp_df)
    print('new_accuracy', new_accuracy)
    if new_accuracy < baseline_accuracy:
      print('kept')
      temp_df[feature] = df[feature]
    else:
      baseline_accuracy = new_accuracy

  return fit_regressor(temp_df)

print(backwards_selection())