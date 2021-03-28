import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

df = pd.read_csv('titanic/train.csv')

keep_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df = df[keep_cols]

def convert_sex_to_int(sex):
  if sex == 'male':
    return 0
  elif sex == 'female':
    return 1

df['Sex'] = df['Sex'].apply(convert_sex_to_int)

age_nan = df['Age'].apply(lambda entry: np.isnan(entry))
age_not_nan = df['Age'].apply(lambda entry: not np.isnan(entry))
mean_age = df['Age'][age_not_nan].mean()

df['Age'][age_nan] = mean_age


def dummy_greater_than_0(x):
  if x > 0:
    return 1
  else:
    return 0


df['SibSp>0'] = df['SibSp'].apply(dummy_greater_than_0)

df['Parch>0'] = df['Parch'].apply(dummy_greater_than_0)


df['Cabin']= df['Cabin'].fillna('None')

def get_cabin_type(cabin):
  if cabin != 'None':
    return cabin[0]
  else:
    return cabin

df['CabinType'] = df['Cabin'].apply(get_cabin_type)

for cabin_type in df['CabinType'].unique():
  dummy_variable_name = 'CabinType={}'.format(cabin_type)
  dummy_variable_values = df['CabinType'].apply(lambda entry: int(entry == cabin_type))
  df[dummy_variable_name] = dummy_variable_values

del df['CabinType']


df['Embarked'] = df['Embarked'].fillna('None')

for embarked in df['Embarked'].unique():
  dummy_variable_name = 'Embarked={}'.format(embarked)
  dummy_variable_values = df['Embarked'].apply(lambda entry: int(entry == embarked))
  df[dummy_variable_name] = dummy_variable_values

del df['Embarked']


features_to_use = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']
columns_needed = ['Survived'] + features_to_use
df = df[columns_needed]


df_train = df[:500]
df_test = df[500:]

arr_train = np.array(df_train)
arr_test = np.array(df_test)

y_train = arr_train[:,0]
y_test = arr_test[:,0]

X_train = arr_train[:,1:]
X_test = arr_test[:,1:]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

coef_dict = {}
feature_columns = df_train.columns[1:]
feature_coefficients = regressor.coef_
for i in range(len(feature_columns)):
  column = feature_columns[i]
  coefficient = feature_coefficients[i]
  coef_dict[column] = coefficient

y_test_predictions = regressor.predict(X_test)
y_train_predictions = regressor.predict(X_train)

def convert_regressor_output_to_survival_value(output):
  if output < 0.5:
    return 0
  else:
    return 1

y_test_predictions = [convert_regressor_output_to_survival_value(output) for output in y_test_predictions]
y_train_predictions = [convert_regressor_output_to_survival_value(output) for output in y_train_predictions]

def get_accuracy(predictions, actual):
  num_correct = 0
  num_incorrect = 0
  for i in range(len(predictions)):
    if predictions[i] == actual[i]:
      num_correct += 1
    else:
      num_incorrect += 1
  return num_correct / (num_correct + num_incorrect)


print('\nfeatures:', features_to_use)
print('\ntraining accuracy:', get_accuracy(y_train_predictions, y_train))
print('testing accuracy:', get_accuracy(y_test_predictions, y_test), '\n')

coef_dict['constant'] = regressor.intercept_
print('coefficients', {k:round(v,4) for k,v in coef_dict.items()})
