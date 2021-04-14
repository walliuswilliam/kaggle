import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
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

df.loc[age_nan, ['Age']] = df['Age'][age_not_nan].mean()


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


# df_train = df[:500]
# df_test = df[500:]

# arr_train = np.array(df_train)
# arr_test = np.array(df_test)

# y_train = arr_train[:,0]
# y_test = arr_test[:,0]

# x_train = arr_train[:,1:]
# x_test = arr_test[:,1:]

# regressor = LogisticRegression(max_iter=1000)
# regressor.fit(x_train, y_train)

# coef_dict = {}
# feature_columns = df_train.columns[1:]
# feature_coefficients = regressor.coef_

# for i in range(len(feature_columns)):
#   column = feature_columns[i]
#   coefficient = feature_coefficients[0][i]
#   coef_dict[column] = coefficient

# y_test_predictions = regressor.predict(x_test)
# y_train_predictions = regressor.predict(x_train)

def convert_regressor_output_to_survival_value(output):
  if output < 0.5:
    return 0
  else:
    return 1

# y_test_predictions = [convert_regressor_output_to_survival_value(output) for output in y_test_predictions]
# y_train_predictions = [convert_regressor_output_to_survival_value(output) for output in y_train_predictions]

def get_accuracy(predictions, actual):
  num_correct = 0
  num_incorrect = 0
  for i in range(len(predictions)):
    if predictions[i] == actual[i]:
      num_correct += 1
    else:
      num_incorrect += 1
  return num_correct / (num_correct + num_incorrect)


# print('\nfeatures:', features_to_use)
# print('\ntraining accuracy:', get_accuracy(y_train_predictions, y_train))
# print('testing accuracy:', get_accuracy(y_test_predictions, y_test), '\n')

# coef_dict['constant'] = regressor.intercept_[0]
# print('coefficients', {a:round(b,4) for a,b in coef_dict.items()})


terms = ['Sex', 'Pclass', 'Fare', 'Age', 'SibSp', 'SibSp>0', 'Parch>0', 'Embarked=C', 'Embarked=None', 'Embarked=Q', 'Embarked=S', 'CabinType=A', 'CabinType=B', 'CabinType=C', 'CabinType=D', 'CabinType=E', 'CabinType=F', 'CabinType=G', 'CabinType=None', 'CabinType=T']

interaction_list = []
for term in terms:
  for term_2 in terms[terms.index(term)+1:]:
    embarked = 'Embarked' in term and 'Embarked' in term_2
    cabin = 'Cabin' in term and 'Cabin' in term_2
    sib = 'Sib' in term and 'Sib' in term_2
    if not embarked and not cabin and not sib:
      interaction_list.append(term + ' * ' + term_2)

for term in interaction_list:
  interactions = term.split(' * ')
  df[term] = df[interactions[0]]*df[interactions[1]]

df_train = df[:500]
df_test = df[500:]

arr_train = np.array(df_train)
arr_test = np.array(df_test)

y_train = arr_train[:,0]
y_test = arr_test[:,0]

x_train = arr_train[:,1:]
x_test = arr_test[:,1:]

regressor = LogisticRegression(max_iter=10000)
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


print('training accuracy:', round(get_accuracy(y_train_predictions, y_train), 3))
print('testing accuracy:', round(get_accuracy(y_test_predictions, y_test), 3))
