import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import sys


df = pd.read_csv('titanic/csv/processed_data.csv')
df = df.head(100)
df = df[['Survived','Sex','Pclass','Fare','Age','SibSp']]


arr_reset = df.to_numpy()
accuracies = []
k_list = [1,3,5,10,15,20,30,40,50,75]
for k in k_list:
  accuracy = 0
  for row_index in range(len(arr_reset)):
    arr = arr_reset.copy()
    real_value = arr[row_index][0]
    observation = arr[row_index][1:]
    arr = np.delete(arr, row_index, 0)

    y = [row[0] for row in arr]
    x = np.delete(arr, 0, 1)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, y)
    prediction = knn.predict([observation])[0]
    if prediction == real_value:
      accuracy += 1
  accuracies.append(accuracy/df.shape[0])


plt.style.use('bmh')
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Leave One Out Cross Validation')
plt.savefig('titanic/leave_one_out.png')