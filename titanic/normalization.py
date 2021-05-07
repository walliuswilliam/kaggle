import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time

#relative time = 2x
start_time = time.time()

unnormalized = pd.read_csv('titanic/csv/processed_data.csv')
unnormalized = unnormalized[["Survived", "Sex", "Pclass", "Fare", "Age","SibSp"]].head(100)
simple_scaling = unnormalized.copy()
min_max = unnormalized.copy()
z_score = unnormalized.copy()

dependent_var = 'Survived'

for column in simple_scaling:
  if column != dependent_var:
    simple_scaling[column] = simple_scaling[column]/max(unnormalized[column])

for column in min_max:
  if column != dependent_var:
    min_max[column] = (min_max[column]-min(unnormalized[column]))/(max(unnormalized[column])-min(unnormalized[column]))

for column in z_score:
  if column != dependent_var:
    z_score[column] = (z_score[column] - unnormalized[column].mean())/unnormalized[column].std()

df_list = [unnormalized, simple_scaling, min_max, z_score]

def get_accuracies(df, k_list, dependent_var):
  arr_reset = df.to_numpy().tolist()
  dv_index = list(df.columns).index(dependent_var)
  accuracies = []
  for k in k_list:
    accuracy = 0
    for row_index in range(len(arr_reset)):
      arr = arr_reset.copy()
      real_value = arr[row_index][dv_index]
      observation = arr[row_index][:dv_index] + arr[row_index][dv_index+1:]

      arr = np.delete(arr, row_index, 0)

      y = [row[0] for row in arr]
      x = np.delete(arr, 0, 1)

      knn = KNeighborsClassifier(n_neighbors=k)
      knn.fit(x, y)
      prediction = knn.predict([observation])[0]
      if prediction == real_value:
        accuracy += 1
    accuracies.append(accuracy/df.shape[0])
  return accuracies

plt.style.use('bmh')
k_list = [x for x in range(100) if x%2 == 1]
for df in df_list:
  plt.plot(k_list, get_accuracies(df, k_list, dependent_var))

plt.xlabel('k')
plt.ylabel('accuracy')
plt.legend(['unnormalized', 'simple_scaling', 'min_max', 'z_score'])
plt.title('Leave-One-Out Accuracy for Various Normalizations')
plt.savefig('titanic/plots/normalization.png')

end_time = time.time()
print('time taken:', (end_time - start_time)/2, 'seconds')