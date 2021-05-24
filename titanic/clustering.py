import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


unnormalized = pd.read_csv('titanic/csv/processed_data.csv')
min_max = unnormalized[["Sex", "Pclass", "Fare", "Age", "SibSp"]].copy()
df = unnormalized[["Sex", "Pclass", "Fare", "Age", "SibSp"]].copy()

for column in min_max:
  if column != 'Survived':
    min_max[column] = (min_max[column]-min(df[column]))/(max(df[column])-min(df[column]))

def create_clusters(k):
  cluster_dict = {key+1:[value] for key,value in zip(range(k),range(k))}
  for num in range(k,len(data)):
    cluster_dict[(num%k)+1].append(num)
  return cluster_dict

k_list = []
error_list = []
for k in range(1, 26):
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(min_max)
  error = kmeans.inertia_
  
  k_list.append(k)
  error_list.append(error)

plt.plot(k_list, error_list)
plt.xticks([x for x in range(1,26)])
plt.xlabel('k')
plt.ylabel('sum squared accuracy')
plt.savefig('titanic/plots/elbow_graph.png')


kmeans = KMeans(n_clusters=4)
kmeans.fit(df)

df['cluster'] = kmeans.labels_
counts = df['cluster'].value_counts()
df['Survived'] = unnormalized['Survived'].copy()
print(df)
df = df.groupby(by='cluster').mean()
df['count'] = counts


print(df)
