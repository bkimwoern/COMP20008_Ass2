# Clustering to determine how age is related to crashes
""""
PLAN
Clustering based on:
light condition
speed zone
Crash Severity


Plot against: Aga vs Crash Severity
"""
import matplotlib.pyplot as pt
import numpy as np
import pandas as pd
from seaborn import clustermap
from sklearn.cluster import KMeans

def maxage(age_interval):
    value = 0
    if '-' in age_interval:
        value =  age_interval.split('-')[1]
    if '+' in age_interval:
        value = age_interval.replace('+','')
    try:
        if value ==  0: return 0
        return int(value)
    except ValueError:
        return 0




# Importing data
accident_data = pd.read_csv('../datasets/filtered_accident.csv')
person_data = pd.read_csv('../datasets/person.csv')
accident_data = accident_data[accident_data['SPEED_ZONE'] < 120]

# Only looking at drivers and passanages
person_data = person_data[~person_data['ROAD_USER_TYPE'].isin([1,7,9])]
person_data = person_data[person_data['AGE_GROUP'] != 'Unknown']
print(person_data)

# Creating new column called restriats worn
person_data['RESTRAINT_WORN'] = person_data['HELMET_BELT_WORN'].apply(lambda x: 1 if x in [1, 4, 7] else 0)
person_data['SEX'] = person_data['SEX'].apply(lambda x: 1 if x == 'M' else 0)

# Take the max possible age of person
person_data['MAX_AGE'] = person_data['AGE_GROUP'].apply(maxage)
person_data.to_csv('newperson.csv', index=False)
person_data = person_data[person_data['MAX_AGE'] != 0]
person_data = person_data[person_data['MAX_AGE'] != 4]

# Merge accident and person data by accident number
crash_data = pd.merge(accident_data, person_data, how='inner', on='ACCIDENT_NO')
crash_data.to_csv('newdata.csv', index=False)

cluster_data1 = crash_data.groupby('SEVERITY', group_keys=False).sample(n=2000)
cluster_data1 = cluster_data1.groupby('MAX_AGE', group_keys=False).sample(n=100)

cluster_data = cluster_data1[['RESTRAINT_WORN', 'SPEED_ZONE', 'MAX_AGE', 'SEX', 'LIGHT_CONDITION']]

# Copy data to normalise
normalised_data = cluster_data.copy(deep=True)
numeric_cols = normalised_data.select_dtypes(include='number').columns
normalised_data[numeric_cols] = (normalised_data[numeric_cols] - normalised_data[numeric_cols].min()) / (normalised_data[numeric_cols].max() - normalised_data[numeric_cols].min())

# Using elbow method to find best k value: The following code is from week 6 workshop
distortions = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(normalised_data)
    distortions.append(kmeans.inertia_)  # The sum of squared errors

# Plotting and saving figure
pt.plot(k_range, distortions, 'bx-')
pt.title('Day, Hour and number of people killed clustering')
pt.xlabel('k Values')
pt.ylabel('Distortion')
pt.savefig('ageHazardElbow.png')

print(normalised_data)

# K value of 7 or 8 was found to be useful
clusters = KMeans(n_clusters=4)
clusters.fit(normalised_data)

colormap = {0: 'red', 1: 'green', 2: 'blue', 3: 'darkviolet', 4: 'orange', 5: 'cadetblue', 6: 'orchid', 7: 'lime'}

# Plotting and saving figure
fig = pt.figure(figsize=(7, 7))
ax = pt.axes(projection="3d")
ax.scatter(cluster_data1['MAX_AGE'],
           cluster_data1['SEVERITY'],
           c=[colormap.get(x) for x in clusters.labels_])


ax.set_xlabel('MAX_AGE')
ax.set_ylabel('SEVERITY')
ax.set_title(f"k = {len(set(clusters.labels_))}")

pt.savefig('DayTimeClustering.png')

# Clustering Data
clusters = KMeans(n_clusters=3)
clusters.fit(normalised_data)

# Colour map to assign different colour to each dot in scatterplot corresponding to its assigned cluser
colourmap = {0: 'cadetblue', 1: 'green', 2: 'blue'}

# Plotting coloured scatter plot
pt.figure(figsize=(12, 10))
pt.scatter(cluster_data1['MAX_AGE'], cluster_data1['SEVERITY'],
           c=[colourmap.get(x) for x in clusters.labels_], alpha=0.4)
pt.xlabel('MAX_AGE')
pt.ylabel('SEVERITY')
pt.title('Crashes based on manufacture year, brand make, and body style: Coloured Based on K Means clustering')
pt.savefig('task3_3_scattercolour.png')



#NOTES, YOU DONT NEED TO USE GROUPBY. YOU CAN JUST CLUSTER THE RAW DATASET
cluster0lst = []
cluster1lst = []
cluster2lst = []
cluster3lst = []
cluster4lst = []
cluster5lst = []
cluster6lst = []
cluster7lst = []

cluster_data1 = cluster_data1[['SEVERITY', 'MAX_AGE', 'SEX']]
i = 0
for row in cluster_data1.iterrows():
    if clusters.labels_[i] == 0:
        cluster0lst.append(row[1])
    elif clusters.labels_[i] == 1:
        cluster1lst.append(row[1])
    elif clusters.labels_[i] == 2:
        cluster2lst.append(row[1])
    elif clusters.labels_[i] == 3:
        cluster3lst.append(row[1])
    elif clusters.labels_[i] == 4:
        cluster4lst.append(row[1])
    elif clusters.labels_[i] == 5:
        cluster5lst.append(row[1])
    elif clusters.labels_[i] == 6:
        cluster6lst.append(row[1])
    elif clusters.labels_[i] == 7:
        cluster7lst.append(row[1])
    else:
        print("ERROR")
        break
    i = i + 1

cluster0 = pd.DataFrame(cluster0lst, columns=cluster_data1.columns).round(decimals=2)
cluster1 = pd.DataFrame(cluster1lst, columns=cluster_data1.columns).round(decimals=2)
cluster2 = pd.DataFrame(cluster2lst, columns=cluster_data1.columns).round(decimals=2)
cluster3 = pd.DataFrame(cluster3lst, columns=cluster_data1.columns).round(decimals=2)
cluster4 = pd.DataFrame(cluster4lst, columns=cluster_data1.columns).round(decimals=2)
cluster5 = pd.DataFrame(cluster5lst, columns=cluster_data1.columns).round(decimals=2)
cluster6 = pd.DataFrame(cluster6lst, columns=cluster_data1.columns).round(decimals=2)
cluster7 = pd.DataFrame(cluster7lst, columns=cluster_data1.columns).round(decimals=2)

# Sorting based on Crash count
cluster0 = cluster0.sort_values(by=['SEVERITY'], ascending=True)
cluster1 = cluster1.sort_values(by=['SEVERITY'], ascending=True)
cluster2 = cluster2.sort_values(by=['SEVERITY'], ascending=True)
cluster3 = cluster3.sort_values(by=['SEVERITY'], ascending=True)
cluster4 = cluster4.sort_values(by=['MAX_AGE'], ascending=False)
cluster5 = cluster5.sort_values(by=['MAX_AGE'], ascending=False)
cluster6 = cluster6.sort_values(by=['MAX_AGE'], ascending=False)
cluster7 = cluster7.sort_values(by=['MAX_AGE'], ascending=False)


# Putting top 10 results into CSVs
numValues = 10
cluster0.to_csv("cluster0.csv", index=False)
cluster1.to_csv("cluster1.csv", index=False)
cluster2.to_csv("cluster2.csv", index=False)
cluster3.to_csv("cluster3.csv", index=False)
cluster4.to_csv("cluster4.csv", index=False)
cluster5.to_csv("cluster5.csv", index=False)
cluster6.to_csv("cluster6.csv", index=False)
cluster7.to_csv("cluster7.csv", index=False)