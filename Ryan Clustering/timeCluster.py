"""This file is used to cluster time based attributes to find patterns that relate to fatal accidents"""
import matplotlib.pyplot as pt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Importing data
accident_data = pd.read_csv('../datasets/accident.csv')

# Format time
accident_data['ACCIDENT_TIME'] = pd.to_datetime(accident_data['ACCIDENT_TIME'], format='%H:%M:%S')

# Number time by hour
accident_data['hour'] = accident_data['ACCIDENT_TIME'].dt.hour

# counting number of each hour
hour_counts = accident_data['hour'].value_counts().sort_index()

# Creating weights for each hour (10000 is added so the number isn't tiny)
hour_weights = 10000 / hour_counts

# Applying weights to people killed
accident_data['weighted_ppl_killed'] = accident_data['NO_PERSONS_KILLED'] * accident_data['hour'].map(hour_weights)

# Drop values with no deaths
accident_data = accident_data[accident_data['SEVERITY'] == 1]

# Group by hour and count sum of deaths
time_data = accident_data.groupby(['hour'])['weighted_ppl_killed'].sum().reset_index()

# Copy data to normalise
normalisedTime_data = time_data.copy(deep=True)

"""Because time is cyclic we need to treat the time varibles carefully. For example, hours are numbered 0-23. But 
0 and 23 are as close together as 0 and 1. Thus we use circular functions (sin and cos) to make new columns that the
clustering will use. This will ensure time can be clustered properly"""
normalisedTime_data['hour_sin'] = np.sin(2 * np.pi * time_data['hour'] / 24)
normalisedTime_data['hour_cos'] = np.cos(2 * np.pi * time_data['hour'] / 24)

# Dropping non adjusted columns
normalisedTime_data = normalisedTime_data.drop(columns=['hour'])

# Normalise weighted deaths to be between -1 and 1
normalisedTime_data['weighted_ppl_killed'] = 2 * ((normalisedTime_data['weighted_ppl_killed'] - normalisedTime_data['weighted_ppl_killed'].min()) / (normalisedTime_data['weighted_ppl_killed'].max() - normalisedTime_data['weighted_ppl_killed'].min())) -1



# Using elbow method to find best k value: The following code is from week 6 workshop
distortions = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(normalisedTime_data)
    distortions.append(kmeans.inertia_)  # The sum of squared errors

# Plotting and saving figure
pt.plot(k_range, distortions, 'bx-')
pt.title('Hour and number of people killed clustering')
pt.xlabel('k Values')
pt.ylabel('Distortion')
pt.savefig('TimeClusteringElbow.png')

# K value of 7 or 8 was found to be useful
clusters = KMeans(n_clusters=4)
clusters.fit(normalisedTime_data)

colormap = {0: 'red', 1: 'green', 2: 'blue', 3: 'darkviolet', 4: 'orange', 5: 'cadetblue', 6: 'orchid', 7: 'lime'}

# Plotting and saving figure
pt.figure(figsize=(7, 7))
pt.scatter(time_data['hour'], time_data['weighted_ppl_killed'],
           c=[colormap.get(x) for x in clusters.labels_], alpha=0.4)
pt.xlabel('Hour (0-23)')
pt.ylabel('Weighted People Killed')
pt.title('Clustering based on number of people killed and hour of the day')
pt.savefig('timeCluster.png')

# Putting top 10 rows within a cluser into its own csv file for output
# Using lists whcih will be converted to dataFrames later (more efficent to use lists when appending rows repeatdly)
cluster0lst = []
cluster1lst = []
cluster2lst = []
cluster3lst = []
cluster4lst = []
cluster5lst = []
cluster6lst = []
cluster7lst = []


i = 0
for row in time_data.iterrows():
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

cluster0 = pd.DataFrame(cluster0lst, columns=time_data.columns).round(decimals=2)
cluster1 = pd.DataFrame(cluster1lst, columns=time_data.columns).round(decimals=2)
cluster2 = pd.DataFrame(cluster2lst, columns=time_data.columns).round(decimals=2)
cluster3 = pd.DataFrame(cluster3lst, columns=time_data.columns).round(decimals=2)
cluster4 = pd.DataFrame(cluster4lst, columns=time_data.columns).round(decimals=2)
cluster5 = pd.DataFrame(cluster5lst, columns=time_data.columns).round(decimals=2)
cluster6 = pd.DataFrame(cluster6lst, columns=time_data.columns).round(decimals=2)
cluster7 = pd.DataFrame(cluster7lst, columns=time_data.columns).round(decimals=2)

# Sorting based on Crash count
cluster0 = cluster0.sort_values(by=['hour'], ascending=False)
cluster1 = cluster1.sort_values(by=['hour'], ascending=False)
cluster2 = cluster2.sort_values(by=['hour'], ascending=False)
cluster3 = cluster3.sort_values(by=['hour'], ascending=False)
cluster4 = cluster4.sort_values(by=['hour'], ascending=False)
cluster5 = cluster5.sort_values(by=['hour'], ascending=False)
cluster6 = cluster6.sort_values(by=['hour'], ascending=False)
cluster7 = cluster7.sort_values(by=['hour'], ascending=False)


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


# HOUR IS ALSO CIRCULAR