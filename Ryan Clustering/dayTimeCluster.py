"""This file is used to cluster time based attributes to find patterns that relate to fatal accidents"""
import matplotlib.pyplot as pt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json


"""
Function takes in clusters, and sorting features. It out puts each cluster to its own CSV and also output one 
varible statiics on all columns for the cluster to a seperate JSON file.
"""
def outputClusters(clustersLabels, sortFeatures, ascending, clusters_data):
    numClusters = len(set(clustersLabels)) # The number of clusters
    clusters = [[] for _ in range(numClusters)] # Creates a array of lists to hold cluster values

    i=0
    for row in clusters_data.iterrows():
        clusters[clustersLabels[i]].append(row[1])
        i = i + 1

    i = 0
    for cluster in clusters:
        dfCluster = pd.DataFrame(cluster, columns=clusters_data.columns).round(decimals=2).sort_values(by=sortFeatures, ascending=ascending)
        dfCluster.to_csv('timeOutput/cluster' + str(i) + '.csv', index=False)


        stats = {}
        # Loop over numeric columns and calculate statistics
        for col in dfCluster.columns:
            stats[col] = {
                'mean': float(dfCluster[col].mean()),
                'median': float(dfCluster[col].median()),
                'std_dev': float(dfCluster[col].std()),
                'min': float(dfCluster[col].min()),
                'max': float(dfCluster[col].max()),
                'count': float(dfCluster[col].count()),
            }

        # Save to JSON file
        with open('timeOutput/dataC' + str(i) + '.json', 'w') as f:
            json.dump(stats, f, indent=4)

        i = i + 1


# Importing data
accident_data = pd.read_csv('../datasets/filtered_accident.csv')

# Format time
accident_data['ACCIDENT_TIME'] = pd.to_datetime(accident_data['ACCIDENT_TIME'], format='%H:%M:%S')

# Number time by hour
accident_data['hour'] = accident_data['ACCIDENT_TIME'].dt.hour

# Counting number of each day
days_counts = accident_data['DAY_OF_WEEK'].value_counts().sort_index()

# Creating weights for each day (10000 is added so the number isn't tiny)
days_weights = 10000 / days_counts

# counting number of each hour
hour_counts = accident_data['hour'].value_counts().sort_index()

# Creating weights for each hour (10000 is added so the number isn't tiny)
hour_weights = 10000 / hour_counts

# Applying weights to people killed
accident_data['weighted_ppl_killed'] = accident_data['NO_PERSONS_KILLED'] * accident_data['DAY_OF_WEEK'].map(days_weights)
accident_data['weighted_ppl_killed'] = accident_data['weighted_ppl_killed'] * accident_data['hour'].map(hour_weights)

# Drop values with no deaths
accident_data = accident_data[accident_data['SEVERITY'] == 1]

# Group by hour and day of the week and count sum of deaths
time_data = accident_data.groupby(['hour','DAY_OF_WEEK'])['weighted_ppl_killed'].sum().reset_index()

# Copy data to normalise
normalisedTime_data = time_data.copy(deep=True)

"""Because time is cyclic we need to treat the time varibles carefully. For example weekdays are numbered 1-7. But 
1 and 7 are as close together as 1 and 2. Thus we use circular functions (sin and cos) to make new columns that the
clustering will use. This will ensure time can be clustered properly"""
normalisedTime_data['day_sin'] = np.sin(2 * np.pi * time_data['DAY_OF_WEEK'] / 7)
normalisedTime_data['day_cos'] = np.cos(2 * np.pi * time_data['DAY_OF_WEEK'] / 7)
normalisedTime_data['hour_sin'] = np.sin(2 * np.pi * time_data['hour'] / 24)
normalisedTime_data['hour_cos'] = np.cos(2 * np.pi * time_data['hour'] / 24)

# Dropping non adjusted columns
normalisedTime_data = normalisedTime_data.drop(columns=['DAY_OF_WEEK'])
normalisedTime_data = normalisedTime_data.drop(columns=['hour'])

# Normalise weighted deaths to be between -1 and 1
normalisedTime_data['weighted_ppl_killed'] = 2 * ((normalisedTime_data['weighted_ppl_killed'] - normalisedTime_data['weighted_ppl_killed'].min()) / (normalisedTime_data['weighted_ppl_killed'].max() - normalisedTime_data['weighted_ppl_killed'].min())) -1



# Using elbow method to find best k value: The following code is from week 6 workshop
seed = 120
distortions = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(normalisedTime_data)
    distortions.append(kmeans.inertia_)  # The sum of squared errors

# Plotting and saving figure
pt.plot(k_range, distortions, 'bx-')
pt.title('Day, Hour and number of people killed clustering')
pt.xlabel('k Values')
pt.ylabel('Distortion')
pt.savefig('DayTimeClusteringElbow.png')

# K value of 7 or 8 was found to be useful
clusters = KMeans(n_clusters=9, random_state=seed)
clusters.fit(normalisedTime_data)

colormap = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'purple',
    5: 'cyan',
    6: 'magenta',
    7: 'gold',
    8: 'black'
}

# Plotting and saving figure
fig = pt.figure(figsize=(7, 7))
ax = pt.axes(projection="3d")
ax.scatter(time_data['hour'],
           time_data['DAY_OF_WEEK'],
           time_data['weighted_ppl_killed'],
           c=[colormap.get(x) for x in clusters.labels_])

ax.set_ylabel('Day of the Week')
ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
ax.set_yticklabels(["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])



ax.set_xlabel('Hour')
ax.set_zlabel('Weighted people killed')
ax.set_title(f"Day, Time clustering against weighted deaths; k= {len(set(clusters.labels_))}")
pt.savefig('DayTimeClustering.png')

outputClusters(clusters.labels_, ['DAY_OF_WEEK', 'hour'], False, time_data)


