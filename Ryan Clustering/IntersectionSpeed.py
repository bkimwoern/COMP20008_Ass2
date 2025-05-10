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
        dfCluster.to_csv('intersectOutput/cluster' + str(i) + '.csv', index=False)


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
        with open('intersectOutput/dataC' + str(i) + '.json', 'w') as f:
            json.dump(stats, f, indent=4)

        i = i + 1


killMulti = 3
injurySeries = 2
injury = 1


# Importing data
accident_data = pd.read_csv('../datasets/filtered_accident.csv')
person_data = pd.read_csv('../datasets/filtered_person.csv')

# Taking realistic speed zone for Australia
accident_data = accident_data[accident_data['SPEED_ZONE'] <= 130]
# Only taking fatal accidents
accident_data = accident_data[accident_data['SEVERITY'] == 1]

# Making a severity index that weights deadly crashes higher based on kills, serious injuries and injuries
accident_data['severity_index'] = accident_data['NO_PERSONS_KILLED'] * killMulti + accident_data['NO_PERSONS_INJ_2'] * injurySeries + accident_data['NO_PERSONS_INJ_3'] * injury


# one hot encoding ACCIDENT_TYPE
dummies = pd.get_dummies(accident_data['ACCIDENT_TYPE'], prefix='ACCIDENT_TYPE', dtype=int)
accident_data = pd.concat([accident_data, dummies], axis=1)
#accident_data = pd.get_dummies(accident_data, columns=['ACCIDENT_TYPE'], dtype=int)
accident_data.to_csv('test.csv', index=False)


seed = 1


# Filter down to attributes we want to cluster on
cluster_data = accident_data[['SPEED_ZONE',
                               'ACCIDENT_TYPE_1',
                               'ACCIDENT_TYPE_2',
                               'ACCIDENT_TYPE_3',
                               'ACCIDENT_TYPE_4',
                               'ACCIDENT_TYPE_5',
                               'ACCIDENT_TYPE_6',
                               'ACCIDENT_TYPE_7',
                               'ACCIDENT_TYPE_8',
                               'ACCIDENT_TYPE_9',]]

print(cluster_data.size)

# Copy data to normalise
normalised_data = cluster_data.copy(deep=True)
numeric_cols = normalised_data.select_dtypes(include='number').columns
normalised_data[numeric_cols] = (normalised_data[numeric_cols] - normalised_data[numeric_cols].min()) / (normalised_data[numeric_cols].max() - normalised_data[numeric_cols].min())

# Using elbow method to find best k value: The following code is from week 6 workshop
distortions = []
k_range = range(1, 15)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(normalised_data)
    distortions.append(kmeans.inertia_)  # The sum of squared errors

# Plotting and saving figure
pt.plot(k_range, distortions, 'bx-')
pt.title('Day, Hour and number of people killed clustering')
pt.xlabel('k Values')
pt.ylabel('Distortion')
pt.savefig('intersectElbow.png')

print(normalised_data)

clusters = KMeans(n_clusters=4, random_state=seed)
clusters.fit(normalised_data)

colormap = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'cadetblue', 5: 'orchid', 6: 'lime'}


# Plotting and saving figure, 3 dimensional
fig = pt.figure(figsize=(7, 7))
ax = pt.axes(projection="3d")
ax.scatter(accident_data['ACCIDENT_TYPE'],
           accident_data['SPEED_ZONE'],
           accident_data['severity_index'],
           c=[colormap.get(x) for x in clusters.labels_], alpha=0.2)


ax.set_xlabel('Accident Type')
ax.set_ylabel('Speed Zone')
ax.set_zlabel('Severity Index')
ax.set_title(f"Clusters on Speed Zone and Protection worn; k = {len(set(clusters.labels_))}")

pt.savefig('intersectionCluster.png')


# Outputting clusters to individual CSV files
crash_data = accident_data[['SPEED_ZONE', 'ACCIDENT_TYPE', 'severity_index']]
outputClusters(clusters.labels_, ['SPEED_ZONE'], True, crash_data)


# To do:
"""
- Create a severity_index column
- Get all one varible data from the clusters and categories the clusters

"""
