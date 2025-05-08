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
Function takes in age interval and returns the upperbound of the interval.
If a value is unable to be turned into an integer the function returns 0
"""
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
        dfCluster.to_csv('outputs/cluster' + str(i) + '.csv', index=False)


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
        with open('outputs/dataC' + str(i) + '.json', 'w') as f:
            json.dump(stats, f, indent=4)

        i = i + 1


killMulti = 3
injurySeries = 2
injury = 1


# Importing data
accident_data = pd.read_csv('../datasets/filtered_accident.csv')
person_data = pd.read_csv('../datasets/person.csv')

# Taking realistic speed zone for Australia
accident_data = accident_data[accident_data['SPEED_ZONE'] <= 130]
# Only taking fatal accidents
accident_data = accident_data[accident_data['SEVERITY'] == 1]

# Making a severity index that weights deadly crashes higher based on kills, serious injuries and injuries
accident_data['severity_index'] = accident_data['NO_PERSONS_KILLED'] * killMulti + accident_data['NO_PERSONS_INJ_2'] * injurySeries + accident_data['NO_PERSONS_INJ_3'] * injury



# Only looking at drivers and passangers risk assesment
person_data = person_data[~person_data['ROAD_USER_TYPE'].isin([1,6,9])]

# Removing values where the age group is unknown
person_data = person_data[person_data['AGE_GROUP'] != 'Unknown']

# Creating new column called restriats worn
person_data['RESTRAINT_WORN'] = person_data['HELMET_BELT_WORN'].apply(lambda x: 1 if x in [1, 3, 6] else 0)

# Cleaning sex and assigning number, 1 if a man and 0 if a female
person_data = person_data[person_data['SEX'].isin(['M', 'F'])]
person_data['SEX'] = person_data['SEX'].apply(lambda x: 1 if x == 'M' else 0)

# Take the max possible age of person, and clean
person_data['MAX_AGE'] = person_data['AGE_GROUP'].apply(maxage)
person_data.to_csv('newperson.csv', index=False)
person_data = person_data[person_data['MAX_AGE'] != 0]
person_data = person_data[person_data['MAX_AGE'] != 4]

# Merge accident and person data by accident number
crash_data = pd.merge(accident_data, person_data, how='inner', on='ACCIDENT_NO')
crash_data.to_csv('newdata.csv', index=False)


crash_data = crash_data[crash_data['SEVERITY'] != 4]

# Sample the data to get a even distrubution of seversity and age group
#cluster_data1 = crash_data.groupby('SEVERITY', group_keys=False).sample(n=crash_data['SEVERITY'].value_counts().min(), replace=True)
cluster_data2 = crash_data.groupby('MAX_AGE', group_keys=False).sample(n=crash_data['MAX_AGE'].value_counts().min())
cluster_data2 = crash_data
# Filter down to attributes we want to cluster on
cluster_data = cluster_data2[['SPEED_ZONE', 'RESTRAINT_WORN']]
# ['SPEED_ZONE', 'RESTRAINT_WORN'] with k = 3
# ['SPEED_ZONE', 'MAX_AGE', 'RESTRAINT_WORN'] with k = 3,4
# ['SPEED_ZONE', 'MAX_AGE']

# ['SPEED_ZONE', 'MAX_AGE', 'RESTRAINT_WORN'] k =3 when looking at SEVERITY ==1

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

clusters = KMeans(n_clusters=4)
clusters.fit(normalised_data)

colormap = {0: 'red', 1: 'green', 2: 'blue', 3: 'darkviolet', 4: 'orange', 5: 'cadetblue', 6: 'orchid', 7: 'lime'}


# Plotting and saving figure, 3 dimensional
fig = pt.figure(figsize=(7, 7))
ax = pt.axes(projection="3d")
ax.scatter(cluster_data2['MAX_AGE'],
           cluster_data2['SPEED_ZONE'],
           cluster_data2['severity_index'],
           c=[colormap.get(x) for x in clusters.labels_], alpha=0.2)


ax.set_xlabel('MAX_AGE')
ax.set_ylabel('SPEED_ZONE')
ax.set_zlabel('severity_index')
ax.set_title(f"k = {len(set(clusters.labels_))}")

pt.savefig('hardardCluster.png')


# Clustering Data
clusters = KMeans(n_clusters=4)
clusters.fit(normalised_data)


# Colour map to assign different colour to each dot in scatterplot corresponding to its assigned cluser

# Plotting coloured scatter plot
pt.figure(figsize=(12, 10))
pt.scatter(cluster_data2['MAX_AGE'], cluster_data2['severity_index'],
           c=[colormap.get(x) for x in clusters.labels_], alpha=0.4)
pt.xlabel('MAX_AGE')
pt.ylabel('severity_index')
pt.title('Crashes based on manufacture year, brand make, and body style: Coloured Based on K Means clustering')
pt.savefig('task3_3_scattercolour.png')

# Outputting clusters to individual CSV files
cluster_data2 = cluster_data2[['RESTRAINT_WORN', 'SPEED_ZONE', 'MAX_AGE', 'SEX', 'LIGHT_CONDITION', 'severity_index', 'SEVERITY']]
outputClusters(clusters.labels_, ['MAX_AGE'], True, cluster_data2)


# To do:
"""
- Create a severity_index column
- Get all one varible data from the clusters and categories the clusters

"""
