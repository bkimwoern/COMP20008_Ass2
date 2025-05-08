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
    if pd.isna(age_interval):
        return 0

    try:
        if '-' in age_interval:
            return int(age_interval.split('-')[1])
        elif '+' in age_interval:
            return int(age_interval.replace('+', ''))
        else:
            return int(age_interval)  # fallback in case it's just a number
    except (ValueError, TypeError, AttributeError):
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
        dfCluster.to_csv('hazardOutput/cluster' + str(i) + '.csv', index=False)


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
        with open('hazardOutput/dataC' + str(i) + '.json', 'w') as f:
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
person_data = person_data[person_data['MAX_AGE'] != 0]

# Merge accident and person data by accident number
crash_data = pd.merge(accident_data, person_data, how='inner', on='ACCIDENT_NO')
crash_data.to_csv('newdata.csv', index=False)

# Filter down to attributes we want to cluster on
cluster_data = crash_data[['SPEED_ZONE', 'UNPROTECTED']]
# ['SPEED_ZONE', 'RESTRAINT_WORN'] with k = 3
# ['SPEED_ZONE', 'MAX_AGE', 'RESTRAINT_WORN'] with k = 3,4
# ['SPEED_ZONE', 'MAX_AGE']

# ['SPEED_ZONE', 'MAX_AGE', 'RESTRAINT_WORN'] k =3 when looking at SEVERITY ==1

# Copy data to normalise
normalised_data = cluster_data.copy(deep=True)
numeric_cols = normalised_data.select_dtypes(include='number').columns
normalised_data[numeric_cols] = (normalised_data[numeric_cols] - normalised_data[numeric_cols].min()) / (normalised_data[numeric_cols].max() - normalised_data[numeric_cols].min())

# Using elbow method to find best k value: The following code is from week 6 workshop
seed = 33 # this seed when using RESTRAINT_WORN
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
pt.savefig('ageHazardElbow.png')

print(normalised_data)

clusters = KMeans(n_clusters=4, random_state=seed)
clusters.fit(normalised_data)

colormap = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'cadetblue', 5: 'orchid', 6: 'lime'}


# Plotting and saving figure, 3 dimensional
fig = pt.figure(figsize=(7, 7))
ax = pt.axes(projection="3d")
ax.scatter(crash_data['MAX_AGE'],
           crash_data['SPEED_ZONE'],
           crash_data['severity_index'],
           c=[colormap.get(x) for x in clusters.labels_], alpha=0.2)


ax.set_xlabel('Age')
ax.set_ylabel('Speed Zone')
ax.set_zlabel('Severity Index')
ax.set_title(f"Clusters on Speed Zone and Protection worn; k = {len(set(clusters.labels_))}")

pt.savefig('hardardCluster.png')


# Outputting clusters to individual CSV files
crash_data = crash_data[['UNPROTECTED', 'SPEED_ZONE', 'MAX_AGE', 'SEX', 'LIGHT_CONDITION', 'severity_index', 'SEVERITY']]
outputClusters(clusters.labels_, ['MAX_AGE'], True, crash_data)


# To do:
"""
- Create a severity_index column
- Get all one varible data from the clusters and categories the clusters

"""
