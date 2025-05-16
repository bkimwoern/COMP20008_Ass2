import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # requires seaborn[stats] installed
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import statsmodels.api as sm # requires statsmodels installed
from patsy import dmatrices 
from scipy.stats import entropy


def cramers_corrected_stat(col1, col2): # adjusted from wikipedia
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = pd.crosstab(col1, col2).to_numpy()
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def compute_entropy(series): # helper function for mutual info
        counts = series.value_counts()
        probs = counts / counts.sum()
        return entropy(probs, base=2)
    
def mutual_info(inputdata, dep_vars, ind_vars, normal='Regular'): # adjusted example from Geeksforgeeks.org
    """_summary_

    Args:
        inputdata (_type_): _description_
        dep_vars (_type_): _description_
        ind_vars (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = inputdata.dropna(subset=dep_vars+ind_vars).copy()
    mutual_info_matrix = np.zeros((len(ind_vars),len(dep_vars)))

    for var in dep_vars:
        data[var] = pd.factorize(data[var])[0]
        
    for var in ind_vars:
        data[var] = pd.factorize(data[var])[0]
    
    entropies = {var: compute_entropy(data[var]) for var in dep_vars + ind_vars}

    for i, var1 in enumerate(ind_vars):
        for j, var2 in enumerate(dep_vars):
            X = data[[var1]]
            y = data[var2]
            mutual_info = mutual_info_classif(X, y)
            h_x = entropies[var1]
            h_y = entropies[var2]
            if normal == 'Regular':
                mutual_info_matrix[i, j] = mutual_info[0] 
            else: 
                mutual_info_matrix[i, j] = mutual_info[0] / ((h_x + h_y)/2)

    mutual_info_df = pd.DataFrame(mutual_info_matrix, index=ind_vars, columns=dep_vars)
    return mutual_info_df

def compare_count_barplot(orgdata, subdata, col, order=None, ylim=100, title1="Overall", title2="Specific", stat='percent'):
    """ plots 2 side by side barplots for data comparison

    Args:
        data1 (dataframe): _description_
        data2 (dataframe): _description_
        col (string): _description_
        ylim (int, optional): _description_. Defaults to 100.
        title1 (str, optional): _description_. Defaults to "All".
        title2 (str, optional): _description_. Defaults to "Fatal".
        stat (str, optional): _description_. Defaults to 'percent'.
    """
    data1 = orgdata.dropna(subset=[col]).copy()
    data2 = subdata.dropna(subset=[col]).copy()
    axs = plt.subplot(1,2,1)
    sns.countplot(data=data1, x=col, stat=stat, ax=axs, order=order)
    axs.set_title(title1)
    axs.set_ylim(0, ylim)
    axs = plt.subplot(1,2,2)
    sns.countplot(data=data2, x=col, stat=stat, ax=axs, order=order)
    axs.set_title(title2)
    axs.set_ylim(0, ylim)
    
# read all relevant data csvs 
accident_df = pd.read_csv("./datasets/filtered_accident.csv") # length: 178695
#accident_df = accident_df.dropna() # length: 170051

person_df = pd.read_csv("./datasets/filtered_person.csv") # length: 417616
#person_df = person_df.dropna() # length: 101167

vehicle_df = pd.read_csv("./datasets/filtered_vehicle_new.csv") # length: 259455
#vehicle_df = vehicle_df.dropna() # length: 0

# computes ratio of unprotected:protected persons involved in a given accident
unprotected_ratio = person_df.groupby('ACCIDENT_NO')['UNPROTECTED'].mean().reset_index(name='UNPROTECTED_RATIO')
accident_df = accident_df.merge(unprotected_ratio, on='ACCIDENT_NO')

# computes ratio of persons killed / total persons involved in a given accident
accident_df['FATAL_RATIO'] = accident_df['NO_PERSONS_KILLED'] / accident_df['NO_PERSONS']

# computes if accident involve deaths
accident_df['IS_LETHAL'] = accident_df['NO_PERSONS_KILLED'].apply(lambda x: 1 if x > 0 else 0)

# computes if recorded person is dead
person_df['DEAD'] = person_df['INJ_LEVEL'].apply(lambda x: 1 if x == 1 else 0)

# sampling a subset of non-fatal accidents, so that the number of fatal and non-fatal class labels are even
# this is done to remove the bias of there being so many non-fatal records (175751) to fatal records (2944)
fatal_records = accident_df[accident_df['IS_LETHAL'] == 1]
non_fatal_entries = accident_df[accident_df['IS_LETHAL'] == 0]
non_fatal_entries = non_fatal_entries.sample(n=fatal_records.shape[0], random_state=42)
balanced_accident_df = pd.concat([fatal_records, non_fatal_entries])

# try for scatterplots with some columns with line of best fit
sns.lmplot(data=balanced_accident_df, x='NO_OF_VEHICLES', y='FATAL_RATIO')
#plt.savefig("./Correlation/Figures/numvehiclesfatalratioplot.png", dpi=300)
plt.show()

sns.lmplot(data=balanced_accident_df, x='UNPROTECTED_RATIO', y='FATAL_RATIO')
#plt.savefig("./Correlation/Figures/unprotectedfatalratioplot.png", dpi=300)
plt.show()

# prints ordinary least squares regression results of scatter plot
y, X = dmatrices('FATAL_RATIO ~ NO_OF_VEHICLES + UNPROTECTED_RATIO', data=balanced_accident_df, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())

# sampling for person_df
fatal_records = person_df[person_df['DEAD'] == 1]
non_fatal_entries = person_df[person_df['DEAD'] == 0]
non_fatal_entries = non_fatal_entries.sample(n=fatal_records.shape[0], random_state=42)
balanced_person_df = pd.concat([fatal_records, non_fatal_entries])
'''
# compute cramers v coefficient association 
accident_cramersv = {}
for col in accident_df:
    if col == 'IS_LETHAL' or col == 'ACCIDENT_NO' or col == 'NO_PERSONS_KILLED' or col == 'SEVERITY':
        continue
    else:
        accident_cramersv[col] = cramers_corrected_stat(accident_df['IS_LETHAL'], accident_df[col])
print('ACCIDENT')
for item in sorted(accident_cramersv.items(), key=lambda x: x[1], reverse=True):
    print(item)
person_cramersv = {}
for col in person_df:
    if col == 'DEAD' or col == 'ACCIDENT_NO' or col == 'INJ_LEVEL' or col == 'INJ_LEVEL_DESC':
        continue
    else:
        person_cramersv[col] = cramers_corrected_stat(person_df['DEAD'], person_df[col])
print("PERSON")
for item in sorted(person_cramersv.items(), key=lambda x: x[1], reverse=True):
    print(item)
'''
# create mutual information matrix of accident_df and represent in heat map
dep_vars = ['SEVERITY', 'IS_LETHAL', 'FATAL_RATIO']
ind_vars = ['AT_INTERSECTION', 'NO_OF_VEHICLES', 'ACCIDENT_TYPE', 'UNPROTECTED_RATIO', 'ROAD_GEOMETRY']

mutual_info_df = mutual_info(balanced_accident_df, dep_vars, ind_vars)
#print(mutual_info_df)
plt.figure(figsize=(10, 8))
sns.heatmap(mutual_info_df, annot=True, cmap='coolwarm', square=True)
plt.title('Pairwise Mutual Information')
plt.savefig("./Correlation/Figures/accidentMIheatmap.png", dpi=300)
plt.show()


dep_vars = ['DEAD', 'INJ_LEVEL']
ind_vars = ['SEX', 'AGE_GROUP', 'IN_METAL_BOX', 'ROAD_USER_TYPE', 'UNPROTECTED']

mutual_info_df = mutual_info(balanced_person_df, dep_vars, ind_vars)
#print(mutual_info_df)
plt.figure(figsize=(10, 8))
sns.heatmap(mutual_info_df, annot=True, cmap='coolwarm', square=True)
plt.title('Pairwise Mutual Information')
plt.savefig("./Correlation/Figures/personMIheatmap.png", dpi=300)
plt.show()

# create normalized mutual information matrix of accident_df and represent in heat map
dep_vars = ['SEVERITY', 'IS_LETHAL', 'FATAL_RATIO']
ind_vars = ['AT_INTERSECTION', 'NO_OF_VEHICLES', 'ACCIDENT_TYPE', 'UNPROTECTED_RATIO', 'ROAD_GEOMETRY']

mutual_info_df = mutual_info(balanced_accident_df, dep_vars, ind_vars, "Normal")
#print(mutual_info_df)
plt.figure(figsize=(10, 8))
sns.heatmap(mutual_info_df, annot=True, cmap='coolwarm', square=True)
plt.title('Pairwise Normalized Mutual Information')
plt.savefig("./Correlation/Figures/accidentNMIheatmap.png", dpi=300)
plt.show()


dep_vars = ['DEAD', 'INJ_LEVEL']
ind_vars = ['SEX', 'AGE_GROUP', 'IN_METAL_BOX', 'ROAD_USER_TYPE', 'UNPROTECTED']

mutual_info_df = mutual_info(balanced_person_df, dep_vars, ind_vars, "Normal")
#print(mutual_info_df)
plt.figure(figsize=(10, 8))
sns.heatmap(mutual_info_df, annot=True, cmap='coolwarm', square=True)
plt.title('Pairwise Normalized Mutual Information')
plt.savefig("./Correlation/Figures/personNMIheatmap.png", dpi=300)
plt.show()

''' # ignore
# create side by side barplots of overall data vs fatal data by column
compare_count_barplot(accident_df, fatal_accident_df, 'SPEED_ZONE', ylim=43, title2="Fatal")
plt.savefig("./Correlation/Figures/speedzonecompplot.png", dpi=300)
plt.show()

compare_count_barplot(accident_df, fatal_accident_df, 'DAY_WEEK_DESC', ylim=17, title2="Fatal", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.savefig("./Correlation/Figures/dayweekcompplot.png", dpi=300)
plt.show()

compare_count_barplot(accident_df, fatal_accident_df, 'ROAD_GEOMETRY', ylim=71, title2="Fatal")
plt.savefig("./Correlation/Figures/roadgeometrycompplot.png", dpi=300)
plt.show()

compare_count_barplot(accident_df, fatal_accident_df, 'ACCIDENT_TYPE', ylim=66, title2="Fatal")
plt.savefig("./Correlation/Figures/accidenttypecompplot.png", dpi=300)
plt.show()

compare_count_barplot(accident_df, fatal_accident_df, 'NO_OF_VEHICLES', ylim=60, title2="Fatal")
plt.savefig("./Correlation/Figures/numvehiclescompplot.png", dpi=300)
plt.show()

compare_count_barplot(person_df, fatal_person_df, 'AGE_GROUP', ylim=21, title2="Fatal", order=["13-15", "16-17", "18-21", "22-25", "26-29", "30-39", "40-49","50-59","60-64","65-69","70+"])
plt.savefig("./Correlation/Figures/agegroupcompplot.png", dpi=300)
plt.show()

compare_count_barplot(person_df, fatal_person_df, 'UNPROTECTED', ylim=70, title2="Fatal")
plt.savefig("./Correlation/Figures/unprotectedcompplot.png", dpi=300)
plt.show()
'''