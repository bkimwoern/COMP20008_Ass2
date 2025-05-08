import os
import sys

import matplotlib.pyplot as plt


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model_eval_helpers import (print_stats,
                                plot_decision_tree,
                                plot_best_depth,
                                plot_mean_squared_errors)


# features relevant to exploration: time, day of week, public holiday, date

def bucket(n):
    if n == 1:
        return "Low"
    elif n == 2:
        return "Moderate"
    else:
        return "High"


def fm_time_feature_set():
    # reading in CSVs relevant to research exploration for time-correlated factors
    accident_df = pd.read_csv('../datasets/filtered_accident.csv')
    person_df = pd.read_csv('../datasets/filtered_person.csv')

    # extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%Y-%m-%d')
    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month
    accident_df['YEAR'] = accident_df['ACCIDENT_DATE'].dt.year
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour

    # adding a new column for IS_LETHAL, indicating no fatalities (0), or at least one IS_LETHAL occurred (1)
    accident_df['IS_LETHAL'] = (accident_df['NO_PERSONS_KILLED'] > 0).astype(int)

    # computes ratio of unprotected:protected persons involved in a given accident
    unprotected_ratio = person_df.groupby('ACCIDENT_NO')['UNPROTECTED'].mean().reset_index(name='UNPROTECTED_RATIO')
    accident_df = accident_df.merge(unprotected_ratio, on='ACCIDENT_NO')

    # shuffle the dataframe, and resets row index
    accident_df = accident_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # sampling a subset of non-fatal accidents, so that the number of fatal and non-fatal class labels are even
    # this is done to remove the bias of there being so many non-fatal records (175751) to fatal records (2944)
    fatal_records = accident_df[accident_df['IS_LETHAL'] > 0]
    non_fatal_entries = accident_df[accident_df['IS_LETHAL'] == 0]
    non_fatal_entries = non_fatal_entries.sample(n=fatal_records.shape[0], random_state=42)
    balanced_df = pd.concat([fatal_records, non_fatal_entries])

    # defining the scope of my feature set (DATE, TIME, DAY), and class label (IS_LETHAL)
    time_analysis_df = balanced_df[['SPEED_ZONE',
                                    'HOUR',
                                    'ACCIDENT_TYPE',
                                    'UNPROTECTED_RATIO',
                                    'NO_OF_VEHICLES',
                                    'NO_PERSONS',
                                    'IS_LETHAL']]

    train, test = train_test_split(time_analysis_df, test_size=0.2, random_state=42)

    X_columns = ['SPEED_ZONE',
                 'HOUR',
                 'ACCIDENT_TYPE',
                 'UNPROTECTED_RATIO',
                 'NO_OF_VEHICLES',
                 'NO_PERSONS',]
    y_column = 'IS_LETHAL'

    X_train = train[X_columns]
    y_train = train[y_column]
    X_test = test[X_columns]
    y_test = test[y_column]

    # appropriate depth determined to be ?
    plot_best_depth(X_train, y_train, X_test, y_test)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    model.fit(X_train, y_train)

    print('MAX_DEPTH:', model.max_depth, '\n')

    print_stats(model, X_train, y_train, X_test, y_test, X_columns, model.classes_)

# fatality model where IS_LETHAL (1, 0) is the class label
def fatality_model_dt():
    # reading in csv's relevant to research exploration for time-correlated factors
    accident_df = pd.read_csv('../datasets/filtered_accident.csv')
    person_df = pd.read_csv('../datasets/filtered_person.csv')

    # removes speed beyond the range of 120km/h
    accident_df = accident_df[accident_df['SPEED_ZONE'] < 120]

    #computes ratio of unprotected:protected persons involved in a given accident
    # encodes 1 if accident occurred at an intersection, 0 if not at an intersection
    accident_df['AT_INTERSECTION'] = accident_df['ROAD_GEOMETRY'].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)

    # computes ratio of unprotected:protected persons involved in a given accident
    unprotected_ratio = person_df.groupby('ACCIDENT_NO')['UNPROTECTED'].mean().reset_index(name = 'UNPROTECTED_RATIO')
    accident_df = accident_df.merge(unprotected_ratio, on='ACCIDENT_NO')

    # computes the proportion of each person's 'AGE_GROUP'(ing) per accident.
    # one hot encode 'AGE_GROUP' while marking it with it's associated 'ACCIDENT_NO'
    age_group_proportion = pd.get_dummies(person_df['AGE_GROUP'], prefix="AGE_GROUP")
    age_group_proportion['ACCIDENT_NO'] = person_df['ACCIDENT_NO']

    # grouping by 'ACCIDENT_NO' and finding the proportion of each age group, merging with rest of dataframe
    age_group_proportion = age_group_proportion.groupby('ACCIDENT_NO').mean().reset_index()
    accident_df = accident_df.merge(age_group_proportion, on='ACCIDENT_NO', how='left')

    # computes the proportion of sex involved within each accident (M, F, UK)
    sex_proportion = pd.get_dummies(person_df['SEX'], dtype=int, prefix="SEX")
    sex_proportion['ACCIDENT_NO'] = person_df['ACCIDENT_NO']

    # replacing 'SEX' column attributes with numerical counterparts
    sex_proportion = sex_proportion.groupby('ACCIDENT_NO').mean().reset_index()
    accident_df = accident_df.merge(sex_proportion, on='ACCIDENT_NO', how='left')

    # extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%Y-%m-%d')
    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour

    # adding a new column for IS_LETHAL, indicating no fatalities (0), or at least one IS_LETHAL occurred (1)
    accident_df['IS_LETHAL'] = (accident_df['NO_PERSONS_KILLED'] > 0).astype(int)

    # one hot encoding ACCIDENT_TYPE
    accident_df = pd.get_dummies(accident_df, columns=['ACCIDENT_TYPE'], drop_first=False)

    accident_df = pd.get_dummies(accident_df, columns=['ROAD_GEOMETRY'], drop_first=False)

    pd.set_option('display.max_rows', None)

    #shuffle the dataframe, and resets row index
    # shuffle the dataframe, and resets row index
    accident_df = accident_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # sampling a subset of non-fatal accidents, so that the number of fatal and non-fatal class labels are even
    # this is done to remove the bias of there being so many non-fatal records (175751) to fatal records (2944)
    fatal_records = accident_df[accident_df['IS_LETHAL'] > 0]
    non_fatal_entries = accident_df[accident_df['IS_LETHAL'] == 0]
    non_fatal_entries = non_fatal_entries.sample(n=fatal_records.shape[0], random_state=42)
    balanced_df = pd.concat([fatal_records, non_fatal_entries])

    # defining the scope of my feature set (DATE, TIME, DAY), and class label (IS_LETHAL)
    analysis_df = balanced_df[['SPEED_ZONE',
                               'NO_OF_VEHICLES',
                               'NO_PERSONS',
                               'UNPROTECTED_RATIO',
                               'LIGHT_CONDITION',
                               'DAY',
                               'MONTH',
                               'DAY_OF_WEEK',
                               'HOUR',
                               'PUBLIC_HOLIDAY',
                               'ACCIDENT_TYPE_1',
                               'ACCIDENT_TYPE_2',
                               'ACCIDENT_TYPE_3',
                               'ACCIDENT_TYPE_4',
                               'ACCIDENT_TYPE_5',
                               'ACCIDENT_TYPE_6',
                               'ACCIDENT_TYPE_7',
                               'ACCIDENT_TYPE_8',
                               'ACCIDENT_TYPE_9',
                               'ROAD_GEOMETRY_1',
                               'ROAD_GEOMETRY_2',
                               'ROAD_GEOMETRY_3',
                               'ROAD_GEOMETRY_4',
                               'ROAD_GEOMETRY_5',
                               'ROAD_GEOMETRY_6',
                               'ROAD_GEOMETRY_7',
                               'ROAD_GEOMETRY_8',
                               'ROAD_GEOMETRY_9',
                               'AGE_GROUP_13-15',
                               'AGE_GROUP_16-17',
                               'AGE_GROUP_18-21',
                               'AGE_GROUP_22-25',
                               'AGE_GROUP_26-29',
                               'AGE_GROUP_30-39',
                               'AGE_GROUP_40-49',
                               'AGE_GROUP_50-59',
                               'AGE_GROUP_60-64',
                               'AGE_GROUP_65-69',
                               'AGE_GROUP_70+',
                               'SEX_F',
                               'SEX_M',
                               'SEX_U',
                               'IS_LETHAL']]
    train, test = train_test_split(analysis_df, test_size=0.2, random_state=42)

    X_columns =['SPEED_ZONE',
                'NO_OF_VEHICLES',
                'NO_PERSONS',
                'UNPROTECTED_RATIO',
                'LIGHT_CONDITION',
                'DAY',
                'MONTH',
                'DAY_OF_WEEK',
                'HOUR',
                'PUBLIC_HOLIDAY',
                'ACCIDENT_TYPE_1',
                'ACCIDENT_TYPE_2',
                'ACCIDENT_TYPE_3',
                'ACCIDENT_TYPE_4',
                'ACCIDENT_TYPE_5',
                'ACCIDENT_TYPE_6',
                'ACCIDENT_TYPE_7',
                'ACCIDENT_TYPE_8',
                'ACCIDENT_TYPE_9',
                'ROAD_GEOMETRY_1',
                'ROAD_GEOMETRY_2',
                'ROAD_GEOMETRY_3',
                'ROAD_GEOMETRY_4',
                'ROAD_GEOMETRY_5',
                'ROAD_GEOMETRY_6',
                'ROAD_GEOMETRY_7',
                'ROAD_GEOMETRY_8',
                'ROAD_GEOMETRY_9',
                'AGE_GROUP_13-15',
                'AGE_GROUP_16-17',
                'AGE_GROUP_18-21',
                'AGE_GROUP_22-25',
                'AGE_GROUP_26-29',
                'AGE_GROUP_30-39',
                'AGE_GROUP_40-49',
                'AGE_GROUP_50-59',
                'AGE_GROUP_60-64',
                'AGE_GROUP_65-69',
                'AGE_GROUP_70+',
                'SEX_F',
                'SEX_M',
                'SEX_U']
    y_column = 'IS_LETHAL'

    X_train = train[X_columns]
    y_train = train[y_column]
    X_test = test[X_columns]
    y_test = test[y_column]

    # appropriate depth determined to be ?
    plot_best_depth(X_train, y_train, X_test, y_test)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=10)
    model.fit(X_train, y_train)

    # print stats of the model
    print_stats(model, X_train, y_train, X_test, y_test, X_columns, [1, 0])

def fatality_model_rg():
    # reading in csv's relevant to research exploration for time-correlated factors
    accident_df = pd.read_csv('../datasets/filtered_accident.csv')
    person_df = pd.read_csv('../datasets/filtered_person.csv')

    # removes speed beyond the range of 120km/h
    accident_df = accident_df[accident_df['SPEED_ZONE'] < 120]

    # encodes 1 if accident occurred at an intersection, 0 if not at an intersection
    accident_df['AT_INTERSECTION'] = accident_df['ROAD_GEOMETRY'].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)

    # encodes 0 if some form safety protection (helmet/seatbelt) is worn, 1 if none have been worn
    person_df = person_df.dropna(subset=['HELMET_BELT_WORN'])
    person_df['UNPROTECTED'] = person_df['HELMET_BELT_WORN'].apply(lambda x: 0 if x in [1, 3, 6] else 1)

    # computes ratio of unprotected:protected persons involved in a given accident
    unprotected_ratio = person_df.groupby('ACCIDENT_NO')['UNPROTECTED'].mean().reset_index(name = 'UNPROTECTED_RATIO')
    accident_df = accident_df.merge(unprotected_ratio, on='ACCIDENT_NO')

    # extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%Y-%m-%d')
    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour

    # adding a new column for IS_LETHAL, indicating no fatalities (0), or at least one IS_LETHAL occurred (1)
    accident_df['IS_LETHAL'] = (accident_df['NO_PERSONS_KILLED'] > 0).astype(int)

    # shuffle the dataframe, and resets row index
    accident_df = accident_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # sampling a subset of non-fatal accidents, so that the number of fatal and non-fatal class labels are even
    # this is done to remove the bias of there being so many non-fatal records (175751) to fatal records (2944)
    fatal_records = accident_df[accident_df['IS_LETHAL'] > 0]
    non_fatal_entries = accident_df[accident_df['IS_LETHAL'] == 0]
    non_fatal_entries = non_fatal_entries.sample(n=fatal_records.shape[0], random_state=42)
    balanced_df = pd.concat([fatal_records, non_fatal_entries])

    # defining the scope of my feature set (DATE, TIME, DAY), and class label (IS_LETHAL)
    analysis_df = balanced_df[['SPEED_ZONE',
                               'UNPROTECTED_RATIO',
                               'AT_INTERSECTION',
                               'NO_PERSONS',
                               'NO_OF_VEHICLES',
                               'DAY',
                               'MONTH',
                               'DAY_OF_WEEK',
                               'HOUR',
                               'PUBLIC_HOLIDAY',
                               'IS_LETHAL']]
    train, test = train_test_split(analysis_df, test_size=0.2, random_state=42)

    X_columns =['SPEED_ZONE',
                'UNPROTECTED_RATIO',
                'AT_INTERSECTION',
                'NO_PERSONS',
                'NO_OF_VEHICLES',
                'DAY',
                'MONTH',
                'DAY_OF_WEEK',
                'HOUR',
                'PUBLIC_HOLIDAY']
    y_column = 'IS_LETHAL'

    X_train = train[X_columns]
    y_train = train[y_column]
    X_test = test[X_columns]
    y_test = test[y_column]

    # appropriate depth determined to be 5
    plot_mean_squared_errors(X_columns, y_train, y_test)

    model = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
    model.fit(X_train, y_train)

    print_stats(model, X_train, y_train, X_test, y_test, X_columns, y_column)
    print('\n')


# severity model where severity of accident (1, 2, 3) is the class label
def severity_model():
    # reading in CSVs relevant to research exploration for time-correlated factors
    accident_df = pd.read_csv('../datasets/filtered_accident.csv')
    person_df = pd.read_csv('../datasets/filtered_person.csv')

    # removes speed beyond the range of 120km/h
    accident_df = accident_df[accident_df['SPEED_ZONE'] < 120]

    # encodes 1 if accident occurred at an intersection, 0 if not at an intersection
    accident_df['AT_INTERSECTION'] = accident_df['ROAD_GEOMETRY'].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)

    # encodes 0 if some form safety protection (helmet/seatbelt) is worn, 1 if none have been worn
    # person_df = person_df.dropna(subset=['HELMET_BELT_WORN'])
    # person_df['UNPROTECTED'] = person_df['HELMET_BELT_WORN'].apply(lambda x: 0 if x in [1, 3, 6] else 1)

    # computes ratio of unprotected:protected persons involved in a given accident
    unprotected_ratio = person_df.groupby('ACCIDENT_NO')['UNPROTECTED'].mean().reset_index(name='UNPROTECTED_RATIO')
    accident_df = accident_df.merge(unprotected_ratio, on='ACCIDENT_NO')

    # print("UNPROTECTED_RATIO AFTER UNKNOWN IMPUTATIONS:")



    # extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%Y-%m-%d')
    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour

    # adding a new column for IS_LETHAL, indicating no fatalities (0), or at least one IS_LETHAL occurred (1)
    accident_df['IS_LETHAL'] = (accident_df['NO_PERSONS_KILLED'] > 0).astype(int)

    # shuffle the dataframe, and resets row index
    accident_df = accident_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # sampling a subset of non-fatal accidents, so that the number of fatal and non-fatal class labels are even
    # this is done to remove the bias of there being so many non-fatal records (175751) to fatal records (2944)
    fatal_records = accident_df[accident_df['IS_LETHAL'] > 0]
    non_fatal_entries = accident_df[accident_df['IS_LETHAL'] == 0]
    non_fatal_entries = non_fatal_entries.sample(n=fatal_records.shape[0], random_state=42)
    balanced_df = pd.concat([fatal_records, non_fatal_entries])

    # defining the scope of my feature set (DATE, TIME, DAY), and class label (IS_LETHAL)
    analysis_df = balanced_df[['SPEED_ZONE',
                               'NO_OF_VEHICLES',
                               'AT_INTERSECTION',
                               'NO_PERSONS',
                               'UNPROTECTED_RATIO',
                               'DAY',
                               'MONTH',
                               'DAY_OF_WEEK',
                               'HOUR',
                               'PUBLIC_HOLIDAY',
                               'IS_LETHAL']]
    train, test = train_test_split(analysis_df, test_size=0.2, random_state=42)

    X_columns = ['SPEED_ZONE',
                 'NO_OF_VEHICLES',
                 'AT_INTERSECTION',
                 'NO_PERSONS',
                 'UNPROTECTED_RATIO',
                 'DAY',
                 'MONTH',
                 'DAY_OF_WEEK',
                 'HOUR',
                 'PUBLIC_HOLIDAY']

    y_column = 'SEVERITY'

    X_train = train[X_columns]
    y_train = train[y_column]
    X_test = test[X_columns]
    y_test = test[y_column]

    # appropriate depth determined to be ?
    plot_best_depth(X_train, y_train, X_test, y_test)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=8)
    model.fit(X_train, y_train)

    # print stats of the model
    print_stats(model, X_train, y_train, X_test, y_test, X_columns, model.classes_)


def main():
    print('CLASSIFIER FATALITY MODEL\n-----------------------------------------------------------------------\n')
    fatality_model_dt()
    # print('SEVERITY MODEL\n-----------------------------------------------------------------------\n')
    # severity_model()
    print('REGRESSOR FATALITY MODEL\n-----------------------------------------------------------------------\n')
    fatality_model_rg()

if __name__ == "__main__":
    main()
