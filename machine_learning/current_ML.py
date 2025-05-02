import os
import sys

import matplotlib.pyplot as plt


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from model_eval_helpers import (model_accuracy,
                                                                plot_confusion_matrix,
                                                                plot_decision_tree,
                                                                find_best_depth)


#features relevant to exploration: time, day of week, public holiday, date

def bucket(n):
    if n == 1:
        return "Low"
    elif n == 2:
        return "Moderate"
    else:
        return "High"

def fm_time_feature_set():
    # reading in csv's relevant to research exploration for time-correlated factors
    accident_df = pd.read_csv('../datasets/accident.csv')
    public_holidays_df = pd.read_csv('../datasets/public_holiday_2012-2024.csv')

    # extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%d/%m/%Y')
    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month
    accident_df['YEAR'] = accident_df['ACCIDENT_DATE'].dt.year
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour
    public_holidays_df['Date'] = pd.to_datetime(public_holidays_df['Date'], format='%d/%m/%Y')

    # adding a new column indicating whether an accident had occurred on a national public holiday
    national_holidays_series = public_holidays_df[public_holidays_df['National_holiday'] == True]
    accident_df['NATIONAL_HOLIDAY'] = accident_df['ACCIDENT_DATE'].isin(national_holidays_series['Date']).astype(int)

    # adding a new column for IS_LETHAL, indicating no fatalities (0), or at least one IS_LETHAL occurred (1)
    accident_df['IS_LETHAL'] = (accident_df['NO_PERSONS_KILLED'] > 0).astype(int)

    # sampling a subset of non-fatal accidents, so that the number of fatal and non-fatal class labels are even
    # this is done to remove the bias of there being so many non-fatal records (175751) to fatal records (2944)

    #shuffle the dataframe, and resets row index
    accident_df = accident_df.sample(frac=1, random_state=42).reset_index(drop=True)
    fatal_records = accident_df[accident_df['IS_LETHAL'] > 0]
    non_fatal_entries = accident_df[accident_df['IS_LETHAL'] == 0]
    non_fatal_entries = non_fatal_entries.sample(n=fatal_records.shape[0], random_state=42)
    balanced_df = pd.concat([fatal_records, non_fatal_entries])

    print('FATAL record\'s distribution of national holidays:')
    print(fatal_records['NATIONAL_HOLIDAY'].value_counts())
    print('\nNON-FATAL record\'s distribution of national holidays:')
    print(non_fatal_entries['NATIONAL_HOLIDAY'].value_counts(), '\n')

    # defining the scope of my feature set (DATE, TIME, DAY), and class label (IS_LETHAL)
    time_analysis_df = balanced_df[['YEAR',
                                    'MONTH',
                                    'HOUR',
                                    'DAY_OF_WEEK',
                                    'NATIONAL_HOLIDAY',
                                    'IS_LETHAL']]

    train, test = train_test_split(time_analysis_df, test_size=0.2, random_state=42)


    X_columns = ['YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'NATIONAL_HOLIDAY']
    y_column = 'IS_LETHAL'

    X_train = train[X_columns]
    y_train = train[y_column]
    X_test = test[X_columns]
    y_test = test[y_column]

    # appropriate depth determined to be 6, when applying "balanced" class_weight.
    find_best_depth(X_train, y_train, X_test, y_test)


    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred, target_names=["No IS_LETHAL", "IS_LETHAL"]))
    model_accuracy(model, X_train, y_train, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test, [1, 0])
    print(time_analysis_df.groupby('IS_LETHAL').mean(), '\n')


    importances = model.feature_importances_
    for feature, importance in zip(X_columns, importances):
        print(f"{feature}: {importance:.4f}")

# fatality model where IS_LETHAL (1, 0) is the class label
def fatality_model():
    # reading in csv's relevant to research exploration for time-correlated factors
    accident_df = pd.read_csv('../datasets/filtered_accident.csv')
    person_df = pd.read_csv('../datasets/filtered_person.csv')

    #removes speed beyond the range of 120km/h
    accident_df = accident_df[accident_df['SPEED_ZONE'] < 120]

    # encodes 1 if accident occurred at an intersection, 0 if not at an intersection
    accident_df['AT_INTERSECTION'] = accident_df['ROAD_GEOMETRY'].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)

    # encodes 0 if some form safety protection (helmet/seatbelt) is worn, 1 if none have been worn
    person_df = person_df.dropna(subset=['HELMET_BELT_WORN'])
    person_df['UNPROTECTED'] = person_df['HELMET_BELT_WORN'].apply(lambda x: 0 if x in [1, 3, 6] else 1)

    #computes ratio of unprotected:protected persons involved in a given accident
    unprotected_ratio = person_df.groupby('ACCIDENT_NO')['UNPROTECTED'].mean().reset_index(name = 'UNPROTECTED_RATIO')
    accident_df = accident_df.merge(unprotected_ratio, on='ACCIDENT_NO')

    # extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%Y-%m-%d')
    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour

    # adding a new column for IS_LETHAL, indicating no fatalities (0), or at least one IS_LETHAL occurred (1)
    accident_df['IS_LETHAL'] = (accident_df['NO_PERSONS_KILLED'] > 0).astype(int)

    #shuffle the dataframe, and resets row index
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
    #find_best_depth(X_train, y_train, X_test, y_test)

    model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    model.fit(X_train, y_train)

    #print accuracy of training and testing data
    model_accuracy(model, X_train, y_train, X_test, y_test)

    #print confusion matrix
    plot_confusion_matrix(model, X_test, y_test, [1, 0])

    importances = model.feature_importances_
    for feature, importance in zip(X_columns, importances):
        print(f"{feature}: {importance:.4f}")


# severity model where severity of accident (1, 2, 3) is the class label
def severity_model():
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
    unprotected_ratio = person_df.groupby('ACCIDENT_NO')['UNPROTECTED'].mean().reset_index(name='UNPROTECTED_RATIO')
    accident_df = accident_df.merge(unprotected_ratio, on='ACCIDENT_NO')

    # extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%Y-%m-%d')
    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour

    # shuffle the dataframe, and resets row index
    accident_df = accident_df.sample(frac=1, random_state=42).reset_index(drop=True)

    #drop outlier (severity == 4) where accidents occur with zero injury.
    accident_df = accident_df[accident_df['SEVERITY'] !=4]

    # determine the number of samples for each class type
    sample_size = accident_df['SEVERITY'].value_counts().min()

    # balance the sampling across all class types
    balanced_df = pd.concat([group.sample(n=sample_size, random_state=42)
                             for _, group in accident_df.groupby('SEVERITY')],
                            ignore_index=True)

    # defining the scope of my feature set, and class label (SEVERITY)
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
                               'SEVERITY']]
    train, test = train_test_split(analysis_df, test_size=0.2, random_state=42)

    X_columns = ['SPEED_ZONE',
                 'UNPROTECTED_RATIO',
                 'AT_INTERSECTION',
                 'NO_PERSONS',
                 'NO_OF_VEHICLES',
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

    # appropriate depth determined to be 4
    # find_best_depth(X_train, y_train, X_test, y_test)

    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model.fit(X_train, y_train)

    # print accuracy of training and testing data
    model_accuracy(model, X_train, y_train, X_test, y_test)

    # print confusion matrix
    plot_confusion_matrix(model, X_test, y_test, [1, 2, 3])

    #importances = model.feature_importances_
    #for feature, importance in zip(X_columns, importances):
    #    print(f"{feature}: {importance:.4f}")

def main():
    print('FATALITY MODEL\n-----------------------------------------------------------------------\n')
    fatality_model()
    print('SEVERITY MODEL\n-----------------------------------------------------------------------\n')
    severity_model()

if __name__ == "__main__":
    main()
