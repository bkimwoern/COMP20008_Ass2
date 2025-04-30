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



def main():
    #reading in csv's relevant to research exploration for time-correlated factors
    accident_df = pd.read_csv('../datasets/accident.csv')
    public_holidays_df = pd.read_csv('../datasets/public_holiday_2012-2024.csv')

    #extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%d/%m/%Y')

    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month
    accident_df['YEAR'] = accident_df['ACCIDENT_DATE'].dt.year
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour

    public_holidays_df['Date'] = pd.to_datetime(public_holidays_df['Date'], format='%d/%m/%Y')

    #adding a new column indicating whether an accident had occurred on a national public holiday
    national_holidays_series = public_holidays_df[public_holidays_df['National_holiday'] == True]
    accident_df['NATIONAL_HOLIDAY'] = accident_df['ACCIDENT_DATE'].isin(national_holidays_series['Date']).astype(int)

    #adding a new column for FATALITY_STATUS, indicating no fatalities (0), or at least one fatality occurred (1)
    accident_df['FATALITY_STAT'] = (accident_df['NO_PERSONS_KILLED'] > 0).astype(int)

    #defining the scope of my feature set (DATE, TIME, DAY), and class label (FATALITY)
    time_analysis_df = accident_df[['YEAR',
                                    'MONTH',
                                    'HOUR',
                                    'DAY_OF_WEEK',
                                    'NATIONAL_HOLIDAY',
                                    'FATALITY_STAT']]


    train, test = train_test_split(time_analysis_df, test_size=0.2,  random_state=42, shuffle=True)

    X_columns = ['YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR', 'NATIONAL_HOLIDAY']
    y_column = 'FATALITY_STAT'

    X_train = train[X_columns]
    y_train = train[y_column]
    X_test = test[X_columns]
    y_test = test[y_column]

    # appropriate depth determined to be 6, when applying "balanced" class_weight.
    find_best_depth(X_train, y_train, X_test, y_test)

    model = DecisionTreeClassifier(criterion='entropy', max_depth=6, class_weight='balanced')
    model.fit(X_train, y_train)



    #y_pred = model.predict(X_test)
    #print(classification_report(y_test, y_pred, target_names=["No Fatality", "Fatality"]))

    model_accuracy(model, X_train, y_train, X_test, y_test)

    plot_confusion_matrix(model, X_test, y_test, [1, 0])

    print(time_analysis_df.groupby('FATALITY_STAT').mean())

    #decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=)

if __name__ == "__main__":
    main()
