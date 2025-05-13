import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from contextlib import redirect_stdout

# from machine_learning.model_eval_helpers import evaluate_decision_tree, evaluate_random_forest, print_stats
from model_eval_helpers import (evaluate_decision_tree, evaluate_random_forest, print_stats)


def create_model_dataframe():
    """ Function applies final pre-processing for specific implementation into machine learning models,
        additionally defines the feature suite used by both models. """

    # reading in CSVs relevant to research exploration for time-correlated factors
    accident_df = pd.read_csv('datasets/filtered_accident_no_nan.csv')
    person_df = pd.read_csv('datasets/filtered_person_no_nan.csv')

    # removes speed beyond the range of 120km/h
    accident_df = accident_df[accident_df['SPEED_ZONE'] < 120]

    # computes ratio of unprotected:protected persons involved in a given accident
    # encodes 1 if accident occurred at an intersection, 0 if not at an intersection
    accident_df['AT_INTERSECTION'] = accident_df['ROAD_GEOMETRY'].apply(lambda x: 1 if x in [1, 2, 3, 4] else 0)

    # computes ratio of unprotected:protected persons involved in a given accident
    unprotected_ratio = person_df.groupby('ACCIDENT_NO')['UNPROTECTED'].mean().reset_index(name='UNPROTECTED_RATIO')
    accident_df = accident_df.merge(unprotected_ratio, on='ACCIDENT_NO')

    # computes the proportion of each person's 'AGE_GROUP'(ing) per accident.
    # one hot encode 'AGE_GROUP' while marking it with it's associated 'ACCIDENT_NO'
    age_group_proportion = pd.get_dummies(person_df['AGE_GROUP'], dtype=int, prefix="AGE_GROUP:", prefix_sep=' ')
    age_group_proportion['ACCIDENT_NO'] = person_df['ACCIDENT_NO']

    # grouping by 'ACCIDENT_NO' and finding the proportion of each age group, merging with rest of dataframe
    age_group_proportion = age_group_proportion.groupby('ACCIDENT_NO').mean().reset_index()
    accident_df = accident_df.merge(age_group_proportion, on='ACCIDENT_NO', how='left')

    # computes the proportion of sex involved within each accident (M, F, UK)
    sex_proportion = pd.get_dummies(person_df['SEX'], dtype=int, prefix='SEX:', prefix_sep=' ')
    sex_proportion['ACCIDENT_NO'] = person_df['ACCIDENT_NO']

    # replacing 'SEX' column attributes with numerical counterparts
    sex_proportion = sex_proportion.groupby('ACCIDENT_NO').mean().reset_index()
    accident_df = accident_df.merge(sex_proportion, on='ACCIDENT_NO', how='left')

    # one hot encoding DAY_OF_WEEK (cyclic, but still categorical and will work better for tree)
    accident_df = pd.get_dummies(accident_df, columns=['DAY_WEEK_DESC'], prefix='DAY:', prefix_sep=' ')

    # one hot encoding ACCIDENT_TYPE
    accident_df = pd.get_dummies(accident_df, columns=['ACCIDENT_TYPE_DESC'], prefix='TYPE:', prefix_sep=' ')

    # one hot encoding ROAD_GEOMETRY
    accident_df = pd.get_dummies(accident_df, columns=['ROAD_GEOMETRY_DESC'], prefix='GEOM:', prefix_sep=' ')

    # extracting the month, year, and hour for compatibility with DecisionTreeClassifier
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'], format='%Y-%m-%d')
    accident_df['MONTH'] = accident_df['ACCIDENT_DATE'].dt.month

    #  due to the continuous and cyclic nature of hours within a day, using circular sine and cosine functions
    # preserves continuity across hours, such that 23:00 and 00:00 are treated as close together in time.
    accident_df['HOUR'] = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S').dt.hour
    accident_df['HOUR_SIN'] = np.sin(2 * np.pi * accident_df['HOUR'] / 24)
    accident_df['HOUR_COS'] = np.cos(2 * np.pi * accident_df['HOUR'] / 24)

    # adding a new column for IS_LETHAL, indicating no fatalities (0), or at least one IS_LETHAL occurred (1)
    accident_df['IS_LETHAL'] = (accident_df['NO_PERSONS_KILLED'] > 0).astype(int)

    # shuffle the dataframe, and resets row index
    accident_df = accident_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # sampling a subset of non-fatal accidents, so that the number of fatal and non-fatal class labels are even
    # this is done to remove the bias of there being so many non-fatal records (175751) to fatal records (2944)
    fatal_records = accident_df[accident_df['IS_LETHAL'] > 0]
    non_fatal_records = accident_df[accident_df['IS_LETHAL'] == 0]
    non_fatal_records = non_fatal_records.sample(n=fatal_records.shape[0], random_state=42)
    balanced_df = pd.concat([fatal_records, non_fatal_records])

    # pd.set_option('display.max_rows', None)

    # ---{ defining the scope of feature suite, defining class label }--- #
    X_columns = [
        'SPEED_ZONE',
        'NO_OF_VEHICLES',
        'NO_PERSONS',
        'UNPROTECTED_RATIO',
        'LIGHT_CONDITION',
        'MONTH',
        'PUBLIC_HOLIDAY',
        'TYPE: Collision with vehicle', 'TYPE: Struck Pedestrian', 'TYPE: Struck animal',
        'TYPE: Collision with a fixed object', 'TYPE: collision with some other object',
        'TYPE: Vehicle overturned (no collision)', 'TYPE: Fall from or in moving vehicle',
        'TYPE: No collision and no object struck', 'TYPE: Other accident',
        'GEOM: Cross intersection', 'GEOM: T intersection', 'GEOM: Y intersection',
        'GEOM: Multiple intersection', 'GEOM: Not at intersection', 'GEOM: Dead end', 'GEOM: Road closure',
        'GEOM: Private property', 'GEOM: Unknown',
        'AGE_GROUP: 13-15', 'AGE_GROUP: 16-17', 'AGE_GROUP: 18-21', 'AGE_GROUP: 22-25', 'AGE_GROUP: 26-29',
        'AGE_GROUP: 30-39', 'AGE_GROUP: 40-49', 'AGE_GROUP: 50-59', 'AGE_GROUP: 60-64', 'AGE_GROUP: 65-69',
        'AGE_GROUP: 70+',
        'SEX: F', 'SEX: M', 'SEX: U',
        'HOUR_SIN', 'HOUR_COS',
        'DAY: Sunday', 'DAY: Monday', 'DAY: Tuesday', 'DAY: Wednesday', 'DAY: Thursday', 'DAY: Friday',
        'DAY: Saturday'
    ]
    y_column = 'IS_LETHAL'

    # return final balanced data_frame with curated feature_suite
    return balanced_df[X_columns + [y_column]], X_columns, y_column


def model_dt(max_depth):
    """ Function that defines the 'static' hyperparameters for the Decision Tree Classifier model.

        When evaluating the Decision Tree Model, the best depth will be determined dynamically based on the
        criterion for each split being based on entropy. """

    return DecisionTreeClassifier(
        criterion='entropy',  # split based on entropy criterion
        max_depth=max_depth  # dynamically determine the best depth from the depths in range(1, 11)
    )


def decision_tree_model():
    """ Decision Tree Classifier Model """

    # processed feature suite for specialised use within machine learning models
    analysis_df, X_columns, y_column = create_model_dataframe()

    # split data into training set (%60), validation set(20%) and testing set (20%)
    # stratifying to ensure even proportion of class types (lethal 1, non-lethal 0)
    train_validate, test = train_test_split(
        analysis_df,
        test_size=0.2,
        random_state=42,
        stratify=analysis_df['IS_LETHAL']
    )

    # ---{ separating each set by feature/X_columns and the class label/y_column }--- #
    X_train_validate = train_validate[X_columns]
    y_train_validate = train_validate[y_column]
    X_test = test[X_columns]
    y_test = test[y_column]

    # find the best depth using cross validation, prints out the top 10 most selected features based on MI
    best_depth, selected_features = evaluate_decision_tree(
        lambda max_depth: model_dt(max_depth),
        X_train_validate,
        y_train_validate
    )

    # refining the scope of feature suite to selected features
    X_train_validate = X_train_validate[selected_features]
    X_test = X_test[selected_features]

    # appropriate depth determined to be 5
    final_model = model_dt(max_depth=best_depth)
    final_model.fit(X_train_validate, y_train_validate)

    # print stats of the model
    print_stats(
        final_model,
        X_train_validate,
        y_train_validate,
        X_test, y_test,
        selected_features,
        [1, 0],
    )


def model_rf(max_depth, n_estimators):
    """ Function that defines static hyperparameters for the Random Forest Classifier model.

        n_estimators and depth is determined dynamically through the best F1 generated by a combination of the two. """

    return RandomForestClassifier(
        n_estimators=n_estimators,  # dynamically determine best n_estimator from [20, 40, 60, 80, 100]
        max_depth=max_depth,  # dynamically determine the best depth within range(5, 21)
        criterion='entropy',  # same criterion as other model, for same baseline of comparison
        bootstrap=True,  # ensures each sample pulled from dataset is drawn randomly with replacement
        min_samples_leaf=5,  # ...
        max_features='sqrt',  # the number of features used per tree
        random_state=42  # for reproducible results
    )


def random_forest_tree_model():
    """Random Forest Classifier Model"""

    # same feature suite as other model
    analysis_df, X_columns, y_column = create_model_dataframe()

    # first split separates the testing set (20%) from training and validation (80%)
    train_and_validate, test = train_test_split(
        analysis_df,
        test_size=0.20,
        random_state=42,
        stratify=analysis_df['IS_LETHAL']
    )

    # following split separates the validation set (20%), bootstrap sampling is used in Random Forest Classifier model.
    # stratify to ensure even proportion of class types (lethal 1, non-lethal 0)
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.2,
        random_state=42,
        stratify=train_and_validate['IS_LETHAL']
    )

    # ---{ separating each set by feature/X_columns and the class label/y_column }--- #
    X_train = train[X_columns]
    y_train = train[y_column]

    X_validate = validate[X_columns]
    y_validate = validate[y_column]

    X_test = test[X_columns]
    y_test = test[y_column]

    # finding the best_depth, best_n and selected features from evaluation of random forest function (evaluate_rf)
    best_depth, best_n, selected_features = evaluate_random_forest(
        lambda max_depth, n_estimators: model_rf(max_depth, n_estimators),
        X_train,
        y_train,
        X_validate,
        y_validate,
        n_estimators=[20, 40, 60, 80, 100],
        depths=range(5, 21),
    )

    # merging the training and validation sets as these will be used to train the final model
    X_train_and_validate = pd.concat([X_train, X_validate])
    y_train_and_validate = pd.concat([y_train, y_validate])

    # defining scope of feature suite to the selected features
    X_train_and_validate = X_train_and_validate[selected_features]
    X_test = X_test[selected_features]

    # final training and fitting of the Random Forest Classifier Model
    final_model = model_rf(best_depth, best_n)
    final_model.fit(X_train_and_validate, y_train_and_validate)

    # evaluation statistics of the final Random Forest Classifier Model's performance
    print_stats(
        final_model,
        X_train_and_validate,
        y_train_and_validate,
        X_test, y_test,
        selected_features,
        [1, 0],
        rf=True
    )


def main():
    with open("machine_learning/evaluations.txt", "w") as f, redirect_stdout(f):
        print("\n::::------------------[ MODEL: DECISION TREE CLASSIFIER ]------------------::::\n")
        decision_tree_model()
        print("\n::::---------------[ MODEL: RANDOM FOREST TREE CLASSIFIER  ]---------------::::\n")
        random_forest_tree_model()


if __name__ == "__main__":
    main()
