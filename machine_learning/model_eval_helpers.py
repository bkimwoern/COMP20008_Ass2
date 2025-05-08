import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score


def print_model_accuracy(model, X_train, y_train, X_test, y_test):
    print("MODEL ACCURACY")
    training_accuracy = model.score(X_train, y_train)
    testing_accuracy = model.score(X_test, y_test)
    print('Accuracy of training set:', training_accuracy)
    print('Accuracy of testing set:', testing_accuracy)
    print('Accuracy difference:', testing_accuracy - training_accuracy, '\n')

def plot_best_depth(X_train, y_train, X_test, y_test):
    depths = range(1, 11)
    train_accuracies = []
    test_accuracies = []

    for depth in depths:
        model = DecisionTreeRegressor(criterion='squared_error', max_depth=depth)
        model.fit(X_train, y_train)
        train_accuracies.append(model.score(X_train, y_train))
        test_accuracies.append(model.score(X_test, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(depths, test_accuracies, marker='s', label='Testing Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.xticks(depths)
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, class_labels, plot=True):
    y_pred = model.predict(X_test)
    c_mtrx = confusion_matrix(y_test, y_pred, labels=class_labels)
    f1_score(y_test, y_pred, average="binary")
    recall_score(y_test, y_pred, average="binary")
    precision_score(y_test, y_pred, average="binary")
    disp = ConfusionMatrixDisplay(confusion_matrix=c_mtrx, display_labels=class_labels)
    disp.plot(cmap=plt.cm.BuGn)
    plt.title("Confusion Matrix")
    plt.show()


def plot_decision_tree(model, feature_names, class_labels):
    plt.figure(figsize=(20, 10))
    plot_tree(model,
              feature_names=feature_names,
              class_names=class_labels,
              filled=True)
    plt.title("Decision Tree")
    plt.show()

def print_feature_importances(model, X_columns):
    print("FEATURE IMPORTANCES (cumulative information gain per class):")
    importances = pd.Series(model.feature_importances_, index=X_columns).sort_values(ascending=False)
    for feature, importance in importances.items():
        print(f"{feature}: {importance:.4f}")


def print_stats(model, X_train_and_validation, y_train_and_validation, X_test, y_test, X_columns, class_labels):
    print('MAX_DEPTH:', model.max_depth, '\n')

    # print accuracy of training and testing data
    print_model_accuracy(model, X_train_and_validation, y_train_and_validation, X_test, y_test)

    # print the sorted (desc) order of feature importances (cumulative information gain?)
    print_feature_importances(model, X_columns)

    # plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test, model.classes_)


# def print_statistics(model, X_test, y_test, class_labels):

# baseline comparison for mention in report? use this to justify sampling smaller set same size as ... use this to
# baseline compared to non-fatal accident x% vs y%?
# discuss

# ensure testing set and training set has the same distribution of class categories (stratified cross validation)

# N-fold cross validation (cross validation)?

# distribution analysis (before modelling)

# up-sample > down-sample
def plot_mean_squared_errors(train, test, X_columns, y_train, y_test):
    for x in X_columns:
        X_train = train[x].values.reshape(-1, 1)  # reshape(-1, 1) is used to reshape the time_on_site column of the train dataset into a 2-dimensional array of size (n_samples, 1)
        X_test = test[x].values.reshape(-1, 1)
        model = DecisionTreeRegressor(criterion='mean_squared_error', max_depth=4)
        model.fit(X_train, y_train)  # LinearRegression()
        y_predict = model.predict(X_test)

        # Plot the scatter plot and the line of regression
        plt.scatter(X_test, y_test)
        plt.plot(X_test, y_predict, color='red')
        plt.title('Test set scatter plot')
        plt.xlabel(x)
        plt.ylabel('fatal')

        plt.show()


# hyperparameter tuning using stratified cross validation
def find_best_depth_cv(X, y, depths=range(1,11), k=5, top_k=15, print_metric_summary=True):
    skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    metrics_per_depth = {}
    metrics_summary = {}

    top_feature_counter = Counter()
    for depth in depths:
        metrics_per_depth[depth] = {"f1": [], "recall": [], "precision": []}


        for train_index, validation_index in skf.split(X, y):
            # determining indexes of training and validation set from the skf.get_N_splits(X, y)
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_validation = X.iloc[validation_index]
            y_validation = y.iloc[validation_index]

            mi = mutual_info_classif(X_train, y_train, discrete_features='auto', random_state=42)
            mi_series = pd.Series(mi, index=X_train.columns)
            top_features = mi_series.sort_values(ascending=False).head(top_k).index.tolist()
            top_feature_counter.update(top_features)

            X_train = X_train[top_features]
            X_validation = X_validation[top_features]

            # training the model on our training portion
            model = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
            model.fit(X_train, y_train)

            # predicting class label based on validation set
            y_prediction = model.predict(X_validation)

            # scoring
            metrics_per_depth[depth]["f1"].append(f1_score(y_validation, y_prediction, average='binary'))
            metrics_per_depth[depth]["recall"].append(recall_score(y_validation, y_prediction, average='binary'))
            metrics_per_depth[depth]["precision"].append(precision_score(y_validation, y_prediction, average='binary'))

    # for each depth, we are appending the average f1 score, recall and precision across each fold,
    # then append it to the summary
    for depth in metrics_per_depth:
        metrics_summary[depth] = {
            "average_f1": np.mean(metrics_per_depth[depth]["f1"]),
            "average_recall": np.mean(metrics_per_depth[depth]["recall"]),
            "average_precision": np.mean(metrics_per_depth[depth]["precision"])}

    # calculating max_depth hyprparameter based on the depth yielding highest F1 score
    best_depth = max(metrics_summary, key=lambda x: metrics_summary[x]["average_f1"])
    feature_selection = [feature for feature, _ in top_feature_counter.most_common(top_k)]

    # printing average evaluation metric for each detph, as well as the overall best depth based on F1 score.
    if print_metric_summary:
        print("\nAverage Evaluation Scores per Depth:")
        for depth in metrics_per_depth:
            f1 = metrics_summary[depth]["average_f1"]
            recall = metrics_summary[depth]["average_recall"]
            precision = metrics_summary[depth]["average_precision"]
            print(f"    Depth {depth}: F1={f1:3f}, Recall={recall:3f}, Precision={precision:3f}")

        print(f"\nBest Depth by F1 Score: {best_depth}")

        print("\nMost Frequently Selected Features By M.I, Across all folds (top_k=%d):" % top_k)
        for feature, count in top_feature_counter.most_common(top_k):
            print(f"{feature}: selected in {count} folds")

    plot_best_depth_cv(metrics_summary, depths)


    # return best depth and updated feature selection based on most frequently selected features yielding most M.I
    return best_depth, feature_selection

def plot_best_depth_cv(metrics_summary, depths):
    plt.figure(figsize=(10, 6))
    plt.plot(depths, [metrics_summary[d]["average_f1"] for d in depths], marker='o', label='Average F1 Score')
    plt.plot(depths, [metrics_summary[d]["average_recall"] for d in depths], marker='s', label='Average Recall score')
    plt.plot(depths, [metrics_summary[d]["average_precision"] for d in depths], marker='^',
             label='Average Precision score')
    plt.xlabel('Max Depth')
    plt.ylabel('Evaluation Scores')
    plt.title('Decision Tree Evaluation Metrics vs Max Depth')
    plt.xticks(depths)
    plt.legend()
    plt.show()