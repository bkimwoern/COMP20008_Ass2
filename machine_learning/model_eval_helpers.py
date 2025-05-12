import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score


def print_model_accuracy(model, X_train, y_train, X_test, y_test):
    print("---------------------------[:: MODEL ACCURACIES :: ]---------------------------")
    training_accuracy = model.score(X_train, y_train)
    testing_accuracy = model.score(X_test, y_test)
    print("---> Accuracy of training set....:", training_accuracy)
    print("---> Accuracy of testing set.....:", testing_accuracy)
    print("---> Accuracy difference.........:", training_accuracy - testing_accuracy, '\n')


def plot_confusion_matrix(model, X_test, y_test, class_labels, rf=False):

    y_pred = model.predict(X_test)
    c_mtrx = confusion_matrix(y_test, y_pred, labels=class_labels)
    recall_score(y_test, y_pred, average="binary")
    precision_score(y_test, y_pred, average="binary")

    # ---{
    print("\n------------------------[:: FINAL EVALUATION SCORES ::]------------------------")
    print("---> F1 SCORE.............:", f1_score(y_test, y_pred, average="binary"))
    print("---> RECALL SCORE.........:", recall_score(y_test, y_pred, average="binary"))
    print("---> PRECISION SCORE......:", precision_score(y_test, y_pred, average="binary"))
    print("---> CONFUSION MATRIX.....:\n", c_mtrx)

    disp = ConfusionMatrixDisplay(confusion_matrix=c_mtrx, display_labels=class_labels)
    disp.plot(cmap="BuGn")
    if rf:
        plt.title("Confusion Matrix of Random Forest Classifier Model")
        plt.savefig("machine_learning/rf_confusion_matrix.png")
    else:
        plt.title("Confusion Matrix of Decision Tree Classifier Model")
        plt.savefig("machine_learning/dt_confusion_matrix.png")
    plt.show()
    print("\n")


def print_feature_importances(model, selected_features):
    print("--------------------------[:: FEATURE IMPORTANCES ::]--------------------------")
    importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
    for feature, importance in importances.items():
        print(f"---> {feature}: {importance:.4f}")

"""Function that prints evaluation statistics for the final two model's performances."""
def print_stats(model, X_train, y_train, X_test, y_test, selected_features, class_labels, rf=False):
    print("-------------------------------[:: MAX DEPTH ::]-------------------------------\n",
          "---> Depth =", model.max_depth, "\n")

    # if model is type random forest, also print the best n_estimators
    if rf:
        print("-------------------------------[:: N_ESTIMATORS ::]-------------------------------\n",
              "---> Number of estimators =", model.n_estimators, "\n")

    # print accuracy of training and testing data
    print_model_accuracy(model, X_train, y_train, X_test, y_test)

    # print the sorted (desc) order of feature importances (cumulative information gain?)
    print_feature_importances(model, selected_features)

    # plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test, class_labels, rf)

"""Function that returns the Top_F (a desired number of features) features that yield the highest mutual information 
with respect to the class label. 

Used per fold for Decision Tree Classifier Model.
Used per depth for Random Forest Classifier Model."""
def feature_selection(X, y, top_F):
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_series = pd.Series(mi, index=X.columns)
    return mi_series.sort_values(ascending=False).head(top_F).index.tolist()

"""Function that determines the best depth for the final Decision Tree Classifier Model, whilst using 
Stratified Cross Validation where the number of Folds is set to 5.

Best Depth is determined by the depth that yielded the highest average F1 score across each fold."""
def evaluate_decision_tree(model_fn,  X_train_validate, y_train_validate, depths=range(1,11), N=5, top_F=10,
                print_metric_summary=True):

    skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
    metrics_per_depth = {}
    metrics_summary = {}
    top_feature_counter = Counter()

    # ---{ evaluating model performance at each tree depth }--- #
    for depth in depths:
        metrics_per_depth[depth] = {"F1": [], "Recall": [], "Precision": []}

        # ---{ evaluating model performance for every fold }--- #
        for train_index, validation_index in skf.split(X_train_validate, y_train_validate):
            # determining indexes of training and validation set from the skf.get_N_splits(X, y)
            X_train = X_train_validate.iloc[train_index]
            y_train = y_train_validate.iloc[train_index]

            X_validation = X_train_validate.iloc[validation_index]
            y_validation = y_train_validate.iloc[validation_index]

            top_features = feature_selection(X_train, y_train, top_F)
            top_feature_counter.update(top_features)
            X_train = X_train[top_features]
            X_validation = X_validation[top_features]

            # training the model on our training portion
            model = model_fn(depth)
            model.fit(X_train, y_train)

            # predicting class label based on validation set
            y_prediction = model.predict(X_validation)

            # scoring
            metrics_per_depth[depth]["F1"].append(f1_score(y_validation, y_prediction, average='binary'))
            metrics_per_depth[depth]["Recall"].append(recall_score(y_validation, y_prediction, average='binary'))
            metrics_per_depth[depth]["Precision"].append(precision_score(y_validation, y_prediction, average='binary'))

    # for each depth, we are appending the average f1 score, recall and precision across each fold,
    # then append it to the summary
    for depth in metrics_per_depth:
        metrics_summary[depth] = {
            "Average F1": np.mean(metrics_per_depth[depth]["F1"]),
            "Average Recall": np.mean(metrics_per_depth[depth]["Recall"]),
            "Average Precision": np.mean(metrics_per_depth[depth]["Precision"])}

    # calculating max_depth hyperparameter based on the depth yielding highest F1 score
    best_depth = max(metrics_summary, key=lambda depth_level: metrics_summary[depth_level]["Average F1"])
    selected_features = [feature for feature, _ in top_feature_counter.most_common(top_F)]

    # printing average evaluation metric for each depth, as well as the overall best depth based on F1 score.
    if print_metric_summary:

        # ----{ prints the average of the evaluation metrics per depth, averages the score across the folds }--- #
        print("----------------[:: AVERAGE OF EVALUATION SCORES, PER DEPTH ::]----------------")
        for depth in metrics_per_depth:
            F1 = metrics_summary[depth]["Average F1"]
            Recall = metrics_summary[depth]["Average Recall"]
            Precision = metrics_summary[depth]["Average Precision"]
            print(f"---> Depth={depth}  ::  F1={F1:.4f}  ::  Recall={Recall:.4f}  ::  Precision={Precision:.4f}")

        # ----{ prints the best depth for final model usage, based on which depth yielded the highest F1 score }--- #
        print(f":: Best Depth by F1 Score = {best_depth}")

    # plotting the average F1, Recall, and Precision across each depth
    plot_best_depth_cv(metrics_summary, depths, "Average ")
    print("\n")

    # return best depth and updated feature selection based on most frequently selected features yielding most M.I
    return best_depth, selected_features

"""Function that uses evaluation metrics to determine the best n_estimators and depth hyperparameters for the final
Random Forest Classifier Model.

Best Depth and n_estimators is determined by the combination of these two parameters that yielded the highest 
F1 Score"""
def evaluate_random_forest(model_fn, X_train, y_train, X_validate, y_validate, depths, n_estimators, print_metric_summary=True):

    # narrow scope of selected features to the top 25 by highest mutual information (M.I)
    selected_features = feature_selection(X_train, y_train, top_F=25)

    # ---{ stores evaluation metrics at various n and depth combinations }--- #
    metrics_per_combination = {}
    print("------------[:: EVALUATION SCORES, PER DEPTH AND N_ESTIMATORS   ::]------------\n")

    # ---{ iterate through each depth specified range }--- #
    for depth in depths:

        # ---{ iterate through n within the n_estimators  array }--- #
        for n in n_estimators:

            # setting a key with the desired tuning params
            params = (depth, n)

            # fitting the model with other preset parameters, as well as varying depth and n vals.
            model = model_fn(depth, n)
            model.fit(X_train, y_train)
            y_prediction = model.predict(X_validate)

            # storing model's evaluation, per depth and n_estimators combination of hyperparameters
            f1 = f1_score(y_validate, y_prediction, average='binary')
            recall = recall_score(y_validate, y_prediction, average='binary')
            precision = precision_score(y_validate, y_prediction, average='binary')

            metrics_per_combination[params] = {
                "F1": f1,
                "Recall": recall,
                "Precision": precision
            }

            # print the evaluation metric of models performance per n_estimators, per depth
            if print_metric_summary:
                  print(f"---> Depth={depth}, n_estimators={n}  ::  ",
                        f"F1={f1:.4f}  |  ",
                        f"Recall={recall:.4f}  |  ",
                        f"Precision={precision:.4f}")

    # ---{ iterating through the metrics_per_combination dict to find the best performance of params by F1 }--- #
    best_combination = max(metrics_per_combination, key=lambda param_combo: metrics_per_combination[param_combo]["F1"])
    best_depth, best_n = best_combination

    # print best depth and number of estimators based on which combination of the two yielded the highest F1 score
    print(f":: Best Depth by F1 Score = {best_depth} ::\n:: Best n_estimators by F1 Score = {best_n} ::\n")

    # returning best_depth, best_n and columns used for the final model. feature selection pruned manually.
    return best_depth, best_n, selected_features


"""Function that plots the average F1, Recall and Precision Scores of the Decision Tree Classifier Model, 
    at each depth tested"""
def plot_best_depth_cv(metrics_summary, depths, score_prefix):
    plt.figure(figsize=(10, 6))

    # plotting average F1 score at each depth
    plt.plot(depths, [metrics_summary[depth][f"{score_prefix}F1"] for depth in depths],
             marker='o', label=f"{score_prefix}F1 Score")

    # plotting average Recall score at each depth
    plt.plot(depths, [metrics_summary[depth][f"{score_prefix}Recall"] for depth in depths],
             marker='s', label=f"{score_prefix}Recall Score")

    # plotting average Precision at each depth
    plt.plot(depths, [metrics_summary[depth][f"{score_prefix}Precision"] for depth in depths],
             marker='^', label=f"{score_prefix}Precision Score")

    plt.xlabel('Max Depth')
    plt.ylabel('Evaluation Scores')
    plt.title('Decision Tree Evaluation Metrics vs Max Depth')
    plt.xticks(depths)
    plt.legend()
    plt.savefig("machine_learning/dt_evaluation_per_depth.png")
    plt.show()