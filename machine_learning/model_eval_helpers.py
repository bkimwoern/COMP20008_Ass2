import pandas as pd
import matplotlib.pyplot as plt
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


def plot_confusion_matrix(model, X_test, y_test, class_labels):
    y_pred = model.predict(X_test)
    c_mtrx = confusion_matrix(y_test, y_pred, labels=class_labels)
    f1_score(y_test, y_pred, class_labels)
    recall_score(y_test, y_pred, class_labels)
    precision_score(y_test, y_pred, class_labels)
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
    print("FEATURE IMPORTANCES:")
    importances = pd.Series(model.feature_importances_, index=X_columns).sort_values(ascending=False)
    for feature, importance in importances.items():
        print(f"{feature}: {importance:.4f}")


def print_stats(model, X_train, y_train, X_test, y_test, X_columns, class_labels):
    print('MAX_DEPTH:', model.max_depth, '\n')

    # print accuracy of training and testing data
    print_model_accuracy(model, X_train, y_train, X_test, y_test)

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
        model = DecisionTreeRegressor().fit(X_train, y_train)  # LinearRegression()
        y_predict = model.predict(X_test)

        # Plot the scatter plot and the line of regression
        plt.scatter(X_test, y_test)
        plt.plot(X_test, y_predict, color='red')
        plt.title('Test set scatter plot')
        plt.xlabel(x)
        plt.ylabel('fatal')

        plt.show()