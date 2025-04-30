import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def model_accuracy(model, X_train, y_train, X_test, y_test):
    training_accuracy = model.score(X_train, y_train)
    testing_accuracy = model.score(X_test, y_test)
    print('Accuracy of training set:', training_accuracy)
    print('Accuracy of testing set:', testing_accuracy)
    print('Accuracy difference:', testing_accuracy - training_accuracy)

def find_best_depth(X_train, y_train, X_test, y_test):
    depths = range(1, 11)
    train_accuracies = []
    test_accuracies = []

    for depth in depths:
        model = DecisionTreeClassifier(criterion='entropy', max_depth=depth,  class_weight='balanced')
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
    disp = ConfusionMatrixDisplay(confusion_matrix=c_mtrx, display_labels=class_labels)
    disp.plot()
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
