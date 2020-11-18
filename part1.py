# bulk of the code comes from CS434 Assignment 3 starter code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Node():
    """
    Node of decision tree

    Parameters:
    -----------
    prediction: int
        Class prediction at this node
    feature: int
        Index of feature used for splitting on
    split: int
        Categorical value for the threshold to split on for the feature
    left_tree: Node
        Left subtree
    right_tree: Node
        Right subtree
    """

    def __init__(self, prediction, feature, left_tree, right_tree):
        self.prediction = prediction
        self.feature = feature
        self.left_tree = left_tree
        self.right_tree = right_tree


class DecisionTreeClassifier():
    """
    Decision Tree Classifier. Class for building the decision tree and making predictions

    Parameters:
    ------------
    max_depth: int
        The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    # take in features X and labels y
    # build a tree
    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.root = self.build_tree(X, y, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]
        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] == 0:
                node = node.left_tree
            elif example[node.feature] == 1:
                node = node.right_tree
        return node.prediction

    # accuracy
    def accuracy_score(self, preds, y):
        correct = 0
        for i in range(len(preds)):
            if preds[i] == y[i]:
                correct += 1
        
        return correct / len(y)

    def calc_entropy(self, column):
        probs = np.bincount(column[:, 0]) / len(column)
        e = 0
        for p in probs:
            if p > 0:
                e += p * np.log2(p)

        return -e

    def calculate_information_gain(self, X, feature, y):
        left_split = np.where(X[:, feature] == 0)
        right_split = np.where(X[:, feature] == 1)

        left_x = X[left_split]
        right_x = X[right_split]

        left_y = y[left_split]
        right_y = y[right_split]

        e_target = self.calc_entropy(y)
        e_target_var = ((left_x.shape[0] / len(X)) * self.calc_entropy(left_y)) + ((right_x.shape[0] / len(X)) * self.calc_entropy(right_y))

        information_gain = e_target - e_target_var
        return information_gain, left_x, right_x, left_y, right_y

    # function to build a decision tree
    def build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on
        self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            if len(X) > 0:
                for feature in self.features_idx:
                    gain, left_X, right_X, left_y, right_y  = self.calculate_information_gain(X, feature, y)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y

        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(
                best_left_X, best_left_y, depth=depth + 1)
            right_tree = self.build_tree(
                best_right_X, best_right_y, depth=depth + 1)
            return Node(prediction=prediction, feature=best_feature, left_tree=left_tree,
                        right_tree=right_tree)
        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, left_tree=None, right_tree=None)

def decision_tree_testing(x_train, y_train, x_test, y_test):
    print('\nDecision Tree\n')
    test_accuracies = []
    training_accuracies = []
    d_max = [2, 5, 10, 20, 25, 30, 35, 40, 45, 50]
    for i in d_max:
        clf = DecisionTreeClassifier(max_depth=i)
        clf.fit(x_train, y_train)
        preds_train = clf.predict(x_train)
        preds_test = clf.predict(x_test)
        training_accuracies.append(clf.accuracy_score(preds_train, y_train))
        test_accuracies.append(clf.accuracy_score(preds_test, y_test))

    decision_tree_results = pd.DataFrame(
        np.transpose([d_max, training_accuracies,
                      test_accuracies]),
        columns=["Max Depth", "Training Accuracy", "Test Accuracies"])

    print(decision_tree_results)
    plt.plot(d_max, test_accuracies, label="Testing Accuracy")
    plt.plot(d_max, training_accuracies, label="Training Accuracy")
    plt.legend()
    plt.xlabel("Maximum Tree Depth")
    plt.title("Tree Depth Vs Accuracy")
    plt.show()

def load_data(rootdir='./'):
	x_train = pd.read_csv(rootdir+'/pa4_train_X.csv').to_numpy()
	y_train = pd.read_csv(rootdir+'/pa4_train_y.csv').to_numpy()
	x_test = pd.read_csv(rootdir+'/pa4_dev_X.csv').to_numpy()
	y_test = pd.read_csv(rootdir+'/pa4_dev_y.csv').to_numpy()
	return x_train, y_train, x_test, y_test

def accuracy_score(preds, y):
	accuracy = (preds == y).sum()/len(y)
	return accuracy

if __name__ == '__main__':
    np.random.seed(212)
    x_train, y_train, x_test, y_test = load_data(os.getcwd())
    decision_tree_testing(x_train, y_train, x_test, y_test)
    print('Done')
