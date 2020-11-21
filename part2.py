# bulk of the code comes from CS434 Assignment 3 starter code
from os import replace
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

    def __init__(self, prediction, feature, left_tree, right_tree, gain):
        self.prediction = prediction
        self.feature = feature
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.gain = gain


class RandomForestClassifier():
    """
    Random Forest Classifier. Build a forest of decision trees.
    Use this forest for ensemble predictions

    YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

    Parameters:
    -----------
    n_trees: int
        Number of trees in forest/ensemble
    max_features: int
        Maximum number of features to consider for a split when feature bagging
    max_depth: int
        Maximum depth of any decision tree in forest/ensemble
    """

    def __init__(self, n_trees, max_features, max_depth):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth

    # fit all trees
    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.roots = []
        print('Fitting Random Forest...\n')
        for i in range(self.n_trees):
            bagged_idx = np.random.choice(np.arange(0, len(X)), size=len(X), replace=True)
            bagged_X = X[bagged_idx]
            bagged_y = y[bagged_idx]

            self.roots.append(self.build_tree(
                bagged_X, bagged_y, self.max_depth))
        print()

    def build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on
        self.features_idx = np.random.choice(
            X.shape[1], self.max_features, replace=False)
        # print(self.features_idx)

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
                    gain, left_X, right_X, left_y, right_y = self.calculate_information_gain(
                        X, feature, y)
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
                        right_tree=right_tree, gain=best_gain)
        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, left_tree=None, right_tree=None, gain=best_gain)

    def check_split(self, X, y, feature, split):
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain(y, left_y, right_y)
        return gain, left_X, right_X, left_y, right_y

    def bag_data(self, X, y, proportion=1.0):
        bagged_X = []
        bagged_y = []
        for i in range(self.n_trees):
            bag_indexes = np.random.random_integers(
                low=0, high=2097, size=(2098))
            bag_x = []
            bag_y = []
            for index in bag_indexes:
                bag_x.append(X[index])
                bag_y.append(y[index])
            bagged_X.append(bag_x)
            bagged_y.append(bag_y)

        # ensure data is still numpy arrays
        return np.array(bagged_X), np.array(bagged_y)

    def predict(self, X, num_trees):
        preds = [self._predict(example, num_trees) for example in X]
        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example, num_trees):
        roots = np.random.choice(np.arange(0, len(self.roots)), size=num_trees, replace=False)
        node = []
        results = []
        for root in roots:
            node = self.roots[root]
            while node.left_tree:
                if example[node.feature] == 0:
                    node = node.left_tree
                elif example[node.feature] == 1:
                    node = node.right_tree
            results.append(node.prediction)
        
        return np.bincount(results).argmax()

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
        e_target_var = ((left_x.shape[0] / len(X)) * self.calc_entropy(left_y)) + (
            (right_x.shape[0] / len(X)) * self.calc_entropy(right_y))

        information_gain = e_target - e_target_var
        return information_gain, left_x, right_x, left_y, right_y


def get_top_three_splits(tree, max_depth):
    print("d_max", max_depth)
    print("feature idx", tree.root.feature, "ig", tree.root.gain)
    print("feature idx", tree.root.left_tree.feature,
          "ig", tree.root.left_tree.gain)
    print("feature idx", tree.root.right_tree.feature,
          "ig", tree.root.right_tree.gain)
    print("---------------------------")


def random_forest_testing(x_train, y_train, x_test, y_test):
    print('\nRandom Forest\n')
    tree_amounts = np.arange(10, 110, step=10)
    feature_amounts = [5, 25, 50, 100]

    for d_max in [2, 10, 25]:
        feature_amount_train_accuracies = {}
        feature_amount_test_accuracies = {} 

        for m in feature_amounts:
            rclf = RandomForestClassifier(max_depth=d_max, max_features=m, n_trees=100)
            rclf.fit(x_train, y_train)

            # get training and validation accuracy for n number of trees
            train_accuracy = []
            test_accuracy = []
            for n in tree_amounts:
                preds_train = rclf.predict(x_train, n)
                preds_test = rclf.predict(x_test, n)
                train_accuracy.append(rclf.accuracy_score(preds_train, y_train))
                test_accuracy.append(rclf.accuracy_score(preds_test, y_test))

            feature_amount_train_accuracies[str(m)] = train_accuracy
            feature_amount_test_accuracies[str(m)] = test_accuracy

        train_results = pd.DataFrame(
            np.transpose([tree_amounts, feature_amount_train_accuracies["5"], feature_amount_train_accuracies["25"], feature_amount_train_accuracies["50"], feature_amount_train_accuracies["100"]]),
            columns=["Number of Trees", "m = 5", "m = 25", "m = 50", "m = 100"])

        plt.plot(tree_amounts, train_results["m = 5"].to_numpy(),
                label="m = 5")
        plt.plot(tree_amounts, train_results["m = 25"].to_numpy(),
                label="m = 25")
        plt.plot(tree_amounts, train_results["m = 50"].to_numpy(),
                 label="m = 50")
        plt.plot(tree_amounts, train_results["m = 100"].to_numpy(),
                 label="m = 100")
        plt.legend()
        plt.xlabel("Number of Trees")
        plt.ylabel("Accuracy %")
        plt.title("Training Accuracy d_max = {}".format(d_max))
        plt.show()

        plt.clf()

        test_results = pd.DataFrame(
            np.transpose([tree_amounts, feature_amount_test_accuracies["5"], feature_amount_test_accuracies["25"], feature_amount_test_accuracies["50"], feature_amount_test_accuracies["100"]]),
            columns=["Number of Trees", "m = 5", "m = 25", "m = 50", "m = 100"])

        plt.plot(tree_amounts, test_results["m = 5"].to_numpy(),
                 label="m = 5")
        plt.plot(tree_amounts, test_results["m = 25"].to_numpy(),
                 label="m = 25")
        plt.plot(tree_amounts, test_results["m = 50"].to_numpy(),
                 label="m = 50")
        plt.plot(tree_amounts, test_results["m = 100"].to_numpy(),
                 label="m = 100")
        plt.legend()
        plt.xlabel("Number of Trees")
        plt.ylabel("Accuracy %")
        plt.title("Testing Accuracy d_max = {}".format(d_max))
        plt.show()


def load_data(rootdir='./'):
	x_train = pd.read_csv(rootdir+'/pa4_train_X.csv').to_numpy()
	y_train = pd.read_csv(rootdir+'/pa4_train_y.csv').to_numpy()
	x_test = pd.read_csv(rootdir+'/pa4_dev_X.csv').to_numpy()
	y_test = pd.read_csv(rootdir+'/pa4_dev_y.csv').to_numpy()
	return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    np.random.seed(1)
    x_train, y_train, x_test, y_test = load_data(os.getcwd())
    random_forest_testing(x_train, y_train, x_test, y_test)
    print('Done')
