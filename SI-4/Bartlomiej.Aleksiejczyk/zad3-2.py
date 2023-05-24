import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib


labels = ["a1", "a2", "a3"]
X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 1],
    [1, 1, 0]
], dtype=int)
Y=[0,0,0,1,1]

clf = tree.DecisionTreeClassifier(criterion = "entropy")
clf = clf.fit(X, Y)
text_representation = tree.export_text(clf)
print(text_representation)
tree.plot_tree(clf)