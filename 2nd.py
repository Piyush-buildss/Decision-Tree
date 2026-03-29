from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(iris.data, iris.target)

plt.figure(figsize=(12,6))
plot_tree(clf, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()