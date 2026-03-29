from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(iris.data, iris.target)

y_pred = clf.predict(iris.data)
print("Accuracy:", accuracy_score(iris.target, y_pred))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

clf2 = DecisionTreeClassifier(max_depth=3)
clf2.fit(X_train, y_train)

y_pred2 = clf2.predict(X_test)
print("Real Accuracy:", accuracy_score(y_test, y_pred2))

iris = datasets.load_iris()
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(iris.data, iris.target)

plt.figure(figsize=(12,6))
plot_tree(clf, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()