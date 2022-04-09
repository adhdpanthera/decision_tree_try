# desicion tree method
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1], 'X_2': [0, 0, 0, 1, 0, 0, 0, 1], 'Y': [1, 1, 1, 1, 0, 0, 0, 0]})

clf = tree.DecisionTreeClassifier(criterion='entropy')
X=data[['X_1', 'X_2']]
y=data.Y

clf.fit(X, y)

tree.plot_tree(clf, feature_names=list(X), class_names=['Negative', 'Positive'], filled=True)
plt.show()