from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn import datasets


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # konfiguruje generator znaczników i mapę kolorów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # rysuje wykres powierzchni decyzyjnej
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # rysuje wykres wszystkich próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl,
                    edgecolor='black')

#1. Rozdziel zestaw danych na podzbiory uczący i testowy,
iris = datasets.load_iris()
X = iris.data[:, [1,2]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

# 2. Sprawdź działanie drzewa dla entropii i współczynnika Giniego - porównaj wyniki i uargumentuj rezultaty.

clf = tree.DecisionTreeClassifier(criterion = 'entropy') # Uzupełnić parametry konstruktora
clf = clf.fit(X, y)
tree.plot_tree(clf)
plt.show()

plot_decision_regions(X=X_test, y=y_test, classifier=clf)
plt.title('Entropia')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc='upper left')
plt.show()

# zapis graficzny drzewa i otwarcie go
dat = export_graphviz(clf, out_file=None,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dat)


tree_gini = tree.DecisionTreeClassifier()
tree_gini.max_depth = 3
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('depth=3')
tree.plot_tree(tree_gini)
score = tree_gini.score(X_test, y_test)
print(f'gini tree score for depth 3: {score}')
plt.show()

tree_gini = tree.DecisionTreeClassifier()
tree_gini.max_depth = 4
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('depth=4')
tree.plot_tree(tree_gini)
score = tree_gini.score(X_test, y_test)
print(f'gini tree score for depth 4: {score}')
plt.show()

tree_gini = tree.DecisionTreeClassifier()
tree_gini.max_depth = 5
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('depth=5')
tree.plot_tree(tree_gini)
score = tree_gini.score(X_test, y_test)
print(f'gini tree score for depth 5: {score}')
plt.show()

tree_gini.max_depth = 6
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('depth=6')
tree.plot_tree(tree_gini)
score = tree_gini.score(X_test, y_test)
print(f'gini tree score for depth 6: {score}')
plt.show()

# powyżej tej wartości drzewa się zapętlają i wyglądają tak samo
tree_gini.max_depth = 7
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('depth=7')
tree.plot_tree(tree_gini)
score = tree_gini.score(X_test, y_test)
print(f'gini tree score for depth 7: {score}')
plt.show()

tree_gini.max_depth = 8
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('depth=8')
tree.plot_tree(tree_gini)
score = tree_gini.score(X_test, y_test)
print(f'gini tree score for depth 8: {score}')
plt.show()


tree_gini.max_depth = 9
tree_gini = tree_gini.fit(X_train, y_train)
plt.title('depth=9')
tree.plot_tree(tree_gini)
score = tree_gini.score(X_test, y_test)
print(f'gini tree score for depth 9: {score}')
plt.show()
#mniejsza złożoność obliczeniowa
#głebokość wpływa na ilość klas. im głębsze tym więcej klas

forest_entropy_five = RandomForestClassifier(criterion='entropy', n_estimators=10)
forest_entropy_five.fit(X=X_test, y=y_test)
score = forest_entropy_five.score(X_test,y_test)
print(f'forest entropy for 5 iter: {score}')
plot_decision_regions(X_test, y_test, classifier=forest_entropy_five, test_idx=range(105, 150))
plt.title('Las dla 5 iteracji')
plt.legend(loc='upper left')
plt.show()

forest_entropy_ten = RandomForestClassifier(criterion='entropy', n_estimators=10)
forest_entropy_ten.fit(X=X_test, y=y_test)
score = forest_entropy_ten.score(X_test,y_test)
print(f'forest entropy for 10 iter: {score}')
plot_decision_regions(X_test, y_test, classifier=forest_entropy_ten, test_idx=range(105, 150))
plt.title('Las dla 10 iteracji')
plt.legend(loc='upper left')
plt.show()

forest_entropy = RandomForestClassifier(criterion='entropy', n_estimators=30)
forest_entropy.fit(X=X_test, y=y_test)
score = forest_entropy.score(X_test,y_test)
print(f'forest entropy for 30 iter: {score}')
plot_decision_regions(X_test, y_test, classifier=forest_entropy, test_idx=range(105, 150))
plt.title('Las dla 30 iteracji')
plt.legend(loc='upper left')
plt.show()

forest_entropy = RandomForestClassifier(criterion='entropy', n_estimators=100)
forest_entropy.fit(X=X_test, y=y_test)
score = forest_entropy.score(X_test,y_test)
print(f'forest entropy for 100 iter: {score}')
plot_decision_regions(X_test, y_test, classifier=forest_entropy, test_idx=range(105, 150))
plt.title('Las dla 100 iteracji')
plt.legend(loc='upper left')
plt.show()

forest_entropy = RandomForestClassifier(criterion='entropy', n_estimators=200)
forest_entropy.fit(X=X_test, y=y_test)
score = forest_entropy.score(X_test,y_test)
print(f'forest entropy for 200 iter: {score}')
plot_decision_regions(X_test, y_test, classifier=forest_entropy, test_idx=range(105, 150))
plt.title('Las dla 200 iteracji')
plt.legend(loc='upper left')
plt.show()

# Większa ilość iteracji sprzyja dokładności
