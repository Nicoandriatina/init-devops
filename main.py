import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris = pd.read_csv(url, header=None, names=column_names)

# Vérifier les valeurs manquantes
print(iris.isnull().sum())

# Supprimer les doublons s'il y en a
iris.drop_duplicates(inplace=True)

# Visualiser les premières lignes du dataset
print(iris.head())

# Statistiques descriptives
print(iris.describe())

# Distribution des espèces
sns.countplot(x="species", data=iris)
plt.show()

# Pairplot pour visualiser les relations entre les variables
sns.pairplot(iris, hue="species")
plt.show()

from sklearn.model_selection import train_test_split

# Séparer les features et la target
X = iris.drop("species", axis=1)
y = iris["species"]

# Diviser le dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

# Définir les paramètres pour KNN
param_grid_knn = {'n_neighbors': np.arange(1, 20)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)
print("Best KNN parameters:", grid_knn.best_params_)

# Définir les paramètres pour SVM
param_grid_svm = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)
print("Best SVM parameters:", grid_svm.best_params_)

# Définir les paramètres pour Random Forest
param_grid_rf = {'n_estimators': [50, 100, 200]}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
print("Best Random Forest parameters:", grid_rf.best_params_)
