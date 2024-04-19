import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

'''fromWeb = True
if fromWeb:
    os.environ['KAGGLE_USERNAME'] = 'riccardomedri'
    os.environ['KAGGLE_KEY'] = 'c8e22e7b961a65aee0e66ae9f00b49c1'
    from kaggle.api.kaggle_api_extended import KaggleApi # MUST BE HERE
    dataset = 'uciml/iris'
    path = 'datasets/iris'
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path)
    api.dataset_download_file(dataset, 'Iris.csv', path)
    api.dataset_download_file(dataset, 'database.sqlite', path)'''

iris = pd.read_csv('datasets/iris/iris.csv')

X = iris.iloc[:,1:5]
y = iris['Species']
names = iris.columns[1:5] 
feature_names = iris.columns[5]
targets = np.unique(y)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,6))
for target, target_name in enumerate(targets): # for each species
    X_plot = X[y == targets[target]]
    ax1.plot(X_plot.iloc[:,0], X_plot.iloc[:,1], linestyle='none', marker='o', label=target_name)
    ax2.plot(X_plot.iloc[:,2], X_plot.iloc[:,3], linestyle='none', marker='o', label=target_name)
ax1.set_xlabel(names[0]);ax1.set_ylabel(names[1])
ax2.set_xlabel(names[2]);ax2.set_ylabel(names[3])
ax1.axis('equal') # only left plot, equal x and y unit lengths
ax1.legend();ax2.legend()
plt.show()


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
print("Feature names:", feature_names)
print("Target names:", target_names)
print("\nFirst 10 rows of X:\n", X[:10])

# X and y already loaded
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=1)
#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier
#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation =
'relu',solver='adam',random_state=1)
#Fitting the training data to the network
classifier.fit(X_train, y_train)
#Predicting y for X_val
y_pred = classifier.predict(X_test)
#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix
#Comparing the predictions against the actual observations
confmat = confusion_matrix(y_pred, y_test)
#Printing the accuracy
print("Accuracy of MLPClassifier, confusion matrix:")
print(confmat)
