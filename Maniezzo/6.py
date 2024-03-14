import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/iris/iris.csv')
X = df.iloc[:, 1:5].values
y = df.iloc[:, 5].values
print(X[0:5])
print(y[0:5])
# categorical to numeric
le = preprocessing.LabelEncoder()
y1 = le.fit_transform(y)
# one-hot encoding of categories, another way
df["Species"] = df["Species"].map({"Iris-setosa": 0, "Iris-virginica": 1,
"Iris-versicolor": 2})
Y = pd.get_dummies(y1).values
print(Y[0:5])
# Scale data to have 0 means and variance 1, it helps convergence
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
test_size = 0.2
indices = np.random.permutation(len(df))
n_test_samples = int(test_size * len(df))
X_train = X_scaled[indices[:-n_test_samples]]
y_train = Y[indices[:-n_test_samples]]
#y_train = y1[indices[:-n_test_samples]]
#y_train = y_train[:,np.newaxis] # transpose
X_test = X_scaled[indices[-n_test_samples:]]
y_test = Y[indices[-n_test_samples:]]
#y_test = y1[indices[-n_test_samples:]]
#y_test = y_test[:,np.newaxis] # transpose

nhid1 = 4
nhid2 = 4
nout = 3
model = tf.keras.Sequential([
    tf.keras.layers.Dense(nhid1,input_dim=4, activation='relu'), # 1st hidden
    tf.keras.layers.Dense(nhid2,activation='relu'), # 2nd hidden
    tf.keras.layers.Dense(nout, activation='softmax')
   ], name='MLP')
print(model.summary())
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.01)
loss = tf.keras.losses.categorical_crossentropy
#tf.keras.losses.mean_squared_error
model.compile(optimizer = optimizer,
            loss = loss,
            metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size=5, epochs=100,verbose=1)
loss_train, train_accuracy = model.evaluate(X_train, y_train)
loss_test, test_accuracy = model.evaluate(X_test, y_test)
print(f'The training set accuracy for the model is {train_accuracy}\n The test set accuracy for the model is {test_accuracy}')

y_pred = model.predict(X_test)
actual = np.argmax(y_test, axis=1)
predicted = np.argmax(y_pred, axis=1)
print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
hist_val = model.fit(X_test, y_test, batch_size=5, epochs=100,verbose=1)
# Hereplot the training and validation loss and accuracy
fig, ax = plt.subplots(1,2,figsize = (12,4))
ax[0].plot(history.history['loss'], 'r',label = 'Training Loss')
ax[0].plot(hist_val.history['loss'],'b',label = 'Validation Loss')
ax[1].plot(history.history['accuracy'], 'r',label = 'Training Accuracy')
ax[1].plot(hist_val.history['accuracy'],'b',label = 'Validation Accuracy')
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[1].set_ylabel('Accuracy %')
fig.suptitle('MLP Training', fontsize = 24)
plt.show()

from sklearn.metrics import roc_curve, auc
Y_pred = model.predict(X_test)
plt.figure(figsize=(9, 8))
plt.plot([0, 1], [0, 1], 'k--')
fpr, tpr, threshold = roc_curve(y_test.ravel(), Y_pred.ravel())
plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(model, auc(fpr, tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()