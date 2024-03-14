from sklearn import preprocessing
import torch
import torch.nn as nn # neural network
import torch.nn.functional as F # activation functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

iris = pd.read_csv('datasets/iris/iris.csv')

X = iris.iloc[:,1:5]
y = iris['Species']

class Model(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3):
        super().__init__()
        self.input = nn.Linear(input_features, hidden_layer1)
        self.hidden = nn.Linear(hidden_layer1, hidden_layer2)
        self.output = nn.Linear(hidden_layer2, output_features)
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x

# Scale data to have 0 means and variance 1, it helps convergence
scaler = preprocessing.StandardScaler() #lo scaler normalizza i dati, questo aiuta nell'apprendimento del modello
X_scaled = scaler.fit_transform(X)
# Split the data set into training and testing
test_size = 0.2
indices = np.random.permutation(len(iris)) # shuffle records / species
n_test_samples = int(test_size * len(iris))
X_train = X_scaled[indices[:-n_test_samples]]
y_train = y[indices[:-n_test_samples]]
X_test = X_scaled[indices[-n_test_samples:]]
y_test = y[indices[-n_test_samples:]]
# no string tensor in pytorch. Strings to index
le = preprocessing.LabelEncoder()
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(le.fit_transform(y_train))
y_test = torch.LongTensor(le.fit_transform(y_test))

model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train) # foreward step
    loss = loss_fn(y_pred, y_train) # compute loss with current weights
    losses.append(loss.item())
    print(f'epoch: {i:2} loss: {loss.item():10.8f}')
    optimizer.zero_grad() # initialize gradients of optimized tensors to 0
    loss.backward() # compute new gradient (backward step)
    optimizer.step()

la = np.array(losses)
plt.figure()
plt.plot(range(epochs), la)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.title("Losses")

enc = OneHotEncoder() # One hot encoding
Y_onehot = enc.fit_transform(y_test[:, np.newaxis]).toarray() # need onehot in ROC
with torch.no_grad(): # do not update gradients
    y_pred = model(X_test).numpy()
    fpr, tpr, threshold = roc_curve(Y_onehot.ravel(), y_pred.ravel())
plt.figure(figsize=(6,6))
plt.plot([0, 1], [0, 1], 'k--') # dotted reference line
plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()