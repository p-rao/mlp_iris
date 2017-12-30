import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#TODO
#from sklearn.datasets import load_iris

# Load dataset
traindata = pd.read_csv('./Datasets/iris/iris.csv')

# Change string value to numeric
traindata.set_value(traindata['species'] == 'Iris-setosa', ['species'], 0)
traindata.set_value(traindata['species'] == 'Iris-versicolor', ['species'], 1)
traindata.set_value(traindata['species'] == 'Iris-virginica', ['species'], 2)
traindata = traindata.apply(pd.to_numeric)

# Change dataframe to array
data_array = traindata.as_matrix()

# Split x and y (feature and target)
X_train, X_test, y_train, y_test = train_test_split(data_array[:, :4],
                                                    data_array[:, 4],
                                                    test_size=0.2)

"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris, Softmax Layer

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
maximum iterations = 1000
"""

mlp = MLPClassifier(hidden_layer_sizes=(10),solver='sgd',learning_rate_init=0.01,max_iter=1000)

# Train the model
mlp.fit(X_train, y_train)

# Test the model
# Changed since earlier type was deprecated
print(f"{mlp.score(X_test,y_test)}")

sl = 5.9
sw = 3.0
pl = 5.1
pw = 1.8
data = [[sl,sw,pl,pw]]
# Changed since earlier type was deprecated
print(f"{mlp.predict(data)}")



