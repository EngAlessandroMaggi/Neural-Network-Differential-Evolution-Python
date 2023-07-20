import numpy as np
import pandas as pd    #reading data
from matplotlib import pyplot as plt

data = pd.read_csv('/Users/alessandromaggi/Downloads/train.csv')  #import data with pandas
data = np.array(data) #I want to work with numpy arrays
m, n = data.shape
np.random.shuffle(data) #shuffle before splitting into dev and training sets

#data_dev
data_dev = data[0:1000].T #transpose x look in the math note
Y_dev = data_dev[0]  #first row
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

#data_train
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape
print("array", Y_train)

#initialize the params
def init_params():
    W1 = np.random.rand(10, 784) - 0.5   #dist. between 0.5 and -0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

#activation function hidden layer
def ReLU(Z):
    return np.maximum(Z, 0)

#activation function output layer
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

#def forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, beta, prev_W1 , prev_b1, prev_W2, prev_b2):

    new_W1 = W1 - alpha * dW1 + beta * (W1 - prev_W1)
    new_b1 = b1 - alpha * db1 + beta * (b1 - prev_b1)
    new_W2 = W2 - alpha * dW2 + beta * (W2 - prev_W2)
    new_b2 = b2 - alpha * db2 + beta * (b2 - prev_b2)

    prev_W1 = W1
    prev_W2 = W2
    prev_b1 = b1
    prev_b2 = b2

    return new_W1, new_b1, new_W2, new_b2, prev_W1, prev_b1, prev_W2, prev_b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations,beta):
    W1, b1, W2, b2 = init_params()
    prev_W1 = 0
    prev_W2 = 0
    prev_b1 = 0
    prev_b2 = 0
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2, prev_W1, prev_b1, prev_W2, prev_b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, beta, prev_W1, prev_b1, prev_W2, prev_b2)
        #if i % 10 == 0:
            #print("Iteration: ", i)
            #predictions = get_predictions(A2)
            #print("Accuracy: ", get_accuracy(predictions, Y))

    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, Y)
    return W1, b1, W2, b2, accuracy

#DIFFERENTIAL EVOLUTION

iteration = 20
Mutant = 0.3 #value between 0 and 1
Hyper_set = 10 #set number of coefficents
Hyper_num = 2 # hyperparameters number (alpha and beta)
H = np.random.randint(9, size=(Hyper_set, Hyper_num)) #integer numbers between 0-9
H = H / 10  # I divide integer numbers in rational number
print("hyperparameters matrix : ", H)
accuracy = np.zeros(Hyper_set)
tentative_point = np.zeros(2)
accuracy_trial = np.zeros(1)

#differential evolution
for i in range(Hyper_set):
    for j in range(H.shape[0]):
        W1, b1, W2, b2,  accuracy[j] = gradient_descent(X_train, Y_train, H[j][0], iteration, H[j][1])
    #mutation
    rnd = np.random.choice(H.shape[0], 10, False) #False =  I can't get the same 2 times
    #randomly choose a,b,c ---> rnd[0], rnd[1], rnd[2]
    tentative_point = H[rnd[0]] + Mutant * (H[rnd[1]] - H[rnd[2]]) #tentative point
    #Accuracy tentative point
    W1, b1, W2, b2, accuracy_trial = gradient_descent(X_train, Y_train, tentative_point[0], iteration, tentative_point[1])
    #sostitution
    #Not negative tentative_point
    if(accuracy_trial > accuracy[rnd[i]] and tentative_point[0] > 0 and tentative_point[1] > 0 ):
        H[i] = tentative_point
#end of differential evolution


#accurancy of hyper-parameters after differential evolution

for j in range(H.shape[0]):
    W1, b1, W2, b2, accuracy[j] = gradient_descent(X_train, Y_train, H[j][0], iteration, H[j][1])

#log of our hyper - parameters and accuracy
print("hyper-parameters matrix after differential evolution : ", H)
print("accuracy matrix after differential evolution : ", accuracy)
index = np.where(accuracy == np.amax(accuracy)) #where is the best accuracy ? (index)
print('Best huper-parameters index :', index)
print('Best couple of hyper-parameters', H[index])

#momentum with best parameters
iteration = 500
accuracy_best = np.zeros(1)
W1, b1, W2, b2, accuracy_best[0] = gradient_descent(X_train, Y_train, H[index][0][0], iteration, H[index][0][1])

print("prediction-label - best couple of hyper-parameters : ")
print("after 500 iterations I have an accuracy : ", accuracy_best)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


for i in range(10):
    test_prediction(i, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)


