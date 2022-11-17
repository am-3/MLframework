# Creating a custom neural network
import os
os.chdir('E:\VS Code Programs\Python_Codes\Deep_Learning')

import numpy as np

import pandas as pd
df = pd.read_csv('insurance.csv')

X = df.drop(['bought_insurance'], axis=1)
X.age = X.age / 100
y = df.bought_insurance

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, (1-epsilon)) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
    
    def fit(self, X_train, y_train, epochs):
        self.w1, self.w2, self.bias = self.gradient_descent(X_train ,y_train, epochs)
    
    def predict(self, X_test):
        weighted_sum = self.w1 * X_test['age'] + self.w2 * X_test['affordibility'] + self.bias
        return sigmoid(weighted_sum)
    
    def gradient_descent(self, X_train, y_train, epochs):
        rate = 0.5
        n = len(X_train.age)

        for i in range(epochs):
            weighted_sum = self.w1*X_train.age + self.w2*X_train.affordibility + self.bias
            y_predicted = sigmoid(weighted_sum)

            loss = log_loss(y_train, y_predicted)

            w1d = (1/n)*np.dot(np.transpose(X_train.age), (y_predicted-y_train))
            w2d = (1/n)*np.dot(np.transpose(X_train.affordibility), (y_predicted-y_train))
            bias_d = np.mean(y_predicted-y_train)

            self.w1 = self.w1 - rate*w1d
            self.w2 = self.w2 - rate*w2d
            self.bias = self.bias - rate*bias_d

            print("epoch:{}, w1:{}, w2:{}, bias:{}, loss:{}".format(i, self.w1, self.w2, self.bias, loss))

        return self.w1, self.w2, self.bias

    def score(self, X_test, y_test):
        weighted_sum = self.w1 * X_test['age'] + self.w2 * X_test['affordibility'] + self.bias
        y_predicted = sigmoid(weighted_sum)

        X_test_new = []
        for i in range(len(y_test)):
            if X_test[i] >= 0.5:
                X_test_new[i] = 1
            elif X_test[i] < 0.5:
                X_test_new[i] = 0
            else:
                continue
        
        count = 0
        for i in range(y_test):
            if X_test_new[i] == y_test[i]:
                count +=1
            else:
                continue
        
        return count/len(y_test)

model = myNN()
print(model.predict(X_test))
print(model.score(X_test, y_test))