import math
import numpy as np
import pandas as pd

class MLP:
    def __init__(self, X, y):
        self.input = X
        self.weights_i_h = np.random.rand(self.input.shape[1], 4)
        self.weights_h_o = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.loss = 0
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def feed_forward_prop(self):
        self.hidden_net = np.dot(self.input, self.weights_i_h)
        self.hidden_out = np.reshape([self.sigmoid(x) for x in self.hidden_net.flatten()], self.hidden_net.shape)
        self.output_net = np.dot(self.hidden_out, self.weights_h_o)
        self.output = np.array([self.sigmoid(x) for x in self.output_net.flatten()])
        
        
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def loss_function(self):
        self.loss = np.dot(self.y - self.output,self.y -self.output)
        return self.loss
    
    def accuracy(self):
        b = np.array([0, 0, 0, 0])
        b[self.output > 0.6] = 1
        c = [1 for i in range(self.output.shape[0]) if b.item(i) == self.y.item(i)]
        return float(sum(c)/self.output.shape[0])
    
    def back_propagation(self):
        delta_h_o = 2*(self.y - self.output) * np.reshape([self.sigmoid_derivative(x) for x in self.output.flatten()], self.output.shape)
        loss_wrt_weights_h_o = np.reshape(np.dot(self.hidden_out.T,delta_h_o ), self.weights_h_o.shape)
        
        sigmoid_derivative_1 = np.reshape([self.sigmoid_derivative(x) for x in self.output.flatten()], self.output.shape)
        sigmoid_derivative_2 = np.reshape([self.sigmoid_derivative(x) for x in self.hidden_out.flatten()], self.hidden_out.shape)
        
        delta_i_h = np.reshape(2*(self.y - self.output) * sigmoid_derivative_1, self.weights_h_o.shape)
        loss_wrt_weights_i_h = np.reshape(np.dot(self.input.T, (np.dot(delta_i_h, self.weights_h_o.T) * sigmoid_derivative_2)), self.weights_i_h.shape)

        self.weights_i_h += loss_wrt_weights_i_h
        self.weights_h_o += loss_wrt_weights_h_o


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 0, 0, 1])        
xor = MLP(X, y)
for i in range(100):
    xor.feed_forward_prop()
    xor.back_propagation()

print("Loss: "+str(xor.loss_function()))
print("Predicted Output: "+str(xor.output))
print("Real Output: "+str(xor.y))
print("Accuracy %: "+str(xor.accuracy()*100))