import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [0,1,0],
                            [1,0,0],
                            [0,1,1],
                            [1,0,1]])

training_outputs = np.array([[0,1,1,0,0]]).T

synapric_w = 2*np.random.random((3,1)) - 1

print("Случайные веса 1:")
print(synapric_w)

for i in range(10000):
    input_layer = training_inputs
    layer1_outputs = sigmoid(np.dot(input_layer,synapric_w))
    err = training_outputs - layer1_outputs
    adjust = np.dot(input_layer.T, err*(layer1_outputs*(1-layer1_outputs)))
    synapric_w +=adjust
print(layer1_outputs)

print("After learning")
print(synapric_w)
print("Result")
print(layer1_outputs)
#new input
new_input = np.array([1,1,0])
output = sigmoid(np.dot(new_input,synapric_w))
print("situation")
print(output)