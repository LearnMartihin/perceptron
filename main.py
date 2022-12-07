import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [0,1,0],
                            [1,1,0],
                            [1,1,1]])

training_outputs = np.array([[1,1,0,0]]).T

synapric_w = 2*np.random.random((3,2)) - 1
synapric_w2 = 2*np.random.random((2,1)) - 1

print("Случайные веса 1:")
print(synapric_w)
print("Случайные веса 2:")
print(synapric_w2)
arr2 = []
for i in range(10000):
    input_layer = training_inputs

    layer1_outputs = sigmoid(np.dot(input_layer,synapric_w))
    layer2_outputs = sigmoid(np.dot(layer1_outputs, synapric_w2))

    derectiv1 = (layer1_outputs*(1-layer1_outputs))
    derectiv2 = (layer2_outputs * (1 - layer2_outputs))

    err = training_outputs - layer2_outputs
    arr = sum(err*err)

    arr2.append(arr[0])
    adjust = np.dot(input_layer.T, err*derectiv1)
    adjust2 = np.dot(layer1_outputs.T, err * derectiv2)

    synapric_w +=adjust
    synapric_w2 += adjust2
plt.plot(arr2, 'k')
plt.show()



print("layer1")
print(layer1_outputs)
print("layer2")
print(layer2_outputs)
print(adjust2)
print("After learning 1")
print(synapric_w)
print("After learning 2")
print(synapric_w2)
print("Result 1")
print(layer1_outputs)
print("Result 2")
print(layer2_outputs)
#new input
new_input = np.array([0,0,0])
wait = np.dot(synapric_w,synapric_w2)
print(wait)
sit = np.dot(new_input,wait)
output = sigmoid(sit)
print("situation")
print(output)


