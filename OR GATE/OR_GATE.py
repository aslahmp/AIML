import numpy as np

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 1]

weight1 = np.random.rand(1)
weight2 = np.random.rand(1)
bias = np.random.rand(1)
learning_rate = 0.1
epochs = 100
print("Initial Values are:")
print(f"Weight1={weight1} \nWeight2={weight2} \nBias={bias}")
for epoch in range(epochs):
    print(f"\nEpoch:{epoch+1}")
    total_error = 0
    for i in range(len(inputs)):
        input1, input2 = inputs[i]
        target_output = outputs[i]

        if (input1 * weight1 + input2 * weight2 + bias) >= 0:
            output = 1
        else :
            output=0

        error = (target_output - output)
        weight1 += learning_rate * error * input1
        weight2 += learning_rate * error * input2
        bias += learning_rate * error
        total_error += error
        print(f"Input: {inputs[i]}, Observed Output: {output}, Targeted Output: {outputs[i]}")
        print(f"Error={error}")
        print(f"Total Error={total_error}")
        print(f"Weight1={weight1}, Weight2={weight2}, Bias={bias}")
    if total_error == 0:
        print(f"\nConverged in {epoch+1} epochs.")
        break
for i in range(len(inputs)):
    input1, input2 = inputs[i]
    output = 1 if (input1 * weight1 + input2 * weight2 + bias) >= 0 else 0
    print(f"Input: {inputs[i]}, Output: {output}")




