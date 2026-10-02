import numpy as np
import matplotlib.pyplot as plt

#input dataset

X = np.array([30, 35, 40, 45, 50, 60, 70, 80], dtype=float)
Y = np.array([0,  0,  0,  0,  0,  1,  1,  1],dtype=float)

# sigmoid function

def sigmoid(z):
    return 1/(1+np.exp(-z))

# initial parameters

w = 0.0
b = 0.0

n = len(Y)
epochs = 1000

learning_rate = 0.001

# Training
for i in range(epochs):
    # forward pass

    z = w * X + b

    probabilities = sigmoid(z)

    # binary cross-entropy loss

    clipped_probabilities = np.clip(
    probabilities,
    1e-15,
    1 - 1e-15
    )

    loss = -np.mean(Y * np.log(clipped_probabilities) + (1-Y) * np.log(1-clipped_probabilities))

    # calculate gradients

    dw = np.sum ((probabilities - Y) * X) / n
    db = np.sum (probabilities - Y) / n

    # gradient descent: update w & b

    w = w - learning_rate * dw
    b = b - learning_rate * db

plt.scatter(X,Y)
plt.title("Humidity vs Rain")
plt.xlabel("Humidity (%)")
plt.ylabel("Rain (0:No, 1:Yes)")
plt.show()

print("w =", w)
print("b =", b)

# prediction

def predict_rain(humidity):
    z = w * humidity + b
    probability = sigmoid(z)
    return probability

# classification

def classify_rain(humidity):
    probability = predict_rain(humidity)
    if probability >= 0.5 :
        return "Rain"
    else :
        return "No Rain"

# test

humidity = float(input("Enter humidity(%): "))
probability = predict_rain(humidity)

print("Probability of rain:", probability)
print("Prediction: ",classify_rain(humidity))

