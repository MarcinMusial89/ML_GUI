import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# import pickle

# Data import form npy file
data, ro, inv_ro = np.load('data.npy', allow_pickle=True)
data = data.T
print('Data size:')
print(np.shape(data))
print(np.shape(ro))
print(np.shape(inv_ro))

# Data check
data_set_number = 5
x = data[data_set_number]
x2 = ro[data_set_number]
x3 = inv_ro[data_set_number]

# Data presentation on chart
figure1 = plt.figure(figsize=(16, 4))
ax1 = figure1.add_subplot(111)
p1, = ax1.plot(x)
line1 = ax1.axvline(x2, c='r')
line2 = ax1.axvline(x3, c='g')
ax1.legend([line1, line2], ['ro', 'inv_ro'])
plt.show()

# Data preparation for model training
X_train, X_test, y_train, y_test = train_test_split(data, ro, test_size=.1, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
# Created model save to pickle file
filename = "LR_inv_ro_estimator.pickle"
#pickle.dump(model, open(filename, "wb"))
# How to load model form pickle file
# loaded_model = pickle.load(open(filename, "rb"))

print("Score:")
print(model.score(X_test, y_test))

# Checking model with other data
data2, ro2, inv_ro2 = np.load('data_test_not_teached.npy', allow_pickle=True)
data2 = data2.T
print('Test data size:')
print(np.shape(data2))
print(np.shape(ro2))
print(np.shape(inv_ro2))
# Chose data set to predict ro or inv_ro
data_set_no = 8
inp_data = data2[data_set_no].reshape(1,-1)
# Estimation of ro/inv_ro by LR model
y_pred = model.predict(inp_data)
print(f"Predict value: {y_pred}")
print(f"Real value: {ro2[data_set_no]}")
# Presentation of results
figure2 = plt.figure(figsize=(16, 4))
ax1 = figure2.add_subplot(111)
p1, = ax1.plot(data2[data_set_no])
line1 = ax1.axvline(y_pred, c='r', linestyle='--')
line2 = ax1.axvline(ro2[data_set_no], c='b')
ax1.legend([line1, line2], ['Predicted', 'Real'])
plt.show()
