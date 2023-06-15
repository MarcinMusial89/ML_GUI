import numpy as np
import seaborn as sns
from tensorflow import keras
from tensorflow.random import set_seed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data, ro, inv_ro = np.load('data.npy', allow_pickle=True)

data = data.T
print(np.shape(data))
print(np.shape(ro))
print(np.shape(inv_ro))

#df = pd.DataFrame()
#df['ro'] = ro
#df['inv ro'] = inv_ro
#df['data'] = data.tolist()
#ro_tf = tf.convert_to_tensor(ro, dtype = tf.int64)
#data_tf = tf.convert_to_tensor(data, dtype = tf.int64)

ro_vec = np.zeros((len(ro), 10000))
for i in range(len(ro)):
    ro_vec[i][int(ro[i])]=1

X_train, X_test, y_train, y_test = train_test_split(data, ro_vec, test_size=0.2, random_state=True)

set_seed(0)
np.random.seed(0)

inputs = keras.Input(shape=X_train.shape[1])
hidden_layer0 = keras.layers.Dense(1000, activation="relu")(inputs)
hidden_layer1 = keras.layers.Dense(30, activation="relu")(hidden_layer0)
output_layer = keras.layers.Dense(10000, activation="softmax")(hidden_layer1)
model = keras.Model(inputs=inputs, outputs=output_layer)

#kompilacja
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'])

result = model.fit(X_train, y_train, batch_size=200, epochs=10, validation_data=(X_test, y_test), verbose=1)
sns.lineplot(x=result.epoch, y=result.history['loss'])
plt.show()
print(model.summary())

data2, ro2, inv_ro2 = np.load('data_test_not_teached.npy', allow_pickle=True)
data2 = data2.T

print(np.shape(data2))
print(np.shape(ro2))
print(np.shape(inv_ro2))

data_set_no = 10

inp_data = data2[data_set_no].reshape(1,-1)

y_pred = model.predict(inp_data)
print(f"Predict value: {y_pred}")
print(f"Real value: {ro2[data_set_no]}")

figure2 = plt.figure(figsize=(16,4))
ax1 = figure2.add_subplot(111)
p1, = ax1.plot(data2[data_set_no])
p2, = ax1.plot(y_pred[0]*2000, c='b')
ax1.axvline(ro2[data_set_no], c='r', linestyle='--')
plt.show()


