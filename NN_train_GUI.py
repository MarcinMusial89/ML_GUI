import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
import seaborn as sns
import pickle

root = tk.Tk()
root.title('ML_GUI')
root.geometry('1200x800')
d97_file = 'null'
pp = 'null'
sp = 'null'

date = datetime.datetime.now()
date = date.strftime('%m_%d_%H_%M_%S')

toolbar = tk.Frame(root)
toolbar.pack(side=tk.TOP, fill=tk.X)

testButt = tk.Frame(root)
testButt.pack(side=tk.TOP, fill=tk.X)

plot_area = tk.Frame(root)
plot_area.pack(side=tk.BOTTOM, fill=tk.X)


def browse_file ():
    global data_file
    data_file = filedialog.askopenfilename(multiple = False, title = 'Select data file',filetypes = (("npy files","*npy"),))
    file_path = data_file.split('.')[0]
    data = np.load(data_file, allow_pickle=True)

    global data_f
    data_f = data[0].T
    global ro
    ro = data[1]
    global inv_ro
    inv_ro = data[2]

    
    info_out.config(text= 'Data sets in file: ' + str(np.shape(data_f)[0]))
    global info_print
    info_print = 'Data sets in file: ' + str(np.shape(data_f)[0])

def get_force_set ():
    fs = 0
    fs = force_set.get()
    return fs
    
def check_LR():
    if data_file == 'null':
        messagebox.showinfo('File error', 'Select file')
    else:
        for widgets in plot_area.winfo_children():
            widgets.destroy()
        X_train, X_test, y_train, y_test = train_test_split(data_f, ro, test_size=.1, random_state=0)

        global model
        model = LinearRegression()
        model.fit(X_train, y_train)
        infoLR = tk.Label(plot_area, text='Model score:')
        infoLR.config(text='Model score: ' + str(round((model.score(X_test, y_test))*100,2)) + '%')
        infoLR.pack(side=tk.TOP, padx=2, pady=2)
        print("Score:")
        print(model.score(X_test, y_test))
        
def NN():

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=1000, activation = 'relu'))
    model.add(keras.layers.Dense(units=30, activation = 'relu'))
    model.add(keras.layers.Dense(units=10000, activation = 'softmax'))
    return model

def check_NN():
    for widgets in plot_area.winfo_children():
        widgets.destroy()
    ro_vec = np.zeros((len(ro), 10000))
    for i in range(len(ro)):
        ro_vec[i][int(ro[i])] = 1

    max_force = 2500

    data_norm = data_f / max_force

    X_train_NN1, X_val_NN1, y_train_NN1, y_val_NN1 = train_test_split(data_norm, ro_vec, test_size=0.2,
                                                                      random_state=True)
    global model
    model = NN()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0005),
                  metrics=['categorical_accuracy'])
    history_NN = model.fit(x=X_train_NN1, y=y_train_NN1, batch_size=200, epochs=20000,
                           validation_data=(X_val_NN1, y_val_NN1), verbose=1)
    figure1 = plt.Figure(figsize=(20, 10), dpi=100)
    ax1 = figure1.add_subplot(111)
    line = FigureCanvasTkAgg(figure1, plot_area)
    line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    sns.lineplot(x=history_NN.epoch, y=history_NN.history['loss'], ax=ax1)
    plt.close()

    
def model_LR():
    for widgets in plot_area.winfo_children():
        widgets.destroy()
    set_no = get_force_set()
    set_no = int(set_no)

    filename = filedialog.askopenfilename(multiple = False, title = 'Select model file',filetypes = (("pickle files","*pickle"),))
    model = pickle.load(open(filename, "rb"))

    inp_data = data_f[set_no].reshape(1, -1)
    y_pred = model.predict(inp_data)
    print(y_pred)
    x = data_f[set_no]
    x2 = ro[set_no]
    x3 = inv_ro[set_no]

    figure1 = plt.Figure(figsize=(20, 10), dpi=100)
    ax1 = figure1.add_subplot(111)
    line = FigureCanvasTkAgg(figure1, plot_area)
    line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    p1, = ax1.plot(x)
    ax1.axvline(y_pred, c='orange', label='LR output')
    ax1.axvline(x2)
    ax1.axvline(x3)
    plt.close()



def model_NN():
    for widgets in plot_area.winfo_children():
       widgets.destroy()
    set_no = get_force_set()
    set_no = int(set_no)

    ro_vec = np.zeros((len(ro), 10000))
    for i in range(len(ro)):
        ro_vec[i][int(ro[i])] = 1
    max_force = 2500
    data_norm = data_f / max_force

    path_NN = filedialog.askopenfilename(multiple = False, title = 'Select model file',filetypes = (("h5 files","*h5"),))
    model_NN = keras.models.load_model(path_NN)
    model_NN.evaluate(data_norm, ro_vec)
    y_pred_NN = model_NN.predict(data_norm)

    x = data_norm[set_no]
    x2 = ro[set_no]
    x3 = inv_ro[set_no]


    figure1 = plt.Figure(figsize=(20, 10), dpi=100)
    ax1 = figure1.add_subplot(111)
    line = FigureCanvasTkAgg(figure1, plot_area)
    line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    p1, = ax1.plot(x)
    p2, = ax1.plot(y_pred_NN[0,:], c='orange', label='NN output')
    ax1.axvline(x2)
    ax1.axvline(x3)
    plt.close()


def save_model():
    filename = filedialog.asksaveasfilename(title='Save NN model', filetypes=(("h5 files", "*h5"),))
    keras.models.save_model(model, filename)

def save_modelRL():
    filename = filedialog.asksaveasfilename(title = 'Save LR model',filetypes = (("pickle files","*pickle"),))
    pickle.dump(model, open(filename, "wb"))


def plot():
    for widgets in plot_area.winfo_children():
       widgets.destroy()
    set_no = get_force_set()
    set_no = int(set_no)
    
    x = data_f[set_no]
    x2 = ro[set_no]
    x3 = inv_ro[set_no]
    
    figure1 = plt.Figure(figsize=(20, 10), dpi=100)
    ax1 = figure1.add_subplot(111)
    line = FigureCanvasTkAgg(figure1, plot_area)
    line.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)  
    p1, = ax1.plot(x)
    ax1.axvline(x2)
    ax1.axvline(x3)
    plt.close()

    
    
file_button = tk.Button(master = toolbar, 
                     command = browse_file,
                     height = 1, 
                     width = 20,
                     text = "Select data file")

info_out = tk.Label(toolbar, text='Data sets in file: ')
info = tk.Label(toolbar, text ='Choose set to plot:' )
input_text = tk.StringVar()
force_set = ttk.Entry(toolbar, width=8, textvariable = input_text, justify = tk.CENTER)



plot_button = tk.Button(master = toolbar, 
                     command = plot,
                     height = 1, 
                     width = 20,
                     text = "Plot data sample")




button_quit = tk.Button(toolbar, text='Quit', height = 1, width=10, command = root.destroy)



file_button.pack(side= tk.LEFT, padx=2, pady=2)
info_out.pack(side= tk.LEFT, padx=2, pady=2)
info.pack(side= tk.LEFT, padx=2, pady=2)
force_set.pack(side= tk.LEFT, padx=2, pady=2)
plot_button.pack(side= tk.LEFT, padx=2, pady=2)


tlrButt = tk.Button(testButt, text = 'Train LinReg', command=check_LR)
tlrButt.pack(side= tk.LEFT, padx=2, pady=2)
tnnButt = tk.Button(testButt, text = 'Train NN', command=check_NN)
tnnButt.pack(side= tk.LEFT, padx=2, pady=2)

mlrButt = tk.Button(testButt, text = 'Model LinReg', command=model_LR)
mlrButt.pack(side= tk.LEFT, padx=2, pady=2)
mnnButt = tk.Button(testButt, text = 'Model NN', command=model_NN)
mnnButt.pack(side= tk.LEFT, padx=2, pady=2)

saveButt = tk.Button(testButt, text = 'Save model NN', command=save_model)
saveButt.pack(side= tk.RIGHT, padx=2, pady=2)

save2Butt = tk.Button(testButt, text = 'Save model RL', command=save_modelRL)
save2Butt.pack(side= tk.RIGHT, padx=2, pady=2)

button_quit.pack(side= tk.RIGHT, padx=2, pady=2)





root.mainloop()

    
