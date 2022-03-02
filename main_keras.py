import csv

import GPy
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential, Model
from sklearn.model_selection._split import KFold

import numpy as np

attNum = 21
epochs = 200
batch_size = 10
n_folds = 10
nHidden = 20
feature_layer_nHidden = 20
turnNum = 10

# The values of first column are heat transfer coefficients. 2~22 columns represent condition values. 
data = np.loadtxt("input.csv", delimiter='\t', dtype='float32')
X = data[:, 1:]
y = data[:, 0]
size = y.size


def loss_function(y_true, y_pred):
    return K.mean(((y_pred - y_true) / y_true) ** 2)


predicts = np.zeros(size)
variances = np.zeros(size)

for turn in range(turnNum):
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    predict_shuffle = []
    variance_shuffle = []
    expect_shuffle = []
    junban = [0 for i in range(size)]
    count = 0
    for train_index, test_index in kfold.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        mean_on_train = x_train.mean(axis=0)
        std_on_train = x_train.std(axis=0)
        x_train_scaled = (x_train - mean_on_train) / std_on_train
        x_test_scaled = (x_test - mean_on_train) / std_on_train
        
        for i in test_index:
            junban[i] = count
            count += 1
            
        expect_shuffle.extend(y_test)
    
        base_model = Sequential()
        base_model.add(Dense(nHidden, activation="relu", input_shape=(attNum,)))
        base_model.add(Dense(nHidden, activation="relu"))
        base_model.add(Dense(nHidden, activation="relu"))
        base_model.add(Dense(feature_layer_nHidden, activation="relu", name='feature_layer'))
        base_model.add(Dense(1, activation='linear'))
        
        base_model.compile(loss=loss_function, optimizer='adam')   
        base_model.fit(x_train_scaled, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
        
        deep_model = Model(inputs=base_model.input, outputs=base_model.get_layer('feature_layer').output)
    
        kernel = GPy.kern.RBF(input_dim=feature_layer_nHidden) + GPy.kern.Bias(input_dim=feature_layer_nHidden)
        gauss_model = GPy.models.GPRegression(deep_model.predict(x_train_scaled), y_train.reshape(-1, 1), kernel=kernel)
        gauss_model.optimize()
        
        gauss_input = deep_model.predict(x_test_scaled)
        mean, var = gauss_model.predict(gauss_input)
        predict_shuffle.extend(mean.reshape(-1))
        variance_shuffle.extend(var.reshape(-1))
     
    expect = np.zeros(size)
    for i in range(size):
        predicts[i] += predict_shuffle[junban[i]] / variance_shuffle[junban[i]]
        variances[i] += 1 / variance_shuffle[junban[i]]
        expect[i] = expect_shuffle[junban[i]]

for i in range(size):
    predicts[i] = predicts[i] / variances[i]
    variances[i] = 1 / variances[i]
    
sdvalue = 0.0
for i in range(size):
    sdvalue += ((predicts[i] - expect[i]) / expect[i]) ** 2
    
sdvalue = 100 * ((sdvalue / size) ** 0.5)
print(sdvalue)

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(expect)
    writer.writerow(predicts)
    writer.writerow(variances)

