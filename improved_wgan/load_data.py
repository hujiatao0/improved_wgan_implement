import numpy as np
import csv
import random

def load_wind():

    with open ('./datasets/wind.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype = 'float32')
    print(rows.shape)
    trX = []
    m = np.ndarray.max(rows)
    print('maximun value in rows', m)
    for i in range(rows.shape[1]):
        row = rows[:-288,i].reshape(-1, 576)
        # row /= m
        if len(trX) == 0:
            trX = row
        else:
            trX = np.concatenate((trX, row), axis = 0)
    print("shape of trX ", trX.shape)
    mean = np.mean(trX, axis=0)
    std = np.std(trX, axis=0)
    for i in range(trX.shape[0]):
      for j in range(trX.shape[1]):
        trX[i, j] = (trX[i, j] - mean[j]) / std[j]
    # print(mean.shape)
    with open('./datasets/wind label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=int)
    print('label shape', rows.shape)
    return trX, rows, mean, std



def data_generator(trX, trY):
    while True:
        data_x = []
        data_y = []
        for i in range(32):
            index = random.randint(0, trX.shape[0] - 1)
            Xs = trX[index].reshape(-1, 1, 24, 24)
            Ys = trY[index]
            Z = np.zeros((1, 5), dtype='float32')
            Z[0, Ys.item()] = 1.
            Ys = Z
            if len(data_x) == 0:
                data_x = Xs
                data_y = Ys
            else:
                data_x = np.concatenate([data_x, Xs], axis=0)
                data_y = np.concatenate([data_y, Ys], axis=0)
        yield data_x, data_y
