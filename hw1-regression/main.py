#coding=UTF-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
class linear_regression_gradient_descent:
    def __init__(self, x, y, eps, learning_rate, max_iter_times):
        '''
        initalize feature、dependent variable 、learning rate、iteration times
        :param x:
        :param y:
        :param alpha:
        :param max_iter_times:
        '''
        self.x = x
        self.y = y
        self.n = len(self.x)
        self.w = np.zeros((x.shape[1], 1))
        self.adagrad = np.zeros((x.shape[1], 1))
        self.learning_rate = learning_rate
        self.iteration = max_iter_times
        self.eps = eps
        self.cost_review = np.zeros((0, 0))

    def error_function(self):
        '''
        compute error of training data in every iteration
        :return:a vector of error
        '''
        # step1 compute cost function
        n = len(self.x)
        y_pred = np.dot(self.x, self.w)
        error = y_pred - self.y
        return error

    def partial_devative(self):
        '''
        compute the partial derivatives of cost functions on theta in every turn
        :return:
        '''
        n = len(self.x)
        error = self.error_function()
        delta_w = 2 * np.dot(self.x.T, error)
        return delta_w

    def batch_gradient_descent(self):
        '''
        gradient descent to solve the parameter of linear regression
        :return:
        '''
        n = len(self.x)
        itera = 0
        error = self.error_function()
        cost = np.sum(error ** 2) / 2 * n
        while (itera < self.iteration or cost > self.eps):
            #step1 compute the partial derivatives of cost functions on theta
            delta_w = self.partial_devative()
            #step2 update theta
            self.adagrad += delta_w ** 2
            self.w = self.w - self.learning_rate * delta_w / np.sqrt(self.adagrad + self.eps)
            #step3 compute cost function
            error = self.error_function()
            cost = np.sum(error ** 2) / 2 * n
           # print cost
            self.cost_review = np.append(self.cost_review, cost)
            itera += 1
        return self.w

if __name__=="__main__":
    raw_data = pd.read_csv('train.csv', encoding='big5')
    data = raw_data.iloc[:, 3:]
    data[data == 'NR'] = 0
    data = data.to_numpy()
    train = []
    rows_num = 0
    while rows_num < len(data):
        item = data[rows_num:rows_num+18, 0:24]
        if len(train) == 0:
            train = item
        else:
            train = np.concatenate((train, item), axis=0)
        rows_num += 18
    train = train.astype(float)
    # 归一化
    mean_x = np.mean(train, axis=1)
    std_x = np.mean(train, axis=1)
    for i in range(len(train)):
        for j in range(len(train[i])):
            if std_x[i] != 0:
                train[i][j] = (train[i][j] - mean_x[i]) / std_x[i]

    X = []
    Y = []
    for i in range(train.shape[1] - 10):
        Y.append(train[9, i+9])
        itm = np.array([1])
        for j in range(train.shape[0]):
            itm = np.concatenate((itm, train[j, i:i+9]), axis=0)
        X.append(itm)
    X = np.array(X)
    Y = np.array(Y).reshape(-1, 1)
    lr = linear_regression_gradient_descent(X, Y, learning_rate=100, eps=1e-9, max_iter_times=100)
    w = lr.batch_gradient_descent()
    np.save('weight.npy', w)
    print("parameter of linear regression:" + str(w))
    print("iteration times are:" + str(len(lr.cost_review)))
    fig = plt.figure(1)
    plt.plot(lr.cost_review, color='r')
    plt.ylim(ymin=np.min(lr.cost_review), ymax=np.max(lr.cost_review))
    plt.xlim(xmin=0, xmax=len(lr.cost_review) + 1)
    plt.ylabel("cost function")
    plt.xlabel("id of iteration")
    plt.title("cost function of linear regression")
    plt.grid()
    plt.show()
    # test_out
    test = pd.read_csv('test.csv', encodings='big5', header=None)
    test[test == 'NR'] = 0
    test = test.to_numpy()
    test = test[:, 2:].astype(float)
    i = 0
    test_X = []
    while i < test.shape[0]:
        for j in range(test.shape[1]):
            if std_x[i % 18] != 0:
                test[i][j] = (test[i][j] - mean_x[i % 18]) / std_x[i % 18]
        itm = np.array([1])
        for j in range(18):
            itm = np.concatenate((itm, test[i+j, :]), axis=0)
        test_X.append(itm)
        i += 18
    test_X = np.array(test_X)
    w = np.load('weight.npy')
    ans_y = np.dot(test_X, w)
    # write 2 csv
    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)


