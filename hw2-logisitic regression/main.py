# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

titles = {}
mean = {}
std = {}
to_one_titles = ['age', 'detailed industry recode', 'detailed occupation recode', 'wage per hour', 'capital gains',
                 'capital losses', 'dividends from stocks', 'num persons worked for employer',
                 'own business or self employed', 'veterans benefits', 'weeks worked in year', 'year']
ignore_titles = ['wage per hour', 'migration code-change in msa', 'migration code-change in reg',
                 'migration code-move within reg', 'migration prev res in sunbelt', 'own business or self employed']

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    # np.mean
    # axis = 0作用于行，即计算每列的平均值
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

def pre_process():
    data = pd.read_csv('./data/train.csv', encoding='UTF8')
    data = data.iloc[:, 1:]
    train_data = numpy.ones((len(data), 1))  # initial bias
    label = numpy.zeros((len(data), 1))
    for column in data.columns:
        print("正在处理%s行数据。。。" % column)
        if column in to_one_titles:
            mean[column] = np.mean(data[column])
            std[column] = np.std(data[column])
            tmp = (data[column] - mean[column]) / (std[column] + 1e-8)
            tmp = np.array(tmp).reshape(-1, 1)
            train_data = np.concatenate((train_data, tmp), axis=1)
        elif column in ignore_titles:
            continue
        elif column == 'y':
            for i in range(len(data)):
                if data[column][i] == ' 50000+.':
                    label[i] = 1.
        else:  # one-hot
            titles[column] = data[column].unique()
            title_column = data[column]
            for title in titles[column]:
                tmp = np.zeros((len(data), 1))
                for i in range(len(data)):
                    if title_column[i] == title:
                        tmp[i] = 1.
                train_data = np.concatenate((train_data, tmp), axis=1)

    print(train_data.shape)
    return train_data, label


# define function
def _shuffle(X, Y):
    # np.arange:
    # Return evenly spaced values within a given interval
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize], Y[randomize]


def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w):
    return _sigmoid(np.matmul(X, w))


def _predict(X, w):
    return np.round(_f(X, w)).astype(np.int)


def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


def _cross_entropy_loss(y_pred, Y_label):
    cross_entropy = -np.matmul(Y_label.T, np.log(y_pred + 1e-9)) - np.matmul((1 - Y_label).T, np.log(1 - y_pred + 1e-9))
    return cross_entropy[0, 0]


def _gradient(X, Y_label, w):
    # np.sum:
    # 当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
    # 当axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列
    y_pred = _f(X, w)
    pred_error = Y_label - y_pred
    gradient = -np.matmul(X.T, pred_error)
    return gradient


def train():
    X_train, Y_train = pre_process()
    w = np.zeros((X_train.shape[1], 1))
    # adagrad = np.zeros((X_train.shape[1], 1))
    leanring_rate = 0.2
    iter_time = 10
    batch_size = 8
    step = 1
    # record loss
    train_loss = []
    train_acc = []
    for t in range(iter_time):
        X_train, Y_train = _shuffle(X_train, Y_train)
        # mini-batch
        for idx in range(int(np.floor(len(X_train) / batch_size))):
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

            gradient = _gradient(X, Y, w)
            w = w - leanring_rate / np.sqrt(step) * gradient
            step = step + 1

        y_train_pred = _f(X_train, w)
        y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(y_train_pred, Y_train))
        train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / X_train.shape[0])

    print('Training loss: {}'.format(train_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    _plot(train_loss, train_acc)


def _plot(train_loss, train_acc):
    plt.plot(train_loss)
    plt.plot(train_acc)
    plt.title('Loss And Accuracy')
    plt.legend(['loss', 'acc'])
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
