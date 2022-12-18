import numpy as np
import random


def main_function(seq: list, window_size: int, m: int, error: float, max_iterations: int, alpha: float, predict: int):
    x, y = create_x_y(seq, window_size)
    x = extend_matrix(x, m)
    w1, w2 = create_weights(window_size, m)
    w1, w2 = learning(error, max_iterations, x, w1, w2, y, alpha)
    out = to_predict(w1, w2, predict, m, x, y)
    return out


def create_x_y(seq: list, p: int):
    one = 1
    x = []
    y = []
    i = 0
    while i + p< len(seq):
        a = []
        for j in range(p):
            a.append(seq[j + i])
        x.append(a)
        y.append(seq[i + p])
        i = i + one

    x = np.array(x)
    y = np.array(y)
    return x, y


def extend_matrix(x, m):
    matrix = fill_with_zeros(len(x), m)
    x = np.append(x, matrix, axis=1)
    return x


def create_weights(p, m):
    w1 = fill_with_zeros(p + m, m)
    w2 = fill_with_zeros(m, 1)
    for i in range(p + m):
        for j in range(m):
            w1[i][j] = random.random()
    for i in range(m):
        w2[i] = random.random()
    # w2 = numpy.random.rand(m, 1)
    # print(wq)
    return w1, w2
####################


def fill_with_zeros(x, y):
    X = np.zeros((x, y))
    return X


def leaky_relu(x):
    return max(0.1*x, x)


def activation_function(x):
    for i in range(len(x[0])):
        x[0][i] = leaky_relu(x[0][i])
    return x


def function_der(x):
    rangge = range(len(x[0]))
    for i in rangge:
        # x[0][i] = 1/((x[0][i] ** 2 + 1)**(0.5))
        x[0][i] = 1
    return x


def w1_count(w1, alpha, delta, z, w2):
    w = w1 - alpha * delta * z.T @ w2.T * function_der(z @ w1)
    return w


def w2_count(w2, alpha, delta, h):
    w = w2 - alpha * delta * h.T * function_der(h @ w2)
    return w


def learning(error, n, x, w1, w2, y, alpha):
    one = 1
    this_error = 1
    k = 1
    while error <= this_error and k <= n:
        this_error = 0
        for i in range(len(x)):
            z = fill_with_zeros(one, len(x[i]))
            for j in range(len(x[i])):
                z[0][j] = x[i][j]
            h = activation_function(z @ w1)
            out = activation_function(h @ w2)
            delta = out - y[i]
            w1 = w1_count(w1, alpha, delta, z, w2)
            w2 = w2_count(w2, alpha, delta, h)
            this_error = this_error + (delta ** 2)[0] / 2
        print("%d: %s" % (k, this_error))
        k = k + 1

    return w1, w2


def to_predict(w1, w2, predict, m, x, y):
    context = np.reshape(y[-1], 1)
    X = x[-1, :-m]
    out = []
    amo = range(predict)
    for i in amo:
        X = X[1:]
        train = np.concatenate((X, context))
        X = np.concatenate((X, context))
        train = np.append(train, np.array([0] * m))
        h = train @ w1
        output = h @ w2
        context = output
        out.append(output[0])
    return out



def choose_seq(seq, window_size, m, error, max_iterations, alpha, predict):
    print("Choose seq:")
    ind = 1
    for i in seq:
        print(ind, "->", i)
        ind = ind + 1
    choose_seq = input()

    if int(choose_seq) == 1:
        print(main_function(seq[0], window_size, m, error, max_iterations, alpha, predict))
    elif int(choose_seq) == 2:
        print(main_function(seq[1], window_size, m, error, max_iterations, alpha, predict))
    elif int(choose_seq) == 3:
        print(main_function(seq[2], window_size, m, error, max_iterations, alpha, predict))
    elif int(choose_seq) == 4:
        print(main_function(seq[3], window_size, m, error, max_iterations, alpha, predict))


def main():
    window_size = 5
    m = 2
    error = 0.0000001
    max_iterations = 50000
    alpha = 0.000000005
    predict = 5
    seq1 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    seq2 = [1, 3, 9, 27, 74, 152, 456, 1368]
    seq3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seq4 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    seq = [seq1, seq2, seq3, seq4]
    choose_seq(seq, window_size, m, error, max_iterations, alpha, predict)


if __name__ == "__main__":
    main()
