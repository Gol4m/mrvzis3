import numpy as np
import random


def main_function(seq: list, window_size: int, m: int, error: float, max_iterations: int, alpha: float, predict: int):
    xl, yl = list(), list()
    x, y = create_x_y(seq, window_size, xl, yl)
    x = extend_matrix(x, m)
    w1, w2 = create_weights(window_size, m)
    w1, w2 = learning(error, max_iterations, x, w1, w2, y, alpha, 1, 1)
    out = list()
    out = to_predict(w1, w2, predict, m, y, out, 0)
    return out


def create_np_arrays(x, y):
    X = np.array(x)
    Y = np.array(y)
    return X, Y


def iteration(i):
    i1 = i + 1
    return i1


def create_x_y(seq: list, p: int, xl: list, yl: list):
    i = 0
    while i + p < len(seq):
        l = list()
        for j in range(p):
            l.append(seq[j + i])
        xl.append(l)
        yl.append(seq[i + p])
        i = iteration(i)
    return create_np_arrays(xl, yl)



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


def count_error(this_error, delta):
    new_error = this_error + (delta ** 2)[0] / 2
    return new_error


def get_delta(out, y, i):
    delta = out - y[i]
    return delta


def learn(z, w1, w2, y, i, alpha, this_error):
    h, out = activation_function(multiply_matrix(z, w1)), activation_function(multiply_matrix(h, w2))
    delta = get_delta(out, y, i)
    w11, w22 = w1_count(w1, alpha, delta, z, w2), w2_count(w2, alpha, delta, h)
    this_errorr = count_error(this_error, delta)
    return h, out, delta, w11, w22, this_errorr


def learning(error, n, x, w1, w2, y, alpha, this_error, k):
    while error <= this_error:
        while k <= n:
            this_error = 0
            for i in range(len(x)):
                z = fill_with_zeros(1, len(x[i]))
                for j in range(len(x[i])):
                    z[0][j] = x[i][j]
                h, out, delta, w1, w2, this_error = learn(z, w1, w2, y, i, alpha, this_error)
            print("%d: %s" % (k, this_error))
            k = iteration(k)
    return w1, w2


def multiply_matrix(X, Y):
    Q = X @ Y
    return Q


def for_x1(m):
    ind = 1
    new_X = list()
    for _ in range(m):
        new_X.extend(-1, :-m)
        ind = ind + 1
    return new_X


def for_x2(X):
    ind = 1
    new_X = list()
    for _ in X:
        new_X.append(X[ind])
        ind = ind + 1
    return new_X




def to_predict(w1, w2, predict, m, y, out, i):
    context = np.reshape(y[-1], 1)
    X = for_x1(X, m)
    while i != range(predict):
        X = for_x2(X)
        X, train = np.concatenate((X, context)), np.concatenate((X, context))
        train = np.append(train, np.array([0] * m))
        h, output = multiply_matrix(train, w1), multiply_matrix(h, w2)
        out.append(output[0])
        i = iteration(i)
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
