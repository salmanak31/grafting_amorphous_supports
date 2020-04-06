import numpy as np
import timeit, functools

s = (15000, 70, 70)

X = np.random.randint(0,high=10,size=s)
Y = np.random.randint(0,high=10,size=s)

s2 = (s[2], 4)

X2 = np.random.randint(0,high=10,size=s2)
Y2 = np.random.randint(0,high=10,size=s2)

def matmul_p(X, Y):
    return np.diagonal(np.matmul(X, np.matmul(Y, X)), axis1=1, axis2=2)

def loop_mult(X, Y, s):
    prod = np.zeros(s)
    for i in range(s[0]):
        prod[i] = np.diagonal(X[i] @ Y[i] @ X[i])
    return prod

def loop_sub(X, s):
    t_size = 75
    training = np.zeros((t_size, s[1]))
    delta = np.zeros((s[0], t_size, s[1]))
    for i in range(s[0]):
        delta[i, :, :] = training - X[i]

    return delta

t = timeit.Timer(functools.partial(matmul_p, X, Y))
t2 = timeit.Timer(functools.partial(loop_mult, X, Y, s))

# print(X)

# print(matmul_p(X, Y))
# print(loop_mult(X, Y, s))

t3 = timeit.Timer(functools.partial(loop_sub, X2, s2))

# print('Vectorized @: {}'.format(t.timeit(2)))
print('Loop: {}'.format(t2.timeit(2)))
print('Subtractions Loop: {}'.format(t3.timeit(10)))


# print(myeinsum(X, Y))
# print(loop_mult(X, Y, 5))