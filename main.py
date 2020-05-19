import numpy as np
import matplotlib.pyplot as plt
import time

def step_size(f, f_der, x, d, eps=1e-4):
    a = 1
    while f(x + a * d) > f(x) + eps * a * (-d.T @ d):
        # print(f(x + a * d), f(x) + eps * a * (-d.T @ d))
        # print(- d.T @ d)
        a = a / 2
    return a

def non_convex_step_size(f, f_der, x, d, eps=1e-4):
    a = 1
    print(f(*map(lambda x, d: x + a * d, x, d)))
    print(f(*x) + eps * a * (-d[0].T @ d[0]))
    while f(*map(lambda x, d: x + a * d, x, d)) > f(*x) + eps * a * (-d[0] @ d[0].T):
        # print(f(x + a * d), f(x) + eps * a * (-d.T @ d))
        # print(- d.T @ d)
        a = a / 2
    return a


def convex_exit_condition(f=None, x=None, new_x=None, **kwargs):
    return f(x) - f(new_x)


def non_convex_exit_condition(grad=None, **kwargs):
    return np.linalg.norm(grad)

def gradient_decent(
        f, f_der, x, tol=1e-2, iter_max=1000,
        exit_condition=convex_exit_condition, accelerate_method=None
):
    log = {}
    log['e'] = [f(x)]
    k = 0
    start_time = time.time()
    old_x = x
    while k < iter_max:
        b = 0.9 #k / (k + 3)
        if accelerate_method == 'Nestrov':
            grad = f_der(x + b * (x - old_x))
        else:
            grad = f_der(x)
        # print('x', x)
        #print('grad', grad)
        acc_term = 0
        if not accelerate_method is None:
            acc_term = x - old_x

        if not accelerate_method is None:
            a = 1e-4
        else:
            a = step_size(f, f_der, x + b * acc_term, -grad)
        # print('a', a)
        # if np.linalg.norm(grad) <= tol:
        new_x = x - a * grad + b * acc_term
        
        #print('grad', grad)
        #print('a', a)
        #print('acc_term', acc_term)
        #print('x', x)
        #print('new_x', new_x)
        #e = exit_condition(**{'f':f, 'x':x, 'new_x':new_x, 'grad':grad})
        c = np.linalg.norm(f_der(new_x))
        #print('e', e)
        if c <= tol:
            x = new_x
            break
        #min_e = min(min_e, e)
        log['e'].append(np.abs(f(x) - f(new_x)))

        k += 1
        old_x = x
        x = new_x

    log['time'] = time.time() - start_time
    return x, log


def non_gradient_decent(
        f, f_der, x, tol=1e-2, iter_max=1000,
        exit_condition=convex_exit_condition, accelerate_method=None
):
    log = {}
    log['e'] = [f(*x)]
    k = 0
    start_time = time.time()
    old_x = x
    while k < iter_max:
        b = 0.9 #k / (k + 3)
        if accelerate_method == 'Nestrov':
            grad = f_der(x + b * (x - y))
        else:
            grad = f_der(*x)
        # print('x', x)
        #print('grad', grad)
        print(f_der(*x))
        a = 1e-5
        # print('a', a)
        # if np.linalg.norm(grad) <= tol:
        new_x = list(map(lambda x, grad: x - a * grad, x, grad))

        print('grad', grad)
        #print('a', a)
        #print('acc_term', acc_term)
        #print('x', x)
        #print('new_x', new_x)
        #e = exit_condition(**{'f':f, 'x':x, 'new_x':new_x, 'grad':grad})
        print(new_x)
        c = list(map(lambda x: np.linalg.norm(x), f_der(*new_x)))
        #print('e', e)
        if any(np.array(c) <= tol):
            x = new_x
            break
        #min_e = min(min_e, e)
        log['e'].append(np.abs(f(*x) - f(*new_x)))

        k += 1
        old_x = x
        x = new_x

    log['time'] = time.time() - start_time
    return x, log



def init_convex_instance(m, n):
    A = np.random.rand(m, n)
    w = np.ones(n)
    e = np.random.rand(m)*0.1
    b = A @ w + e

    f = lambda x: np.linalg.norm(b - A @ x)
    # f = lambda x: x[0, None]*x[0, None] + 3 * x[1, None]*x[1, None]
    f_der = lambda x:  A.T @ (A @ x - b)
    # f_der = lambda x: 2 * x[0, None] + 6 * x[1, None]
    return f, f_der


def init_strongly_convex_instance(m, n, lam=0.01):
    A = np.random.rand(m, n)
    w = np.ones(n)
    e = np.random.rand(m)*0.1
    b = A @ w + e
    f = lambda w: w.T @ (A.T @ A + lam * np.identity(n)) @ w - 2 * b.T @ A @ w + b.T @ b
    f_der = lambda w,: (A.T @ A + lam * np.identity(n)) @ w - A.T @ b
    return f, f_der

def init_non_convex_instance(m, n):
    l = 5
    A = np.random.rand(m, n)
    w = np.random.rand(l, 1)
    W = np.random.rand(n, l)
    e = np.random.rand(m, 1) * 0.1
    print((A@W@w).shape, e.shape)
    b = A @ W @ w + e
    print(b.shape)
    f = lambda W, w: np.linalg.norm(b - A @ W @ w)
    def f_der(W, w):
        print(w.shape, A.T.shape, (A @ W @ w - b).shape, b.shape)
        print((W.T @ A.T @ (A @ W @ w - b)))
        return A.T @ (A @ W @ w - b) @ w.T, W.T @ A.T @ (A @ W @ w - b)
    return f, f_der

m = 1000
n = 20
convex_w_init = np.random.rand(n)
convex_f, convex_f_der = init_convex_instance(m, n)
convex_res = gradient_decent(convex_f, convex_f_der, convex_w_init, exit_condition=convex_exit_condition)

strongly_convex_w_init = np.random.rand(n)
strongly_convex_f, strongly_convex_f_der = init_strongly_convex_instance(m, n, lam=1)
strongly_convex_res = gradient_decent(strongly_convex_f,
                                      strongly_convex_f_der,
                                      strongly_convex_w_init,
                                      exit_condition=convex_exit_condition)

convex_acc_res = gradient_decent(convex_f,
                                 convex_f_der,
                                 convex_w_init,
#                                 iter_max=2,
                                 exit_condition=convex_exit_condition,
                                 accelerate_method='Nestrov')

# l = 5
# non_convex_W_init = np.zeros((n, l)) + 1
# non_convex_w_init = np.zeros((l, 1)) + 1
# non_convex_f, non_convex_f_der = init_non_convex_instance(m, n)
# non_convex_res = non_gradient_decent(non_convex_f,
#                                  non_convex_f_der,
#                                  (non_convex_W_init, non_convex_w_init),
#                                      iter_max=1,
#                                  exit_condition=non_convex_exit_condition)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(convex_res[1]['e'], label='convex')
ax.plot(strongly_convex_res[1]['e'], label='strongly convex')
ax.plot(convex_acc_res[1]['e'], label='convex + Nestrov')
# ax.plot(non_convex_res[1]['e'], label='non convex')
ax.set_yscale('log')
ax.legend()
plt.savefig('complexity.png')
# print(convex_res[0])
# print(strongly_convex_res[0])
# print(convex_acc_res[0])
# print(non_convex_res[0])


# n = 20
# times = []
# ms = [10, 100, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 100000, 1000000]
# for m in ms:
#     convex_w_init = np.random.rand(n)
#     convex_f, convex_f_der = init_convex_instance(m, n)
#     convex_res = gradient_decent(convex_f, convex_f_der, convex_w_init, exit_condition=convex_exit_condition)
#     times.append(convex_res[1]['time'])

# fig, ax = plt.subplots()
# ax.plot(ms, times)
# plt.savefig('time.png')
