import time
import jax
import jax.numpy as jnp
from jax import lax
x =jnp.arange(5)
w = jnp.array([2., 3., 4.])
def convolve(x, w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)
# print(convolve(x, w))
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])
# print(ws)

def manually_batched_convolve(xs, ws):
    output = []
    for i in range(xs.shape[0]):
        output.append(convolve(xs[i], ws[i]))
    return jnp.stack(output)
# print(manually_batched_convolve(xs, ws))

def manually_vectorized_convolve(xs, ws):
    output = []
    for i in range(1, xs.shape[-1] -1):
        output.append(jnp.sum(xs[:,i-1:i+2] * ws, axis=1))
    return jnp.stack(output, axis=1)
print(manually_vectorized_convolve(xs, ws))




# @jax.jit

# def loop_body(prev_i):
#     return prev_i +1
# def g_inner_jitted(x, n):
#     i = 0
#     while i < n:
#         i = jax.jit(loop_body)(i)
#     return x + i
# print(g_inner_jitted(10, 20))

# def body_fun(x,y):
#     return x*y, x**2+y**2
# grad_body_fun = jax.grad(body_fun)
# x = (2.)
# y = (3.)
# print((jax.grad(body_fun, argnums=(0, 1), ) (x, y)))

# def add_fun(i, x):
#     return i+1., x+1.
# print(lax.scan(add_fun, 0, jnp.array([1, 2, 3, 4])))
#
# start = time.time()
# print(jnp.add(10000, 10000))
# end = time.time()
# print("循环运行时间:%.100f秒"%(end-start))
#
# start = time.time()
# print(lax.add(10000, 10000))
# end = time.time()
# print("循环运行时间:%.100f秒"%(end-start))
