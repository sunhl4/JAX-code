import time
import jax
import jax.numpy as jnp
from jax import lax

# @jax.jit
def loop_body(prev_i):
    return prev_i +1
def g_inner_jitted(x, n):
    i = 0
    while i < n:
        i = jax.jit(loop_body)(i)
    return x + i
print(g_inner_jitted(10, 20))

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
