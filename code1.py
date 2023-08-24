import jax.numpy as jnp
import jax
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
import tensorflow as tf
import tensorflow_datasets as tfds


x_train = jnp.load("mnist_train_x.npy")
y_train = jnp.load("mnist_train_y.npy")
def one_hot_nojit (x, k=10, dtype=jnp.float32):
    return jnp.array(x[:,None] == jnp.arange(k), dtype)

def Selu (x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
key = random.PRNGKey(17)
x = random.normal(key,(5,))

init_random_params, predict = stax.serial(
    stax.Dense(1024), stax.Relu,
    stax.Dense(1024), stax.Relu,
    stax.Dense(10), stax.LogSoftmax)

mat_a = jnp.array([[1.7, 1.7],[2.14,2.14]])
weight = jnp.array([[1],[2]])
bias = 0.99
print (jnp.matmul(mat_a, weight) + bias)

if  __name__ == '__main__':
    x = jnp.array([1, 2, 3])
    x = one_hot_nojit(x)
    print(x)
#    print (type(y_train))
#    print(shape(y_train))
#    print(tfds.__version__)
#   tfds.list_builders()