import jax.numpy as jnp
from jax import random


weight = jnp.array([[1.7, 1.7],[2.14,2.14]])
mat_a = jnp.array([[1],[2]])
bias = 0.99
#print (jnp.matmul(weight, mat_a) + bias)

def Dense (dense_shape = [2, 1]):
    rng = random.PRNGKey(17)
    mat_a = random.normal (rng, shape = dense_shape)
    bias = random.normal(rng, shape = (dense_shape[-1],) )
    params = [mat_a, bias]

    def init_parm():
        return params
    def apply_function(inputs, params = params):
        mat_a, b = params
        return jnp.dot(inputs,mat_a) +b
    return apply_function
weight = jnp.array([[1.7, 1.7],[2.14,2.14]])
res = Dense()(weight)
#print (res)

from sklearn.datasets  import load_iris



