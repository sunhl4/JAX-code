import jax.numpy as jnp
import jax
from jax import random, grad
from sklearn.datasets import load_iris
data = load_iris()
iris_data = jnp.float32(data.data)
iris_target = jnp.float32(data.target)
def Dense(dense_shape = [4, 1]):
    def init_fun(input_shape = dense_shape):
        rng = random.PRNGKey(17)
        w, b = random.normal(rng, shape=input_shape), random.normal(rng, shape=(input_shape[-1],))
        return w, b
    def apply_function(inputs, params):
        w, b = params
        return jnp.dot(inputs, w) + b
    return init_fun, apply_function
init_fun, apply_function = Dense()
params = init_fun()

def loss_linear(params, x, y):
    preds = apply_function(x, params)
    return jnp.mean(jnp.power(preds - y,2.0))

learning_rate = 0.005

N = 1000

for i in range(N):
    loss = loss_linear(params, iris_data, iris_target)
    if i % 100 ==0:
        print(f'i: {i}, loss: {loss}')
    params_grad = jax.grad(loss_linear)
    params = [(p-g*learning_rate) for p, g in zip(params, params_grad(params, iris_data, iris_target))]
print(f'i: {N}, loss: {loss}')






