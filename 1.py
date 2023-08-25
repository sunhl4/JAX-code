import jax.numpy as jnp
import jax
from jax import random, grad
from sklearn.datasets import load_iris
data = load_iris()
iris_data = jnp.float32(data.data)
iris_target = jnp.float32(data.target)
iris_data = jax.random.shuffle(random.PRNGKey(17), iris_data)
iris_target = jax.random.shuffle(random.PRNGKey(17), iris_target)
k = 3

def one_hot_nojit (x, k=k, dtype=jnp.float32):
    return jnp.array(x[:,None] == jnp.arange(k), dtype)

iris_target = one_hot_nojit(iris_target)
#print (iris_target)
def Dense():
    def apply_function(inputs, params = params):
        w, b = params
        return jnp.dot(inputs, w) + b
    return apply_function

def Selu (x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

def softmax(x, axis = -1):
    unnormalized = jnp.exp(x)
    return unnormalized / unnormalized.sum(axis, keepdims=True)

def cross_entropy(y_true, y_pred):
    y_true = jnp.array(y_true)
    y_pred = jnp.array(y_pred)
    res = -jnp.sum(y_true * jnp.log(y_pred + 1e-7), axis = -1)
    return res
#    return round(res, 3):把res的值取小数点后

def mlp(x, params):
    a0, b0, a1, b1 = params
    x = Dense()(x, [a0, b0])
    x = jax.nn.tanh(x) #第一层激活函数
    x = Dense()(x, [a1, b1])
    x = softmax(x, axis = -1) #第二层激活函数
    return x

def loss_mlp(params, x, y):
    preds = mlp(x, params)
    loss_value = cross_entropy(y, preds)
    return jnp.mean(loss_value)
rng = jax.random.PRNGKey(17)
a0 = jax.random.normal(rng, shape = [4, 5])
b0 = jax.random.normal(rng, shape = (5,))
a1 = jax.random.normal(rng, shape = [5, k])
b1 = jax.random.normal(rng, shape = (k,))

params = [a0, b0, a1, b1]
learning_rate = 2.17e-4
N = 20000

for i in range(N):

    loss = loss_mlp(params, iris_data, iris_target)
    if i % 1000 ==0:
        predict_result = mlp(iris_data, params)
        predict_class = jnp.argmax(predict_result, axis=1)
        # print(predict_result)
        _iris_target = jnp.argmax(iris_target, axis=1)
        accuracy = jnp.sum(predict_class == _iris_target) / (len(_iris_target))
        print(f'i: {i}, loss: {loss},accuracy: {accuracy}')
        #另一种方式：print("i:"i, "loss:" loss,"accuracy:" accuracy)
    params_grad = jax.grad(loss_mlp)
    params = [(p-g*learning_rate) for p, g in zip(params, params_grad(params, iris_data, iris_target))]
print(f'i: {N}, loss: {loss},accuracy: {accuracy}')



#现成的softmax方法 x = jax.nn.softmax(x)








