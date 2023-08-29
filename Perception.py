import jax.numpy as jnp
import numpy as onp
import jax
X = [[3, 3],[4, 3],[1, 1]]
Label = [1, 1, -1]
b = 0
Alpha = 0
Eta = 1

def Perception_Model(Data, Label, Alpha, b):
    N = jnp.len(Data)
    for i in jnp.range(N):
        Model = Alpha*Label[i] *  Data + b
        if Model <=0:
            Alpha += Eta
            b += Eta * Label[j]
    return Alpha, b

def Loss():
