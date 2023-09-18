import jax
import jax.numpy as jnp
def f(x):
    return x*x*x
D_f = jax.grad(lambda x: jnp.sum(f(x)))

x = jnp.linspace(1, 5, 5)

print(D_f(x))
print(D_f1(x))
