## Standard libraries
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from tqdm.auto import tqdm
from matplotlib.colors import to_rgba


import jax
import jax.numpy as jnp

print("Using jax", jax.__version__)

a = jnp.zeros((2, 5), dtype=jnp.float32)
print(a)

b = jnp.arange(6)
print(b)

print(b.__class__)

print(b.devices())

b_cpu = jax.device_get(b)
print(b_cpu.__class__)

b_gpu = jax.device_put(b_cpu)
print(f"Device put: {b_gpu.__class__} on {b_gpu.devices()}")

print(b_cpu + b_gpu)

print(jax.devices())  # print all our available devices

b_new = b.at[0].set(1)
print("Original array:", b)
print("Changed array:", b_new)

rng = jax.random.PRNGKey(42)
# A non-desirable way of generating pseudo-random numbers...
jax_random_number_1 = jax.random.normal(rng)
jax_random_number_2 = jax.random.normal(rng)
print("JAX - Random number 1:", jax_random_number_1)
print("JAX - Random number 2:", jax_random_number_2)

# Typical random numbers in NumPy
np.random.seed(42)
np_random_number_1 = np.random.normal()
np_random_number_2 = np.random.normal()
print("NumPy - Random number 1:", np_random_number_1)
print("NumPy - Random number 2:", np_random_number_2)

rng, subkey1, subkey2 = jax.random.split(rng, num=3)  # We create 3 new keys
jax_random_number_1 = jax.random.normal(subkey1)
jax_random_number_2 = jax.random.normal(subkey2)
print("JAX new - Random number 1:", jax_random_number_1)
print("JAX new - Random number 2:", jax_random_number_2)


def simple_graph(x):
    x = x + 2
    x = x ** 2
    x = x + 3
    y = x.mean()
    return y


inp = jnp.arange(3, dtype=jnp.float32)
print("Input", inp)
print("Output", simple_graph(inp))

print(jax.make_jaxpr(simple_graph)(inp))

grad_function = jax.grad(simple_graph)
gradients = grad_function(inp)
print("Gradient", gradients)

print(jax.make_jaxpr(grad_function)(inp))

val_grad_function = jax.value_and_grad(simple_graph)
print(val_grad_function(inp))
