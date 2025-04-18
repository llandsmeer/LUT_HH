import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp


import numpy
import jax
import tqdm
import jax.numpy as jnp
from lib import to_fixed, from_fixed, gate_next, n_alpha, n_beta, m_alpha, m_beta, h_alpha, h_beta, dt

vbit = 12
gbit = 12

VOLT_TRANSFORM = -100, 100, vbit
GATE_TRANSFORM = 0, 1, gbit

@jax.jit
def LUT_n_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), n_alpha, n_beta), *GATE_TRANSFORM)
@jax.jit
def LUT_m_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), m_alpha, m_beta), *GATE_TRANSFORM)
@jax.jit
def LUT_h_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), h_alpha, h_beta), *GATE_TRANSFORM)

i = jnp.arange(1<<(vbit+gbit))
v = i >> gbit
g = i & ((1 << gbit) - 1)

vn = jax.vmap(LUT_n_next)(v, g)
numpy.savetxt('reduced/table_n_next.txt', vn, fmt='%x', newline='\n')
del vn

vm = jax.vmap(LUT_m_next)(v, g)
numpy.savetxt('reduced/table_m_next.txt', vm, fmt='%x', newline='\n')
del vm

vh = jax.vmap(LUT_h_next)(v, g)
numpy.savetxt('reduced/table_h_next.txt', vh, fmt='%x', newline='\n')
del vh
