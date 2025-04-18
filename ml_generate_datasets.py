import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
import numpy
import jax
import jax.numpy as jnp
import lib
from lib import to_fixed, from_fixed, gate_next, n_alpha, n_beta, m_alpha, m_beta, h_alpha, h_beta, dt

@jax.jit
def LUT_n_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), n_alpha, n_beta), *GATE_TRANSFORM, rnd=False)
@jax.jit
def LUT_m_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), m_alpha, m_beta), *GATE_TRANSFORM, rnd=False)
@jax.jit
def LUT_h_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), h_alpha, h_beta), *GATE_TRANSFORM, rnd=False)

vbit = 12
gbit = 12
VOLT_TRANSFORM = -100, 100, vbit
GATE_TRANSFORM = 0, 1, gbit

_, vn_mask, vm_mask, vh_mask = lib.latest_stats()

v, g = numpy.where(vn_mask)
i = v << gbit | g
vn = jax.vmap(LUT_n_next)(v, g)
numpy.savez('dataset/n_next', v=v, g=g, i=i, o=vn)

v, g = numpy.where(vm_mask)
i = v << gbit | g
vm = jax.vmap(LUT_m_next)(v, g)
numpy.savez('dataset/m_next', v=v, g=g, i=i, o=vn)

v, g = numpy.where(vh_mask)
i = v << gbit | g
vh = jax.vmap(LUT_h_next)(v, g)
numpy.savez('dataset/h_next', v=v, g=g, i=i, o=vn)

