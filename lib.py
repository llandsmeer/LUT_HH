import jax
import jax.numpy as jnp
import glob
import numpy as np
import matplotlib.pyplot as plt

def latest_stats():
    files = glob.glob('out/mask*.npz')
    max_num = max(int(f[-7:-4]) for f in files)
    filename = f'out/mask{max_num:03}.npz'
    data = np.load(filename)
    vn_mask = data['vn_mask']
    vm_mask = data['vm_mask']
    vh_mask = data['vh_mask']

    return max_num + 1, vn_mask, vm_mask, vh_mask

def to_fixed(x, min_val, max_val, bitsize, rnd=True):
    if bitsize is None:
        return x
    scaled = (x - min_val) / (max_val - min_val) * ((1 << bitsize) - 1)
    if rnd:
        rounded = jnp.round(scaled)
        clamped = jnp.clip(rounded, 0, (1 << bitsize) - 1)
        return clamped.astype('int32')
    else:
        rounded = scaled
        clamped = jnp.clip(rounded, 0, (1 << bitsize) - 1)
        return clamped

def from_fixed(y, min_val, max_val, bitsize):
    if bitsize is None:
        return y
    max_int = (1 << bitsize) - 1
    return y / max_int * (max_val - min_val) + min_val

def exprelr(x): return jax.lax.select(jnp.isclose(x, 0), jnp.ones_like(x), x / jnp.expm1(x))
def m_alpha(V): return exprelr(-0.1*V - 4.0)
def h_alpha(V): return 0.07*jnp.exp(-0.05*V - 3.25)
def n_alpha(V): return 0.1*exprelr(-0.1*V - 5.5)
def m_beta(V):  return 4.0*jnp.exp(-(V + 65.0)/18.0)
def h_beta(V):  return 1.0/(jnp.exp(-0.1*V - 3.5) + 1.0)
def n_beta(V):  return 0.125*jnp.exp(-0.0125*V - 0.8125)

dt = 0.01


celsius = 37
q10 = 3**(0.1*celsius - 0.63)
q10 = 1
def gate_next(v, gate, alpha, beta):
    tau = 1 / (alpha(v) + beta(v)) / q10
    inf = alpha(v) / (alpha(v) + beta(v))
    return inf + (gate - inf) * jnp.exp(-dt / tau)
