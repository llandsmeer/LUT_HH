import sys
sys.path.append('/home/llandsmeer/repos/llandsmeer/reducedhh')

import numpy

import tqdm

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import lib
from lib import to_fixed, from_fixed, gate_next, n_alpha, n_beta, m_alpha, m_beta, h_alpha, h_beta, dt

ena, ek, el = 50, -77, -53
gna, gk, gl = 120, 36, 0.3
Cm = 1

def simulate(volt_bits: None | int = 12, gate_bits: None | int = 12, ou_tau=1, ou_sigma=5, *, seed):
    VOLT_TRANSFORM = -100, 100, volt_bits
    GATE_TRANSFORM = 0, 1, gate_bits
    # LUTS
    def v_next(v, n, m, h, iapp):
        v = from_fixed(v, *VOLT_TRANSFORM)
        n = from_fixed(n, *GATE_TRANSFORM)
        m = from_fixed(m, *GATE_TRANSFORM)
        h = from_fixed(h, *GATE_TRANSFORM)
        ina = gna * m**3 * h*(v-ena)
        ik  = gk * n**4 * (v-ek)
        ileak = gl * (v-el)
        itot = iapp - ina - ik - ileak
        v_next = v + dt * itot / Cm
        return to_fixed(v_next, *VOLT_TRANSFORM)
    def LUT_n_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), n_alpha, n_beta), *GATE_TRANSFORM)
    def LUT_m_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), m_alpha, m_beta), *GATE_TRANSFORM)
    def LUT_h_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), h_alpha, h_beta), *GATE_TRANSFORM)
    ###
    def initial(v, key):
        n_inf = n_alpha(v) / (n_alpha(v) + n_beta(v))
        m_inf = m_alpha(v) / (m_alpha(v) + m_beta(v))
        h_inf = h_alpha(v) / (h_alpha(v) + h_beta(v))
        v = to_fixed(v, *VOLT_TRANSFORM)
        n_inf = to_fixed(n_inf, *GATE_TRANSFORM)
        m_inf = to_fixed(m_inf, *GATE_TRANSFORM)
        h_inf = to_fixed(h_inf, *GATE_TRANSFORM)
        return v, n_inf, m_inf, h_inf, key, 0.
    alpha = jnp.exp(-dt / ou_tau).astype(jnp.float32)
    beta = (ou_sigma * jnp.sqrt(dt)).astype(jnp.float32)
    def loop(state, _):
        v, n, m, h, key, iapp = state
        key, knext = jax.random.split(key, 2)
        w = jax.random.normal(key, dtype=jnp.float32)
        i_next = iapp * alpha + beta * w
        return (v_next(v, n, m, h, iapp),
                LUT_n_next(v, n      ),
                LUT_m_next(v,    m   ),
                LUT_h_next(v,       h),
                knext, i_next), (v, n, m, h, iapp)
    key = jax.random.PRNGKey(seed)
    _, trace = jax.lax.scan(loop, initial(-64.64948, key), length=1000000)
    return trace

vbit = 12
gbit = 12

seed_start = 0
try:
    seed_start, vn_mask, vm_mask, vh_mask = lib.latest_stats()
    vn_mask = jnp.array(vn_mask)
    vm_mask = jnp.array(vm_mask)
    vh_mask = jnp.array(vh_mask)
except Exception as ex:
    print('Could not load previous masks. Start again? [y/N]')
    if input() == 'y':
        vn_mask = jnp.zeros((1<<(vbit), 1<<(gbit)), dtype=bool)
        vm_mask = jnp.zeros((1<<(vbit), 1<<(gbit)), dtype=bool)
        vh_mask = jnp.zeros((1<<(vbit), 1<<(gbit)), dtype=bool)
    else:
        exit(1)

sim = jax.jit(lambda seed: simulate(vbit, gbit, seed=seed))
for seed in tqdm.tqdm(range(100000)):
    seed = seed + seed_start
    v, n, m, h, iapp = sim(seed)
    vn_mask = vn_mask.at[v, n].set(True)
    vm_mask = vm_mask.at[v, m].set(True)
    vh_mask = vh_mask.at[v, h].set(True)
    print(seed, vn_mask.sum(), vm_mask.sum(), vh_mask.sum(), (1<<vbit) * (1<<gbit))
    if seed % 100 == 0:
        numpy.savez(f'out/mask{seed:03d}', vn_mask=vn_mask, vm_mask=vm_mask, vh_mask=vh_mask)
