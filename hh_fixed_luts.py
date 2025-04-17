import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

dt = 0.01

ena, ek, el = 50, -77, -53
gna, gk, gl = 120, 36, 0.3
Cm = 1

celsius = 37
q10 = 3**(0.1*celsius - 0.63)
q10 = 1

def to_fixed(x, min_val, max_val, bitsize):
    if bitsize is None:
        return x
    scaled = (x - min_val) / (max_val - min_val) * ((1 << bitsize) - 1)
    rounded = jnp.round(scaled)
    clamped = jnp.clip(rounded, 0, (1 << bitsize) - 1)
    return clamped.astype('int32')

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

def gate_next(v, gate, alpha, beta):
    tau = 1 / (alpha(v) + beta(v)) / q10
    inf = alpha(v) / (alpha(v) + beta(v))
    return inf + (gate - inf) * jnp.exp(-dt / tau)

def simulate(volt_bits: None | int = 12, gate_bits: None | int = 12, ou_tau=1, ou_sigma=5):
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
    def n_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), n_alpha, n_beta), *GATE_TRANSFORM)
    def m_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), m_alpha, m_beta), *GATE_TRANSFORM)
    def h_next(v, gate): return to_fixed(gate_next(from_fixed(v, *VOLT_TRANSFORM), from_fixed(gate, *GATE_TRANSFORM), h_alpha, h_beta), *GATE_TRANSFORM)
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
                n_next(v, n      ),
                m_next(v,    m   ),
                h_next(v,       h),
                knext, i_next), (v, n, m, h, iapp)
    key = jax.random.PRNGKey(0)
    _, trace = jax.lax.scan(loop, initial(-64.64948, key), length=1000000)
    return trace

vbit = 12
gbit = 12
v, n, m, h, iapp = simulate(vbit, gbit)
vtrue = simulate(None, None)[0]

plt.plot(vtrue, color='black')
plt.plot(from_fixed(v, -100, 100, vbit), '--', color='red')
plt.show()

vn = jnp.zeros((1<<(vbit), 1<<(gbit)), dtype=bool).at[v, n].set(True)
vm = jnp.zeros((1<<(vbit), 1<<(gbit)), dtype=bool).at[v, m].set(True)
vh = jnp.zeros((1<<(vbit), 1<<(gbit)), dtype=bool).at[v, h].set(True)

plt.title('V N^4')
plt.imshow(vn)

plt.figure()
plt.title('V M^3')
plt.imshow(vm)

plt.figure()
plt.title('V H')
plt.imshow(vh)
plt.show()
