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

def to_int(x, min_val, max_val, bitsize):
    scaled = (x - min_val) / (max_val - min_val) * ((1 << bitsize) - 1) * 0.5
    rounded = jnp.round(scaled)
    clamped = jnp.clip(rounded, 0, (1 << bitsize) - 1)
    return clamped.astype('int32')

def exprelr(x): return jax.lax.select(jnp.isclose(x, 0), jnp.ones_like(x), x / jnp.expm1(x))
def m_alpha(V): return exprelr(-0.1*V - 4.0)
def h_alpha(V): return 0.07*jnp.exp(-0.05*V - 3.25)
def n_alpha(V): return 0.1*exprelr(-0.1*V - 5.5)
def m_beta(V):  return 4.0*jnp.exp(-(V + 65.0)/18.0)
def h_beta(V):  return 1.0/(jnp.exp(-0.1*V - 3.5) + 1.0)
def n_beta(V):  return 0.125*jnp.exp(-0.0125*V - 0.8125)

# LUTS

def v_next(v, n, m, h, iapp):
    ina = gna * m**3 * h*(v-ena)
    ik  = gk * n**4 * (v-ek)
    ileak = gl * (v-el)
    itot = iapp - ina - ik - ileak
    return v + dt * itot / Cm

def n_next(v, gate):
    tau = 1 / (n_alpha(v) + n_beta(v)) / q10
    inf = n_alpha(v) / (n_alpha(v) + n_beta(v))
    return inf + (gate - inf) * jnp.exp(-dt / tau)

def m_next(v, gate):
    tau = 1 / (m_alpha(v) + m_beta(v)) / q10
    inf = m_alpha(v) / (m_alpha(v) + m_beta(v))
    return inf + (gate - inf) * jnp.exp(-dt / tau)

def h_next(v, gate):
    tau = 1 / (h_alpha(v) + h_beta(v)) / q10
    inf = h_alpha(v) / (h_alpha(v) + h_beta(v))
    return inf + (gate - inf) * jnp.exp(-dt / tau)

###

def initial(v, key):
    n_inf = n_alpha(v) / (n_alpha(v) + n_beta(v))
    m_inf = m_alpha(v) / (m_alpha(v) + m_beta(v))
    h_inf = h_alpha(v) / (h_alpha(v) + h_beta(v))
    return v, n_inf, m_inf, h_inf, key, 0.

tau = 1
sigma = 5
alpha = jnp.exp(-dt / tau).astype(jnp.float32)
beta = (sigma * jnp.sqrt(dt)).astype(jnp.float32)

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
v, n, m, h, iapp = trace

plt.plot(iapp)
plt.show()

VBIT = 10
NBIT = 10

v = to_int(v, -100, 100, VBIT)
n = to_int(n**4, 0, 1, NBIT)
m = to_int(m**3, 0, 1, NBIT)
h = to_int(h, 0, 1, NBIT)


vn = jnp.zeros((1<<(VBIT-1), 1<<(NBIT-1)), dtype=bool).at[v, n].set(True)
vm = jnp.zeros((1<<(VBIT-1), 1<<(NBIT-1)), dtype=bool).at[v, m].set(True)
vh = jnp.zeros((1<<(VBIT-1), 1<<(NBIT-1)), dtype=bool).at[v, h].set(True)

plt.title('V N^4')
plt.imshow(vn)

plt.figure()
plt.title('V M^3')
plt.imshow(vm)

plt.figure()
plt.title('V H')
plt.imshow(vh)
plt.show()
