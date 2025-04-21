# LUTs for Hodgkin Huxley

LUTs to accelerate Hodgkin Huxley simulations.


### Hodgkin-Huxley simulation

**hh.py** contains the regular hh in floating point.
We are interested in a fully LUT based representation:

```
v(t+dt) = V_LUT(v, n, m, h)
n(t+dt) = N_LUT(v, n)
m(t+dt) = M_LUT(v, h)
h(t+dt) = H_LUT(v, m)
```

We considered two fixed point LUT-based implementations:

 - **hh_fixed_luts.py**: regular state representation (vbit=12 and gbit=12)
 - **hh_fixed_luts_power.py**: solve for `n^4` and `m^3` (vbit=12, gbit=17), saving some multiplications

Given the increased costs we continued with hh_fixed_luts.py for now

### Finding don't care values

Inspired by ReducedLUT (Cassidy et al 2025), we find the 'don't care' values in the LUTs for N, M, and H with extensive simulation:


 - **luts_statistics.py**: generate 'observed pattern' masks
 - **out/*.npz**: storage of observed pattern masks as a matrix (12 bit Voltage x 12 bit Gate)
 - **plot_statistics.py**: plot masks

In general, about 80% of the LUT input patterns are not observed

### 1st approach, direct usage of ReducedLUT

First, we tried directly applying ReducedLUT

 - generate_input_files.py
 - generate_table_files.py

ReducedLUT ran out of memory on a 512GB RAM machine, even after rewriting it to handle 16-bit values instead of 64-bit.
In general the technique does not seem suited for such large LUTs.

### 2nd approach, compression using MLP

Instead, a promising approach seems to be compression using NeuraLUT (Andronic et al 2024), before applying ReducedLUT

 - **ml_generate_datasets.py**: takes the generated masks and generates three files, each containing a training set for the MLP
     - dataset/m_next.npz
     - dataset/h_next.npz
     - dataset/n_next.npz
 - **ml_train.py** trains a regular MLP on using pytorch
 - **ml_train_neuralut.py**: trains a quantized MLP using NeuraLUT
     - **neuralut_mnist_model.py**: MNIST model (regular MLP) from NeuraLUT, slightly adapted
 - **convert_neuralut_model_to_luts.py**: NeuraLUT MNIST model conversion code

## Citations

 - Cassidy, Oliver, et al. "ReducedLUT: Table Decomposition with" Don't Care" Conditions." Proceedings of the 2025 ACM/SIGDA International Symposium on Field Programmable Gate Arrays. 2025.
 - Andronic, Marta, and George A. Constantinides. "NeuraLUT: Hiding neural network density in boolean synthesizable functions." 2024 34th International Conference on Field-Programmable Logic and Applications (FPL). IEEE, 2024.
