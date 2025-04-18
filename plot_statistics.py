import sys
sys.path.append('/home/llandsmeer/repos/llandsmeer/reducedhh')

import matplotlib.pyplot as plt

import lib

_, vn_mask, vm_mask, vh_mask = lib.latest_stats()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, mask, title in zip(axes, [vn_mask, vm_mask, vh_mask], ['vn_mask', 'vm_mask', 'vh_mask']):
    im = ax.imshow(mask)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
