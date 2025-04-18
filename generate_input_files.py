import sys
sys.path.append('/home/llandsmeer/repos/llandsmeer/reducedhh')
import lib

import numpy

def savebin(fname, arr):
    with open(fname, 'w') as f:
        for num in arr:
            f.write('{0:024b}\n'.format(num))

_, vn_mask, vm_mask, vh_mask = lib.latest_stats()

v, g = numpy.where(vn_mask)
i = v << 12 | g
savebin('reduced/input_n_next.txt', i)

v, g = numpy.where(vm_mask)
i = v << 12 | g
savebin('reduced/input_m_next.txt', i)

v, g = numpy.where(vh_mask)
i = v << 12 | g
savebin('reduced/input_h_next.txt', i)
