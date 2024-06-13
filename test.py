from pyRPC3 import RPC3, Channel_Class
import numpy as np
from math import sin, cos

sig = RPC3('')

force1 = np.array([2*sin(i)*cos(i/4) for i in range(10000)])
force2 = force1*2
channel1 = Channel_Class(1, 'Force', 'N', 1, 1e-3)
channel1.value = force1
channel2 = Channel_Class(2, 'Force2', 'N', 1, 1e-3)
channel2.value = force2

# Save rsp file
sig.save('OutputSignal.rsp', 1e-3, [channel1, channel2])
