import nibabel as nib
import os
import struct
import numpy as np
from time import time

idxs = range(0,10000,10)
idxs.sort()

trk_fn = '/Users/pietroastolfi/Desktop/toy_data/tractograms/sub-627549/sub-627549_var-FNAL_tract.trk'
lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
header = lazy_trk.header
header_size = header['hdr_size']
nb_streamlines = header['nb_streamlines']
n_scalars = header['nb_scalars_per_point']
n_properties = header['nb_properties_per_streamline']
point_size = 3 + n_scalars
point_bytes = 4 * point_size

t0 = time()
lengths = []
with open(trk_fn, 'rb') as f:
    f.seek(header_size)
    for idx in range(nb_streamlines):
        buffer = f.read(4)
        l = struct.unpack_from('i', buffer)[0]
        lengths.append(l)
        jump = point_bytes * l + n_properties * 4
        f.seek(jump,1)
    print(time() - t0)

lengths = np.array(lengths, dtype=np.int32)
M = np.array(lengths[idxs]).sum()
streams = np.empty((M * point_size), dtype=np.float32)
with open(trk_fn, 'rb') as f:
    f.seek(header_size)
    streamlines = []
    j = 0
    for idx in idxs:
        # jump considers streamlines points and lenghts before idx
        jump = lengths[:idx].sum() * point_bytes + 4 * (idx + n_properties)
        f.seek(jump,1)

        l = lengths[idx]
        buffer = f.read(4)
        print(idx)
        assert l == struct.unpack_from('i', buffer)[0]
        streamline_n_floats = (l * point_size + n_properties)
        # next step assume n_properties is 0 and n_scalars is 0
        streams[j:j+streamline_n_floats] = np.fromfile(f, 
                    np.float32, streamline_n_floats)
        
        j = j+streamline_n_floats
    print(streams.resize(M,3).shape)
    print(M)
print(time() - t0)
f.close()
