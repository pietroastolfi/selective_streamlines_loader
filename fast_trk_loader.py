import numpy as np
import nibabel as nib
import struct
from time import time


if __name__ == '__main__':
    filename = 'sub-100206_var-FNAL_tract.trk'

    T = nib.streamlines.load(filename, lazy_load=True)
    header_size = T.header['hdr_size']
    n_scalars = T.header['nb_scalars_per_point']
    n_properties = T.header['nb_properties_per_streamline']
    n_streamlines = T.header['nb_streamlines']
    assert(n_streamlines > 0)

    point_size = 3 + n_scalars
    point_format = 'f' * point_size

    max_n_streamlines = 100000
    
    print("LENTO")
    streamlines_length = []
    streamlines = []
    file = open(filename, 'rb')
    file.seek(header_size)
    t0 = time()
    for idx1 in range(max_n_streamlines):
        buffer = file.read(4)
        m = struct.unpack_from('i', buffer)[0]
        streamline = np.zeros((m, 3))
        for idx2 in range(m):
            buffer = file.read(point_size * 4)
            streamline[idx2, :] = struct.unpack_from(point_format, buffer)[:3]

        streamlines.append(streamline)

    print(time() - t0)
    file.close()

    print("MEDIO")
    file = open(filename, 'rb')
    file.seek(header_size)
    streamlines = []
    t0 = time()
    for idx1 in range(max_n_streamlines):
        buffer = file.read(4)
        m = struct.unpack_from('i', buffer)[0]
        streamline_n_floats = ((point_size * m) + n_properties)
        streamline_format = 'f' * streamline_n_floats
        buffer = file.read(streamline_n_floats * 4)
        tmp = struct.unpack_from(streamline_format, buffer)
        streamline = np.array(tmp).reshape(m, 3)  # assume n_properties is 0 and n_scalars is 0
        
        streamlines.append(streamline)

    print(time() - t0)
    file.close()

    print("VELOCE")
    file = open(filename, 'rb')
    file.seek(header_size)
    streamlines = []
    t0 = time()
    for idx1 in range(max_n_streamlines):
        buffer = file.read(4)
        m = struct.unpack_from('i', buffer)[0]
        streamline_n_floats = ((point_size * m) + n_properties)
        streamline = np.fromfile(file, np.float32, streamline_n_floats)
        streamline.resize(m, 3) # assume n_properties is 0 and n_scalars is 0
        
        streamlines.append(streamline)

    print(time() - t0)
    file.close()
