import sys
import nibabel as nib
import os
import struct
import numpy as np
from time import time


if __name__ == '__main__':
    
    # trk_fn = '/Users/pietroastolfi/Desktop/toy_data/tractograms/sub-627549/sub-627549_var-FNAL_tract.trk'
    trk_fn = 'sub-100206_var-FNAL_tract.trk'

    print(trk_fn)
    print("Reading header with nibabel")
    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']

    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    point_bytes = 4 * point_size
    properties_bytes = n_properties * 4

    print("Parsing lenghts of %s streamlines" % nb_streamlines)
    t0 = time()
    lengths = np.empty(nb_streamlines, dtype=np.int)
    if float(sys.version[:3]) > 3.2:
        with open(trk_fn, 'rb') as f:
            f.seek(header_size)
            for idx in range(nb_streamlines):
                l = int.from_bytes(f.read(4), byteorder='little')
                lengths[idx] = l
                jump = point_bytes * l + properties_bytes
                f.seek(jump, 1)
    else:
        with open(trk_fn, 'rb') as f:
            f.seek(header_size)
            for idx in range(nb_streamlines):
                l = np.fromfile(f, np.int32, 1)[0]
                lengths[idx] = l
                jump = point_bytes * l + properties_bytes
                f.seek(jump, 1)

    print("%s sec." % (time() - t0))

    # position in bytes where to find a given streamline in the TRK file:
    index_bytes = lengths * point_bytes + properties_bytes + length_bytes
    index_bytes = np.concatenate([[length_bytes], index_bytes[:-1]]).cumsum() + header_size
    # n_floats = lengths * point_size + n_properties
    n_floats = lengths * point_size  # better because it skips properties, if they exist


    idxs = np.random.choice(500000, 200000, replace=True)
    idxs.sort()

    print("Extracting %s streamlines with the desired id" % len(idxs))
    t0 = time()
    streamlines = []
    with open(trk_fn, 'rb') as f:
        for idx in idxs:
            # move to the position initial position of the coordinates
            # of the streamline:
            f.seek(index_bytes[idx])
            # Parse the floats:
            s = np.fromfile(f, np.float32, n_floats[idx])
            s.resize(lengths[idx], point_size)
            # remove scalars if present:
            if n_scalars > 0:
                s = s[:, :3]

            streamlines.append(s)

        print("%s sec." % (time() - t0))

    # OLD CODE:
    # M = lengths.sum()  ## total number of points in the file
    # streams = np.empty((M * point_size), dtype=np.float32)

    # print("Extracting streamlines with the desired id")
    # t0 = time()
    # with open(trk_fn, 'rb') as f:
    #     f.seek(header_size)
    #     j = 0
    #     for idx in idxs:
    #         # jump considers streamlines points and lenghts before idx
    #         jump = lengths[:idx].sum() * point_bytes + 4 * (idx + n_properties)
    #         f.seek(jump,1)

    #         l = lengths[idx]
    #         buffer = f.read(4)
    #         print(idx)
    #         assert l == struct.unpack_from('i', buffer)[0]
    #         streamline_n_floats = (l * point_size + n_properties)
    #         # next step assume n_properties is 0 and n_scalars is 0
    #         streams[j:j+streamline_n_floats] = np.fromfile(f, \
    #                     np.float32, streamline_n_floats)

    #         j = j+streamline_n_floats
    #     print(streams.resize(M,3).shape)
    #     print(M)

    # print(time() - t0)
    # f.close()
