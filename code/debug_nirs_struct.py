import scipy.io as io
import numpy as np

# Load NIRS data
nirs_cnt = io.loadmat(r'd:\workspace\data\Simultaneous EEG&NIRS\VP001-NIRS\cnt_nback.mat')
nirs_data = nirs_cnt['cnt_nback'][0, 0]

print("Available fields in NIRS struct:")
print(nirs_data.dtype.names)
print("\nField details:")
for field in nirs_data.dtype.names:
    try:
        data = nirs_data[field]
        if isinstance(data, np.ndarray):
            print(f"  {field}: shape={data.shape}, dtype={data.dtype}")
        else:
            print(f"  {field}: {type(data)}")
    except Exception as e:
        print(f"  {field}: Error - {e}")
