import pandas as pd
import time
import numpy as np
np.random.seed(19)
arr = np.zeros((1300_0000, 11))
# df1 = pd.DataFrame(234234.234324, index=range(1300_0000), columns=['id']+['a']*10)
# df2 = pd.DataFrame(np.NAN, index=0, columns=['a']*10)

hdf5_file = r'G:\NRSA_working\test_results.h5'

t1 = time.time()

for i in range(len(arr)):
    if i % 100000 == 0:
        print(i)
    line = np.random.random(11)
    arr[i] = line
arr[:, 0] = np.arange(1, len(arr) + 1, 1)
with pd.HDFStore(hdf5_file, 'a') as store: 
    df1 = pd.DataFrame(arr, columns=['id']+['a']*10)
    df1['id'] = df1['id'].astype(int)
    store.append('results', df1, index=False, append=False, complib='blosc:zstd', complevel=2)   

print(time.time() - t1)
print(arr[:10, :])
with pd.HDFStore(hdf5_file, 'r') as store:  
    df = store['results']  
    df = df.sort_values(by='id')
print(df.head(10))
