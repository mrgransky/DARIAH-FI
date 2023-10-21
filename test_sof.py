import dill
import joblib
import time
import sys
import tracemalloc

import pandas as pd
import numpy as np

def get_rnd_df(row:int=10, col:int=7): # generate random Sparse Pandas dataframe
	# print(f"[r, c]: [{row}, {col}]", end="\t")
	t=time.time()
	np.random.seed(0)
	d=np.random.randint(low=0, high=10, size=(row,col)).astype(np.float32)
	d[d < 3] = np.nan
	df=pd.DataFrame(data=d,
									index=[f"ip{i}" for i in np.random.choice(range(max(row, 10)), row, replace=False) ],
									columns=[f"col_{c}" for c in np.random.choice(range(max(col, 10)), col, replace=False) ],
									dtype=pd.SparseDtype(dtype=np.float32),
							)
	df.index.name='usr'
	# print(f"elapsed_t: {time.time()-t:.2f} sec")
	return df

def save_dill(obj):
	with open('dill_data.gz', 'wb') as f:
		dill.dump(obj, f)

def load_dill(fpath):
	with open(fpath, mode='rb') as f:
		return dill.load(f) 

def save_joblib(obj):
	with open('joblib_data.gz', 'wb') as f:
		joblib.dump(obj, f)

def load_joblib(fpath):
	with open(fpath, mode='rb') as f:
		return joblib.load(f) 

t=time.time()
tracemalloc.start()
dfs=[get_rnd_df(row=np.random.randint(low=2e3, high=3e3), col=np.random.randint(low=1e6, high=3e6)) for _ in range(int(1e+3))] # 1000 DFs
current_mem_dfs, peak_mem_dfs = tracemalloc.get_traced_memory()
print(f"Current : {current_mem_dfs / (1024 * 1024):.2f} MB | Peak: {peak_mem_dfs / (1024 * 1024):.2f} MB")
print(f"elapsed_t dfs {time.time()-t:.2f} sec for listing {len(dfs)} Pandas DFs")

tracemalloc.reset_peak()

# using dill
t=time.time()
save_dill(obj=dfs)
_, peak_save_dill = tracemalloc.get_traced_memory()
print(f"Peak memory usage for dill save: {peak_save_dill / (1024 * 1024):.2f} MB")
print(f"elapsed_t dill save {time.time()-t:.2f} sec")

tracemalloc.reset_peak()

t=time.time()
load_dill(fpath="dill_data.gz")
_, peak_load_dill = tracemalloc.get_traced_memory()
print(f"Peak memory usage for dill load: {peak_load_dill / (1024 * 1024):.2f} MB")
print(f"elapsed_t dill load {time.time()-t:.2f} sec")

tracemalloc.reset_peak()

# using joblib
t=time.time()
save_joblib(obj=dfs)
_, peak_save_joblib = tracemalloc.get_traced_memory()
print(f"Peak memory usage for joblib save: {peak_save_joblib / (1024 * 1024):.2f} MB")
print(f"elapsed_t joblib save {time.time()-t:.2f} sec")

tracemalloc.reset_peak()

t=time.time()
load_joblib(fpath="joblib_data.gz")
_, peak_load_joblib = tracemalloc.get_traced_memory()
print(f"Peak memory usage for joblib load: {peak_load_joblib / (1024 * 1024):.2f} MB")
print(f"elapsed_t joblib load {time.time()-t:.2f} sec")

# Stop tracing memory allocations
tracemalloc.stop()