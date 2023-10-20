import dill
import joblib
import time

import pandas as pd
import numpy as np

def get_rnd_df(row:int=10, col:int=7): # generate random Sparse Pandas dataframe
	print(f"[r, c]: [{row}, {col}]", end="\t")
	t=time.time()
	np.random.seed(0)
	d=np.random.randint(low=0, high=10, size=(row,col)).astype(np.float32)
	d[d < 3] = np.nan
	df=pd.DataFrame(data=d,
									index=[f"ip{i}" for i in np.random.choice(range(max(row, 10)), row, replace=False) ],
									columns=[f"col_{c}" for c in np.random.choice(range(max(col, 10)), col, replace=False) ],
									dtype=pd.SparseDtype(dtype=np.float32), # sparse: memory efficient xxx but SUPER SLOW xxx
							)
	df.index.name='usr'
	print(f"elapsed_t: {time.time()-t:.2f} sec")
	return df

df1=get_rnd_df(row=int(1e+3), col=int(2e+3)) # resembles my real data
df2=get_rnd_df(row=int(1e+3), col=int(2e+3)) # resembles my real data

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
dfs=[get_rnd_df(row=int(9*i), col=int(40*i)) for i in range(1, int(1e+3))]
print(f"elapsed_t dfs{time.time()-t:2f} sec | {len(dfs)} pandas DFs")

# using dill
t=time.time()
save_dill(obj=dfs)
print(f"elapsed_t dill save{time.time()-t:2f} sec")

t=time.time()
load_dill(fpath="dill_data.gz")
print(f"elapsed_t dill load{time.time()-t:2f} sec")

# using joblib
t=time.time()
save_joblib(obj=dfs)
print(f"elapsed_t joblib save{time.time()-t:2f} sec")

t=time.time()
load_joblib(fpath="joblib_data.gz")
print(f"elapsed_t joblib load{time.time()-t:2f} sec")
