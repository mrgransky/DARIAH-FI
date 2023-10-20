import dill
import joblib

import pandas as pd
import numpy as np

def get_rnd_df(row:int=10, col:int=7): # generate random Sparse Pandas dataframe
	np.random.seed(0)
	d=np.random.randint(low=0, high=10, size=(row,col)).astype(np.float32)
	d[d < 3] = np.nan
	df=pd.DataFrame(data=d,
									index=[f"ip{i}" for i in np.random.choice(range(max(row, 10)), row, replace=False) ],
									columns=[f"col_{c}" for c in np.random.choice(range(max(col, 10)), col, replace=False) ],
									dtype=pd.SparseDtype(dtype=np.float32), # sparse: memory efficient xxx but SUPER SLOW xxx
							)
	df.index.name='usr'
	return df

df1=get_rnd_df(row=int(1e+3), col=int(2e+3)) # resembles my real data
df2=get_rnd_df(row=int(1e+3), col=int(2e+3)) # resembles my real data

def save_dill(obj):
	with open('dill_data.gz', 'wb') as f:
		dill.dump(obj, f)

def load_dill(fpath)
	with open(fpath, mode='rb') as f:
		return dill.load(f) 


def save_joblib(obj):
	with open('joblib_data.gz', 'wb') as f:
		joblib.dump(obj, f)

def load_joblib(fpath)
	with open(fpath, mode='rb') as f:
		return joblib.load(f) 

dfs=[get_rnd_df(row=int(20*i), col=int(40*i)) for i in range(1e+3)]

# using dill
save_dill(obj=dfs)
load_dill(fpath="dill_data.gz")

# using joblib
save_joblib(obj=dfs)
load_joblib(fpath="joblib_data.gz")