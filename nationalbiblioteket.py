import numpy as np
import pandas as pd

import re
import string
import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt

files_list = ["digi_hakukaytto_v1.csv", 
								"digi_nidekaytto_v1.csv",
								"digi_sivukaytto_v1.csv"]

usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket/no_ip_logs',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket/no_ip_logs",
				}

dpath = usr_[os.environ['USER']]
print(dpath)
#sys.exit()

search_idx, volume_idx, page_idx = 0, 1, 2

#chk_sz = 21e6

"""
print(f">> Reading {os.path.join(dpath, files_list[page_idx])} in {chk_sz} chunks...")
page_chnk = pd.read_csv(os.path.join(dpath, files_list[page_idx]), 
													low_memory=False, 
													chunksize=chk_sz,
													iterator=True,
													)
page_list = []
for chunk in page_chnk:
	print(chunk.shape)
	page_list.append(chunk)

page_df = pd.concat(page_list, axis=0)
del page_list

print(page_df.shape)
print(page_df.head())
print(page_df.info())
print("#"*100)

print(f">> Reading {os.path.join(dpath, files_list[search_idx])} ...")
search_df = pd.read_csv(os.path.join(dpath, files_list[search_idx]), 
												low_memory=False,
												)

print(search_df.shape)
print(search_df.head())
print(search_df.info())
print("#"*100)

print(f">> Reading {os.path.join(dpath, files_list[volume_idx])} ...")
volume_df = pd.read_csv(os.path.join(dpath, files_list[volume_idx]), 
												low_memory=False,
												)
print(volume_df.shape)
print(volume_df.head())
print(volume_df.info())
print("#"*100)

sys.exit(0)

print(df.shape)
cols = list(df.columns)
print( len(cols), cols )
print("#"*100)
"""

def get_df(idx, custom_chunk_size=None):
	fname = os.path.join(dpath, files_list[idx])
	print(f">> Reading {fname} ...")

	if custom_chunk_size:
		print(f">> into {custom_chunk_size} chunks...")
		
		chnk_df = pd.read_csv(fname, 
													low_memory=False, 
													chunksize=custom_chunk_size,
													iterator=True,
													)
		mylist = []
		for ch in chnk_df:
			print(ch.shape)
			mylist.append(ch)

		df=pd.concat(mylist, axis=0)
		del mylist
		return df
	else:
		df = pd.read_csv(	fname,
											low_memory=False,
											)
		return df

def save_dfs():
	print(">> Saving...")

	#page_df = get_df(idx=page_idx, custom_chunk_size=23e6)
	page_df = get_df(idx=page_idx)
	volume_df = get_df(idx=volume_idx)
	search_df = get_df(idx=search_idx)

	dfs_dict = {
		"search":	search_df,
		"vol":	volume_df,
		"pg":		page_df,
	}

	fname = "_".join(list(dfs_dict.keys()))+"_dfs.dump"
	print(f">> Dumping {fname} ...")
	joblib.dump(	dfs_dict, 
								os.path.join( dpath, f"{fname}" ),
								compress='lz4', # zlib more info: https://joblib.readthedocs.io/en/latest/auto_examples/compressors_comparison.html#sphx-glr-auto-examples-compressors-comparison-py
								)
	fsize = os.stat( os.path.join( dpath, f"{fname}" ) ).st_size / 1e6

	print(f">> Dumping Done => size: {fsize:.1f}MB ...")


def load_dfs(fpath=""):
	print(f">> Loading {fpath} ...")
	d = joblib.load(fpath)
	print(list(d.keys()))

	search_df = d["search"]
	volume_df = d["vol"]
	search_df = d["pg"]

	print(f"\n>> Search DF: {search_df.shape}...")
	print(search_df.head(20))
	print("#"*100)

	print(f"\n>> Volume DF: {volume_df.shape}...")
	print(volume_df.head(20))
	print("#"*100)

	print(f"\n>> Page DF: {page_df.shape}...")
	print(page_df.head(20))
	print("#"*100)

	print(f">> LOADING COMPLETED!")

def main():
	save_dfs()
	load_dfs( fpath=os.path.join(dpath, "search_vol_pg_dfs.dump") )

if __name__ == '__main__':
	os.system('clear')
	main()
	sys.exit(0)


name_dict = {	0: ["search_usage_id", 
									"material_type", 
									"date", 
									"search_phrase", 
									"date_start", 
									"date_end", 
									"publication_places",
									"publishers",
									"titles",
									"languages",
									"pages",
									"scores", #TODO: must be verified: Finnish: 'tulokset'
									"rights",
									"fuzzy", #TODO: must be verified: Finnish: 'sumea'
									"illustrations",
									"index_prefix",
									"tags",
									"authors",
									"collections",
									"type",
									"text_meta",
									"text_ocr",
									"clip_keywords", #TODO: must be verified: Finnish: 'leike_asiasanat'
									"clip_categories",
									"clip_subjects",
									"clip_generated",
									"duration_ms",
									"order",
									"require_all_words",
									"find_volumes",
									"last_page",
									"no_access_results",
									"import_time",
									"import_start_date",
									],
							1: ["volume_usage_id", 
									"volume_id", 
									"date", 
									"referer", 
									"robot", 
									"access_grounds", 
									"access_grounds_details", 
									"user_agent" ],
							2: ["first", "2nd", "3rd"],}

df.columns = name_dict.get(qidx)

print(df.head(20))
print("#"*100)

cols_new = list(df.columns)
print( len(cols_new), cols_new )
print("#"*100)
#sys.exit(0)

print(df.info())
print("#"*100)
print(f"\n>> How many nan?")
print(df.isna().sum())

"""
print(f"\n>> Cleaning ...")
df_clean = df.dropna(axis=0,
										how="any",
										)

print(df_clean.shape)
print(df_clean.head(10))
"""
