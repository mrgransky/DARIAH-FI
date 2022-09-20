import re
import string
import os
import sys
import joblib
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

sz=15
params = {
	'figure.figsize':		(int(sz*1.4), int(sz*1.0)), # W, H
	'figure.dpi':				400,
	'legend.fontsize':	int(sz*1.0),
	'axes.labelsize':		int(sz*1.0),
	'axes.titlesize':		int(sz*1.1),
	'xtick.labelsize':	int(sz*1.0),
	'ytick.labelsize':	int(sz*1.0),
	'lines.linewidth' :	int(sz*0.1),
	'lines.markersize':	int(sz*0.8),
	'font.family':			"serif",
	'figure.constrained_layout.use': True,
}
pylab.rcParams.update(params)

sns.set(font_scale=1.5, 
				style="white", 
				palette='deep', 
				font="serif", 
				color_codes=True,
				)

files_list = ["digi_hakukaytto_v1.csv", 
							"digi_nidekaytto_v1.csv",
							"digi_sivukaytto_v1.csv",
							]

usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket/no_ip_logs',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket/no_ip_logs",
				}

dpath = usr_[os.environ['USER']]
rpath = os.path.join( dpath[:dpath.rfind("/")], "results")

if not os.path.exists(rpath): 
	print(f"\n>> Creating DIR:\n{rpath}")
	os.makedirs( rpath )

#sys.exit()

search_idx, volume_idx, page_idx = 0, 1, 2

name_dict = {
	search_idx: [	"search_usage_id", 
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
	volume_idx: ["volume_usage_id", 
								"volume_id", 
								"date", 
								"referer", 
								"robot", 
								"access_grounds", 
								"access_grounds_details", 
								"user_agent",
							],
	page_idx: 	["page_usage_id", 
								"page_id", 
								"date", 
								"referer", 
								"robot", 
								"access_grounds", 
								"access_grounds_details", 
								"user_agent", 
							],
							}

def visuzalize_nan(df, name=""):
	print(f">> Visualizing missing data of {name} ...")

	print(f">>>>> Heatmap >>>>>")
	ax = sns.heatmap(
			df.isna(),
			cmap=sns.color_palette("Greys"),
			cbar_kws={'label': 'NaN (Missing Data)', 'ticks': [0.0, 1.0]},
			)

	ax.set_ylabel(f"Samples\n\n{df.shape[0]}$\longleftarrow${0}")
	ax.set_yticks([])
	ax.xaxis.tick_top()
	ax.tick_params(axis='x', labelrotation=90)
	plt.suptitle(f"Missing {name} Data (NaN)")
	plt.savefig(os.path.join( rpath, f"{name}_missing_heatmap.png" ), )

	print(f">>>>> Barplot >>>>>")
	g = sns.displot(
			data=df.isna().melt(value_name="NaN"),
			y="variable",
			hue="NaN",
			multiple="stack",
			height=12,
			#kde=True,
			aspect=1.3,
	)
	g.set_axis_labels("Counts", "Features")
	plt.savefig(os.path.join( rpath, f"{name}_missing_barplot.png" ), )

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
		df.columns = name_dict.get(idx)
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

	print(f">> Dumping Done => size: {fsize:.1f} MB ...")

def load_dfs(fpath=""):
	fsize = os.stat( fpath ).st_size / 1e9
	print(f">> Loading {fpath} | size: {fsize:.1f} GB ...")
	st_t = time.time()

	d = joblib.load(fpath)

	elapsed_t = time.time() - st_t

	s_df = d["search"]
	v_df = d["vol"]
	p_df = d["pg"]
	elapsed_t = time.time() - st_t

	print(f"\n>> Search_DF: {s_df.shape} Tot. missing data: {s_df.isnull().values.sum()}")
	print( list(s_df.columns ) )
	print()
	print(s_df.head(25))
	print(s_df.isna().sum())
	#print(s_df.info(verbose=True, memory_usage='deep'))
	print("#"*130)

	print( s_df["material_type"].value_counts() )
	print("-"*150)
	print( s_df["publishers"].value_counts() )
	print("-"*150)
	print( s_df["publication_places"].value_counts() )
	print("-"*150)
	print( s_df["languages"].value_counts() )
	print("-"*150)
	print( s_df["rights"].value_counts() )
	print("-"*150)
	print( s_df["fuzzy"].value_counts() )
	print("-"*150)
	print( s_df["illustrations"].value_counts() )
	print("-"*150)
	print( s_df["tags"].value_counts() )

	print("-"*150)
	print( s_df["authors"].value_counts() )

	print("-"*150)
	print( s_df["collections"].value_counts() )

	print("-"*150)
	print( s_df["type"].value_counts() )

	print("-"*150)
	print( s_df["require_all_words"].value_counts() )

	print("-"*150)
	print( s_df["no_access_results"].value_counts() )

	"""
	print(f"\n>> Volume_DF: {v_df.shape} Tot. missing data: {v_df.isnull().values.sum()}")
	print( list(v_df.columns ) )
	print()
	print(v_df.head(5))
	#print(v_df.isna().sum())
	print(v_df.info(verbose=True, memory_usage='deep'))
	print("#"*130)

	print(f"\n>> Page_DF: {p_df.shape} Tot. missing data: {p_df.isnull().values.sum()}")
	print( list(p_df.columns ) )
	print()
	print(p_df.head(5))
	#print(p_df.isna().sum())
	print(p_df.info(verbose=True, memory_usage='deep'))
	print("#"*130)
	"""
	print(f"\n>> LOADING COMPLETED in {elapsed_t:.2f} ms!")
	print(f"\nSearch_DF: {s_df.shape} Volume_DF: {v_df.shape} Page_DF: {p_df.shape}")
	return s_df, v_df, p_df

def main():
	#save_dfs()
	search_df, vol_df, pg_df = load_dfs( fpath=os.path.join(dpath, "search_vol_pg_dfs.dump") )
	visuzalize_nan(search_df, name="search")
	visuzalize_nan(vol_df, name="volume")
	visuzalize_nan(pg_df, name="page")

if __name__ == '__main__':
	os.system('clear')
	main()
	sys.exit(0)
