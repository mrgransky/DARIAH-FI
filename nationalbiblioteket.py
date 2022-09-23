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

# more info for adjustment of rcparams:
# https://matplotlib.org/stable/tutorials/introductory/customizing.html
sz=12
params = {
	'figure.figsize':	(sz*1.4, sz*1.0),  # W, H
	'figure.dpi':		200,
	'figure.autolayout': True,
	#'figure.constrained_layout.use': True,
	'legend.fontsize':	sz*0.8,
	'axes.labelsize':	sz*0.2,
	'axes.titlesize':	sz*0.2,
	'xtick.labelsize':	sz*1.0,
	'ytick.labelsize':	sz*1.0,
	'lines.linewidth' :	sz*0.1,
	'lines.markersize':	sz*0.8,
	'font.size':		sz*1.0,
	'font.family':		"serif",
}
pylab.rcParams.update(params)

sns.set(font_scale=1.3, 
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


languages={"FINNISH": False, "ENGLISH": True}

dpath = usr_[os.environ['USER']]
rpath = os.path.join( dpath[:dpath.rfind("/")], f"results")
search_idx, volume_idx, page_idx = 0, 1, 2

if not os.path.exists(rpath): 
	print(f"\n>> Creating DIR:\n{rpath}")
	os.makedirs( rpath )

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
								"results",
								"rights",
								"fuzzy_search", 
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
								"require_all_search_terms",
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

def basic_visualization(df, name=""):
	print(f">> Visualizing missing data of {name} ...")

	print(f">>>>> Barplot >>>>>")
	g = sns.displot(
			data=df.isna().melt(value_name="Missing"),
			y="variable",
			hue="Missing",
			multiple="stack",
			height=16,
			#kde=True,
			aspect=1.2,
	)
	g.set_axis_labels("Samples", "Features")
	for axb in g.axes.ravel():
		# add annotations
		for c in axb.containers:
			# custom label calculates percent and add an empty string so 0 value bars don't have a number
			labels = [f"{(v.get_width()/df.shape[0]*100):.1f} %" if v.get_width() > 0 else "" for v in c]
			axb.bar_label(c,
										labels=labels,
										label_type='edge',
										#fontsize=13,
										rotation=0,
										padding=5,
										)
			break; # only annotate the first!
		axb.margins(y=0.3)
	plt.savefig(os.path.join( rpath, f"{name}_missing_barplot.png" ), )
	plt.clf()

	print(f">>>>> Heatmap >>>>>")
	f, ax = plt.subplots()
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
	plt.clf()

	print(f">>>>> Histogram >>>>>")
	hist = df.hist()
	plt.suptitle(f"Histogram {name} Data")
	plt.savefig(os.path.join( rpath, f"{name}_histogram.png" ), )
	plt.clf()

def get_df(idx, adjust_cols=True, keep_original=False):
	fname = os.path.join(dpath, files_list[idx])
	print(f">> Reading {fname} ...")

	df = pd.read_csv(fname, low_memory=False,)
	
	if ('kielet' in list(df.columns)) and (keep_original==False):
		df['kielet'] = df['kielet'].str.replace(' ','', regex=True)
		df['kielet'] = df['kielet'].str.replace('FIN','FI', regex=True)
		df['kielet'] = df['kielet'].str.replace('SWE','SE', regex=True)
		df['kielet'] = df['kielet'].str.replace('ENG','EN', regex=True)
		df['kielet'] = df['kielet'].str.replace('RUS','RU', regex=True)
		
		df['kielet'] = df['kielet'].str.replace('SE,FI','FI,SE', regex=True)
		df['kielet'] = df['kielet'].str.replace('FI,FI','FI', regex=True)
		df['kielet'] = df['kielet'].str.replace('FIU,FI','FI,FIU', regex=True)
		df['kielet'] = df['kielet'].str.replace('SE,SE','SE', regex=True)
		df['kielet'] = df['kielet'].str.replace('EN,FI','FI,EN', regex=True)
		df['kielet'] = df['kielet'].str.replace('EN,SE','SE,EN', regex=True)
		df['kielet'] = df['kielet'].str.replace('RU,FI','FI,RU', regex=True)
		df['kielet'] = df['kielet'].str.replace('RU,EN','EN,RU', regex=True)



	if adjust_cols:
		df.columns = name_dict.get(idx)
	
	return df

def save_dfs(qlang="ENGLISH"):
	rename_columns = languages[qlang]
	print(f">> Saving in {qlang} => rename_columns: {rename_columns} ...")

	search_df = get_df(idx=search_idx, adjust_cols=rename_columns)
	page_df = get_df(idx=page_idx, adjust_cols=rename_columns)
	volume_df = get_df(idx=volume_idx, adjust_cols=rename_columns)

	dfs_dict = {
		"search":	search_df,
		"vol":	volume_df,
		"pg":		page_df,
	}

	fname = "_".join(list(dfs_dict.keys()))+f"_dfs_{qlang}.dump"
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
	print(s_df.info(verbose=True, memory_usage='deep'))
	#print("/"*150)
	#print(s_df.dtypes)
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
	print( s_df["fuzzy_search"].value_counts() )
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
	print( s_df["require_all_search_terms"].value_counts() )

	print("-"*150)
	print( s_df["no_access_results"].value_counts() )

	"""
	print(f"\n>> Volume_DF: {v_df.shape} Tot. missing data: {v_df.isnull().values.sum()}")
	print( list(v_df.columns ) )
	print()
	print(v_df.head(5))
	print(v_df.isna().sum())
	#print(v_df.dtypes)
	print(v_df.info(verbose=True, memory_usage='deep'))
	print("#"*130)

	print(f"\n>> Page_DF: {p_df.shape} Tot. missing data: {p_df.isnull().values.sum()}")
	print( list(p_df.columns ) )
	print()
	print(p_df.head(5))
	print(p_df.isna().sum())
	print(p_df.info(verbose=True, memory_usage='deep'))
	#print(p_df.dtypes)
	print("#"*130)
	"""

	print(f"\n>> LOADING COMPLETED in {elapsed_t:.2f} ms!")
	print(f"\nSearch_DF: {s_df.shape} Volume_DF: {v_df.shape} Page_DF: {p_df.shape}")
	return s_df, v_df, p_df

def plt_bar(df, name="", N=40):
	"""
	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(17,7))
	#fig, axs = plt.subplots(nrows=1, ncols=2)

	clrs = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

	axs[0].pie(gender_counts, 
						labels=gender_ung, 
						autopct='%1.1f%%', 
						startangle=90,
						colors=clrs)
	axs[0].axis('equal')
	axs[0].set_title(f"Gender Distribution")
	"""

	plt.figure()
	df["languages"].value_counts().sort_values(ascending=False)[:N].plot(	kind="barh", 
																																				title=f"{N} most frequent Language in NLF search Engine")
	plt.savefig(os.path.join( rpath, f"{name}_lang.png" ), )
	plt.ylabel("Languages")
	plt.xlabel("Counts")
	
	plt.clf()

def main():
	# rename_columns: True: saving doc in english
	# rename_columns: False: saving doc in Finnish (Original) => no modification!

	#QUERY_LANGUAGE = "FINNISH"
	QUERY_LANGUAGE = "ENGLISH"

	save_dfs(qlang=QUERY_LANGUAGE)

	search_df, vol_df, pg_df = load_dfs( fpath=os.path.join(dpath, f"search_vol_pg_dfs_{QUERY_LANGUAGE}.dump") )

	plt_bar( search_df, name=f"search_{QUERY_LANGUAGE}" )



	#basic_visualization(search_df, name=f"search_{QUERY_LANGUAGE}")
	#basic_visualization(vol_df, name=f"volume_{QUERY_LANGUAGE}")
	#basic_visualization(pg_df, name=f"page_{QUERY_LANGUAGE}")

if __name__ == '__main__':
	os.system('clear')
	main()
	sys.exit(0)
