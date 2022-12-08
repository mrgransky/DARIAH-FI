import re
import string
import os
import sys
import joblib
import time
import argparse
import glob

from natsort import natsorted
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

from utils import *
from wordcloud import WordCloud

import matplotlib
matplotlib.use("Agg")

parser = argparse.ArgumentParser(description='National Library of Finland (NLF) Data Analysis')
parser.add_argument('--query', default=0, type=int) # smallest
args = parser.parse_args()

sz=13 # >>>>>>>>> 12 original <<<<<<<<<<<
params = {
	'figure.figsize':	(sz*1.7, sz*1.0),  # W, H
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

def get_query_dataframe(QUERY=0):
	#print(f"\nGiven log file index: {QUERY}")
	all_dump_file_paths = natsorted( glob.glob( os.path.join(dfs_path, "*.dump") ) )
	#print(all_dump_file_paths)

	all_files = [dfile[dfile.rfind("/")+1:dfile.rfind(".dump")] for dfile in all_dump_file_paths]
	#print(all_files)

	query_dataframe_file = all_files[QUERY] 
	result_directory = os.path.join(rpath, query_dataframe_file)
	make_folder(folder_name=result_directory)
	print(f"Input Query: {QUERY}")

	return query_dataframe_file

def get_result_directory(QUERY=0):
	#print(f"\nGiven log file index: {QUERY}")
	all_dump_file_paths = natsorted( glob.glob( os.path.join(dfs_path, "*.dump") ) )
	#print(all_dump_file_paths)

	all_files = [dfile[dfile.rfind("/")+1:dfile.rfind(".dump")] for dfile in all_dump_file_paths]
	#print(all_files)

	query_dataframe_file = all_files[QUERY] 
	res_dir = os.path.join(rpath, query_dataframe_file)

	return res_dir

def plot_missing_features(df, fname, RES_DIR):
	print(f">> Visualizing missing data of {fname} ...")

	print(f">>>>> Barplot >>>>>")
	g = sns.displot(
			data=df.isna().melt(value_name="Missing"),
			y="variable",
			hue="Missing",
			multiple="stack",
			height=15,
			#kde=True,
			aspect=1.8,
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
	plt.savefig(os.path.join( RES_DIR, f"{fname}_missing_barplot.png" ), )
	plt.clf()

	print(f">>>>> Heatmap >>>>>")
	f, ax = plt.subplots()
	ax = sns.heatmap(
			df.isna(),
			cmap=sns.color_palette("Greys"),
			cbar_kws={'label': 'NaN (Missing Data):', 'ticks': [0.0, 1.0]},
			)

	ax.set_ylabel(f"Samples\n\n{df.shape[0]}$\longleftarrow${0}")
	ax.set_yticks([])
	ax.xaxis.tick_top()
	ax.tick_params(axis='x', labelrotation=90)
	plt.suptitle(f"Missing {fname} Data (NaN)")
	plt.savefig(os.path.join( RES_DIR, f"{fname}_missing_heatmap.png" ), )
	plt.clf()

def plot_language(df, fname, RES_DIR, N=10):
	unq = df["language"].value_counts()
	print(f">> langueage:\n{unq}")

	df_tmp = df.dropna(axis=0, how="any", subset=["language"]).reset_index(drop=True)

	lu, lc = np.unique(df_tmp["language"], return_counts=True)
	print(lu.shape[0], lu, lc)

	print(f"\n>> sorting for Top {N} ...")
	lc_sorted_idx = np.argsort(-lc)

	language_ung = lu[lc_sorted_idx][:N]
	language_counts = lc[lc_sorted_idx][:N]
	print(language_ung.shape[0], language_ung, language_counts)

	clrs = ["#1d7874",
          "#f4c095",
          "#ee2e31",
          "#ffb563",
        	"#918450",
          "#f85e00",
          "#9a031e",
          "#d6d6d6",
          "#ffee32",
          "#333533",
          "#a41623",
          "#679289",
          "#202020",
          '#1f77b4', 
					'#cc9911', 
					'#e377c2', 
					'#7f7f7f', 
					'#99ff99',
					'#ff7f0e', 
					'#16b3ff',
					"#ffd100",
          '#9467bd', 
					'#d62728', 
					'#0ecd19', 
					'#ffcc99', 
					'#bcbd22', 
					'#ffc9', 
					'#17becf',
					'#2ca02c', 
					'#8c564b', 
					'#ff9999',
					]

	patches, lbls, pct_texts = plt.pie(language_counts,
																				labels=language_ung, 
																				autopct='%1.1f%%', 
																				#startangle=180,
																				#radius=3, USELESS IF axs[0].axis('equal')
																				#pctdistance=1.5,
																				#labeldistance=0.5,
																				rotatelabels=True,
																				#counterclock=False,
																				colors=clrs,
																				)
	for lbl, pct_text in zip(lbls, pct_texts):
		pct_text.set_rotation(lbl.get_rotation())

	plt.title(f"Top {N} Searched Languages in NLF | Total Entry: {df_tmp['language'].shape[0]}")
	plt.savefig(os.path.join( RES_DIR, f"{fname}_pie_chart_language.png" ), )
	plt.clf()

def plot_word(df, fname, RES_DIR, Nq=10, Nu=5):
	unq = df["query_word"].value_counts()
	print(f">> query_word:\n{unq}")

	# Query Words:
	df_tmp = df.dropna(axis=0, how="any", subset=["query_word"]).reset_index(drop=True)

	qu, qc = np.unique(df_tmp["query_word"], return_counts=True)
	print(qu.shape[0], qu, qc)

	print(f"\n>> Sorting Top {Nq} Query Words / {df_tmp.shape[0]} | {fname}")
	qc_sorted_idx = np.argsort(-qc)
	query_ung = qu[qc_sorted_idx][:Nq]
	query_counts = qc[qc_sorted_idx][:Nq]
	print(query_ung.shape[0], query_ung, query_counts)
	#return

	""" #users:
	df_tmp = df.dropna(axis=0, how="any", subset=["query_word"]).reset_index(drop=True)

	qu, qc = np.unique(df_tmp["query_word"], return_counts=True)
	print(qu.shape[0], qu, qc)

	print(f"\n>> Sorting Top {Nq} Query Words / {df_tmp.shape[0]} | {fname}")
	qc_sorted_idx = np.argsort(-qc)
	query_ung = qu[qc_sorted_idx][:Nq]
	query_counts = qc[qc_sorted_idx][:Nq]
	print(query_ung.shape[0], query_ung, query_counts)
	"""

	plt.subplots()
	palette = ["#ee2e31",
						"#ffb563",
						"#918450",
						"#f85e00",
						"#a41623",
						"#1d7874",
						"#679289",
						"#f4c095",
						"#9a031e",
						"#d6d6d6",
						"#ffee32",
						"#ffd100",
						"#333533",
						"#202020",
						]
	p = sns.barplot(x=query_ung,
									y=query_counts,
									palette=palette, 
									saturation=1, 
									edgecolor = "#1c1c1c",
									linewidth = 2,
									)

	p.axes.set_title(f"\nTop {Nq} Query Word / {df_tmp.shape[0]} | {fname}\n", fontsize=18)
	plt.ylabel("Counts", fontsize = 15)
	plt.xlabel("\nQuery Phrase", fontsize = 15)
	# plt.yscale("log")
	plt.xticks(rotation=90)
	for container in p.containers:
			p.bar_label(container,
									label_type = "center",
									padding = 6,
									size = 15,
									color = "black",
									rotation = 90,
									bbox={"boxstyle": "round", 
												"pad": 0.6, 
												"facecolor": "orange", 
												"edgecolor": "black", 
												"alpha": 1,
												}
									)

	sns.despine(left=True, bottom=True)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_top_{Nq}_query_words.png" ), )
	plt.clf()
	#sys.exit()


	wordcloud = WordCloud(width=800, 
												height=400, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=50, 
												stopwords=None,
												repeat= True).generate(df["query_word"].str.cat(sep=","))

	plt.figure(figsize = (20, 8),facecolor = "#ffd100") 
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.margins(x = 0, y = 0)
	plt.tight_layout(pad = 0)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_query_words_cloud.png" ), )
	plt.clf()


	""" 
	GENDERS = {}

	for g in gender_unq:
			lst = []
			for p in profession_unq:
					#print(g, p)
					#c = df.query(f"Profession=='{str(p)}' and Gender=='{str(g)}'").Gender.count()
					c = df[(df["Profession"] == p) & (df["Gender"] == g) ].Gender.count()
					#print(c)
					
					lst.append(c)
			GENDERS[g] = lst

	print(GENDERS)

	WIDTH = 0.35
	BOTTOM = 0

	for k, v in GENDERS.items():
			#print(k, v)
			axs[1].bar(x=profession_unq, 
								height=v, 
								width=WIDTH,
								bottom=BOTTOM, 
								color=clrs[list(GENDERS.keys()).index(k)],
								label=k,
								)
			BOTTOM += np.array(v)

	axs[1].set_ylabel('Counts')
	axs[1].set_xlabel('Profession')
	axs[1].set_title('Profession by Gender')

	axs[1].legend(ncol=len(GENDERS), loc="best", frameon=False)
	plt.suptitle(f"{distribution} Distribution")
	plt.savefig(gender_distribution_fname)
	plt.show() 
	"""





def plot_ocr_term(df, fname, RES_DIR):
	unq = df["ocr_term"].value_counts()
	print(f">> ocr_term:\n{unq}")

	wordcloud = WordCloud(width=800, 
												height=400, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=50, 
												stopwords=None,
												repeat= True).generate(df["ocr_term"].str.cat(sep=", | , | ,"))

	plt.figure(figsize = (20, 8),facecolor = "#ffd100") 
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.margins(x = 0, y = 0)
	plt.tight_layout(pad = 0)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_OCR_terms_cloud.png" ), )
	plt.clf()

def plot_doc_type(df, fname, RES_DIR):
	unq = df["document_type"].value_counts()
	print(f">> doc_type:\n{unq}")

	wordcloud = WordCloud(width=800, 
												height=400, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=50, 
												stopwords=None,
												repeat= True).generate(df["document_type"].str.cat(sep=","))

	plt.figure(figsize = (20, 8),facecolor = "#ffd100") 
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.margins(x = 0, y = 0)
	plt.tight_layout(pad = 0)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_doc_type_cloud.png" ), )
	plt.clf()

def plot_user(df, fname, RES_DIR, N=10):
	unq = df["user_ip"].value_counts()
	print(f">> user_ip:\n{unq}")

	df_tmp = df.dropna(axis=0, how="any", subset=["user_ip"]).reset_index(drop=True)

	lu, lc = np.unique(df_tmp["user_ip"], return_counts=True)
	print(lu.shape[0], lu, lc)

	print(f"\n>> Sorting Top {N} users / {df_tmp.shape[0]} | {fname}")
	lc_sorted_idx = np.argsort(-lc)
	language_ung = lu[lc_sorted_idx][:N]
	language_counts = lc[lc_sorted_idx][:N]
	print(language_ung.shape[0], language_ung, language_counts)
	#return

	plt.subplots()
	palette = ["#1d7874",
						"#679289",
						"#f4c095",
						"#ee2e31",
						"#ffb563",
						"#918450",
						"#f85e00",
						"#a41623",
						"#9a031e",
						"#d6d6d6",
						"#ffee32",
						"#ffd100",
						"#333533",
						"#202020",
						]
	p = sns.barplot(x=language_ung,
									y=language_counts,
									palette=palette, 
									saturation=1, 
									edgecolor = "#1c1c1c",
									linewidth = 2,
									)

	p.axes.set_title(f"\nTop {N} Users / {df_tmp.shape[0]} | {fname}\n", fontsize=18)
	plt.ylabel("Counts", fontsize = 20)
	plt.xlabel("\nUser Name", fontsize = 20)
	# plt.yscale("log")
	plt.xticks(rotation=90)
	for container in p.containers:
			p.bar_label(container,
									label_type = "center",
									padding = 6,
									size = 15,
									color = "black",
									rotation = 90,
									bbox={"boxstyle": "round", 
												"pad": 0.6, 
												"facecolor": "orange", 
												"edgecolor": "black", 
												"alpha": 1,
												}
									)

	sns.despine(left=True, bottom=True)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_top_{N}_users.png" ), )
	plt.clf()

def main():
	print("#"*70)
	print(f"\t\t\tDATA ANALYSIS")
	print("#"*70)

	result_directory = get_result_directory(QUERY=args.query)
	print(result_directory)
	make_folder(folder_name=result_directory)

	QUERY_FILE = get_query_dataframe(QUERY=args.query)
	df = load_df(infile=QUERY_FILE)

	print(df.shape)
	cols = list(df.columns)
	print(len(cols), cols)
	print("#"*150)

	print(df.head(10))
	print("-"*150)
	print(df.tail(10))

	print(df.isna().sum())
	print("-"*150)
	print(df.info(verbose=True, memory_usage="deep"))

	# missing features:
	plot_missing_features(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# users:
	plot_user(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# language
	plot_language(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# publication
	plot_doc_type(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# query words & terms:
	plot_word(df, fname=QUERY_FILE, RES_DIR=result_directory)
	plot_ocr_term(df, fname=QUERY_FILE, RES_DIR=result_directory)

if __name__ == '__main__':
	os.system('clear')
	main()