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

clrs = ["#1d7874",
          "#f4c095",
          "#ee2e31",
          '#1f77b4',
          "#ffb563",
        	"#918450",
          "#d6d6d6",
          "#ffee32",
          "#333533",
          "#a41623",
          "#679289", 
					'#16b3ff', 
					'#ff9999',
          "#202020",
          "#f85e00",
					'#2ca02c', 
					'#e377c2', 
					'#7f7f7f', 
					'#99ff99',
					'#ff7f0e',
					"#ffd100",
          '#9467bd', 
					'#cc9911', 
					'#d62728', 
					'#0ecd19', 
					'#ffcc99', 
					'#bcbd22', 
					'#ffc9', 
					'#17becf',
          "#9a031e", 
					'#8c564b',
					]

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
			cbar_kws={'label': 'NaN (Missing Data)', 
								'ticks': [0.0, 1.0]},
			)

	ax.set_ylabel(f"Samples\n\n{df.shape[0]}$\longleftarrow${0}")
	ax.set_yticks([])
	ax.xaxis.tick_top()
	ax.tick_params(axis='x', labelrotation=90)
	plt.suptitle(f"Missing Data (NaN)\n{fname}")
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

def plot_word(df, fname, RES_DIR, Nq=25, Nu=20):
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
	#print(query_ung.shape[0], query_ung, query_counts)
	print("#"*140)

	#print(df_tmp["query_word"].str.cat(sep=",").split(","))

	#sys.exit(0)
	#return

	#users:
	df_tmp_user = df.dropna(axis=0, how="any", subset=["user_ip"]).reset_index(drop=True)

	uu, uc = np.unique(df_tmp_user["user_ip"], return_counts=True)
	#print(uu.shape[0], uu, uc)

	print(f"\n>> Sorting Top {Nu} Users / {df_tmp_user.shape[0]} | {fname}")
	uc_sorted_idx = np.argsort(-uc)
	user_ung = uu[uc_sorted_idx][:Nu]
	user_counts = uc[uc_sorted_idx][:Nu]
	#print(user_ung.shape[0], user_ung, user_counts)

	wordcloud = WordCloud(width=1400, 
												height=550, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=80,
												stopwords=None,
												collocations=False,
												).generate_from_frequencies(dict(zip(qu, qc)))



	plt.figure(figsize=(14, 8),
						facecolor="grey",
						) 
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.title(f"{len(qu)} unique Query Phrases Cloud Distribution (total: {df_tmp['query_word'].shape[0]})\n{fname}", color="white")
	plt.margins(x=0, y=0)
	plt.tight_layout(pad=0) 
	plt.savefig(os.path.join( RES_DIR, f"{fname}_query_words_cloud.png" ), )
	plt.clf()

	plt.subplots()
	
	p = sns.barplot(x=query_ung,
									y=query_counts,
									palette=clrs, 
									saturation=1, 
									edgecolor = "#1c1c1c",
									linewidth = 2,
									)

	p.axes.set_title(	f"\nTop-{Nq} Query Phrases out of total of: {df_tmp.shape[0]}\n{fname}\n"
										#,fontsize=18,
										)
	plt.ylabel("Counts", 
							#fontsize=15,
							)
	plt.xlabel("\nQuery Phrase", 
							#fontsize=15,
							)
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

	GENDERS = {}

	for g in user_ung:
			lst = []
			for p in query_ung:
					#print(g, p)
					#c = df.query(f"Profession=='{str(p)}' and Gender=='{str(g)}'").Gender.count()
					c = df[(df["query_word"] == p) & (df["user_ip"] == g) ].user_ip.count()
					#print(c)
					
					lst.append(c)
			GENDERS[g] = lst

	print(GENDERS)

	WIDTH = 0.35
	BOTTOM = 0

	for k, v in GENDERS.items():
			#print(k, v)
			plt.bar(x=query_ung, 
								height=v, 
								width=WIDTH,
								bottom=BOTTOM, 
								color=clrs[list(GENDERS.keys()).index(k)],
								label=k,
								)
			BOTTOM += np.array(v)

	#axs[1].set_ylabel('Counts')
	#axs[1].set_xlabel('Profession')
	#axs[1].set_title('Profession by Gender')

	plt.legend(	loc="best", 
							frameon=False,
							title=f"Top-{Nu} Users"
							#ncol=len(GENDERS), 
							)
	plt.suptitle(f"Top-{Nq} Query Phrases Searched by Top-{Nu} Users | {fname}")
	plt.xticks(rotation=90)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_USR_vs_query_words.png" ), )
	plt.clf()

def plot_ocr_term(df, fname, RES_DIR, N=20):
	unq = df["ocr_term"].value_counts()
	print(f">> ocr_term:\n{unq}")

	df_tmp = df.dropna(axis=0, how="any", subset=["ocr_term"]).reset_index(drop=True)

	ocr_u, ocr_c = np.unique(df_tmp["ocr_term"], return_counts=True)
	#print(ocr_u.shape[0], ocr_u, ocr_c)

	print(f"\n>> Sorting Top {N} OCR terms / {df_tmp.shape[0]} | {fname}")
	ocr_c_sorted_idx = np.argsort(-ocr_c)
	ocr_ung = ocr_u[ocr_c_sorted_idx][:N]
	ocr_counts = ocr_c[ocr_c_sorted_idx][:N]
	#print(ocr_ung.shape[0], ocr_ung, ocr_counts)

	wordcloud = WordCloud(width=1400, 
												height=550, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=80,
												stopwords=None,
												collocations=False,
												).generate_from_frequencies(dict(zip(ocr_u, ocr_c)))

	plt.figure(figsize=(14, 8),
						facecolor="grey",
						) 
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.title(f"{len(ocr_u)} unique OCR Terms Cloud Distribution (total: {df_tmp['ocr_term'].shape[0]})\n{fname}", color="white")
	plt.margins(x=0, y=0)
	plt.tight_layout(pad=0) 

	plt.savefig(os.path.join( RES_DIR, f"{fname}_OCR_terms_cloud.png" ), )
	plt.clf()

def plot_doc_type(df, fname, RES_DIR):
	unq = df["document_type"].value_counts()
	print(f">> doc_type:\n{unq}")


	df_tmp = df[["document_type"]]
	df_tmp["document_type"] = df_tmp["document_type"].str.split(",")
	print(df_tmp.head(20))

	wordcloud = WordCloud(width=800, 
												height=250, 
												background_color="black",
												colormap="RdYlGn",
												max_font_size=50,
												stopwords=None,
												collocations=False,
												).generate(df["document_type"].str.cat(sep=",")) #Concatenate strings in the Series/Index with given separator.

	plt.figure(figsize=(10, 4),
						facecolor="grey",
						) 
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.title(f"Document Types in Cloud | {fname}", color="white")
	plt.margins(x=0, y=0)
	plt.tight_layout(pad=0) 

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
	p = sns.barplot(x=language_ung,
									y=language_counts,
									palette=clrs, 
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

def plot_user_vs_doc_type(df, fname, RES_DIR, Nu=10):
	# doc_type:
	df_tmp_doc_type = df.dropna(axis=0, how="any", subset=["document_type"]).reset_index(drop=True)
	dt_u, dt_c = np.unique(df_tmp_doc_type["document_type"], return_counts=True)
	print(dt_u.shape[0], dt_u, dt_c)


	#users:
	gk = df.groupby('document_type', as_index=False )#['user_ip'].count().sort_values(by="user_ip", ascending=False)
	#print(gk.get_group('JOURNAL')['user_ip'])



	df_test = df.groupby(df["timestamp"].dt.hour)[["user_ip", "query_word", "ocr_term",]].count()
	fig, axs = plt.subplots()
	
	df_test.plot( rot=0, 
								ax=axs,
								kind='bar',
								xlabel="Hour", 
								ylabel="Activity",
								title=f"24-Hour Activity\n{fname}",
								)
	df_test.plot(kind='line', rot=0, ax=axs, marker="*", linestyle="-")
	
	#print(df_test)

	plt.legend(	loc="best", 
							ncol=df_test.shape[0],
							frameon=False,
							)
	
	plt.savefig(os.path.join( RES_DIR, f"{fname}_USR_vs_hour_activity.png" ), )
	plt.clf()

	uu, uc = np.unique(df["user_ip"], return_counts=True)
	#print(uu.shape[0], uu, uc)

	print(f"\n>> Sorting Top {Nu} Users / {df.shape[0]} | {fname}")
	uc_sorted_idx = np.argsort(-uc)
	user_ung = uu[uc_sorted_idx][:Nu]
	user_counts = uc[uc_sorted_idx][:Nu]
	#print(user_ung.shape[0], user_ung, user_counts)


	sys.exit()

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
	#plot_missing_features(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# users vs document_type:
	plot_user_vs_doc_type(df, fname=QUERY_FILE, RES_DIR=result_directory)


	# users:
	#plot_user(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# language
	#plot_language(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# publication
	plot_doc_type(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# query words & terms:
	plot_word(df, fname=QUERY_FILE, RES_DIR=result_directory)
	plot_ocr_term(df, fname=QUERY_FILE, RES_DIR=result_directory)

if __name__ == '__main__':
	os.system('clear')
	main()