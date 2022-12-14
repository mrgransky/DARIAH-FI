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

clrs = ["#333533", 
				'#d62728', 
				'#1f77b4',
				'#8c564b',
				"#679289",
				"#918450",
				"#ffb563",
				'#ffc9', 
				'#ff9999',
				"#f85e00",
				'#2ca02c', 
				"#ee2e31",
				'#16b3ff', 
				'#ffcc99', 
				'#7f7f7f', 
				'#bcbd22',
				"#9a031e",
				'#e377c2', 
				"#202020",
				"#1d7874",
				"#f4c095",
				'#99ff99',
				'#ff7f0e',
				"#d6d6d6",
				"#ffee32",
				'#17becf',
				"#a41623",
				"#ffd100",
				'#9467bd', 
				'#cc9911', 
				'#0ecd19', 
				]

my_cols = [ "#ffd100", '#16b3ff', '#0ecd19', '#ff9999',]

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
	df_cleaned = df.dropna(axis=0, how="any", subset=["language"]).reset_index()

	df_unq = df_cleaned.assign(language=df_cleaned['language'].str.split(',')).explode('language')
	
	print("#"*150)
	print(df_unq["language"].value_counts())
	print("#"*150)

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_cleaned["language"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8, 
																			edgecolor="#1c1c",
																			linewidth=0.5,
																			),
											)
	
	axs.axis('equal')
	axs.set_title(f"Raw NLF Languages\n{df_cleaned['language'].shape[0]}/{df['language'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v:.2f}" for l, v in zip(	df_cleaned["language"].value_counts(normalize=True).index, 
																						df_cleaned["language"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=8,
						)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_pie_chart_RAW_language.png" ), 
							bbox_inches="tight",
							)
	plt.clf()

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_unq["language"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8, 
																			edgecolor="#1c1c",
																			linewidth=0.5,
																			),
											)
	
	axs.axis('equal')
	axs.set_title(f"Unique NLF Language\n{df_unq['language'].shape[0]}/{df['language'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v:.2f}" for l, v in zip(	df_unq["language"].value_counts(normalize=True).index, 
																						df_unq["language"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=9,
						)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_pie_chart_unique_language.png" ), 
							bbox_inches="tight",
							)
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
	#

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
	df_cleaned = df.dropna(axis=0, how="any", subset=["document_type"]).reset_index()
	print( df_cleaned["document_type"].value_counts() )
	print("-"*20)

	dt = df_cleaned["document_type"].value_counts(normalize=True)
	print("-"*20)

	print(sum(df_cleaned["document_type"].value_counts()))

	print("-"*20)

	print(len(df_cleaned["document_type"].value_counts()))
	print("*"*150)

	df_unq = df_cleaned.assign(document_type=df_cleaned['document_type'].str.split(',')).explode('document_type')
	df_unq["document_type"] = df_unq["document_type"].str.title()

	print( df_unq["document_type"].value_counts() )
	print("-"*20)

	print( df_unq["document_type"].value_counts(normalize=True) )
	print("-"*20)

	print(sum(df_unq["document_type"].value_counts()))
	print("-"*20)

	print(len(df_unq["document_type"].value_counts()))
	print("#"*150)

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_cleaned["document_type"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8, 
																			edgecolor="#1c1c",
																			linewidth=0.3,
																			),
											)
	
	axs.axis('equal')
	axs.set_title(f"Raw NLF document type\n{df_cleaned['document_type'].shape[0]}/{df['document_type'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	print([ f"{l} {v:.2f}" for l, v in zip(df_cleaned["document_type"].value_counts(normalize=True).index, 
																			df_cleaned["document_type"].value_counts(normalize=True).values,
																			)
				]
			)
	
	print()

	ax2.legend(patches,
						[ f"{l} {v:.2f}" for l, v in zip(	df_cleaned["document_type"].value_counts(normalize=True).index, 
																						df_cleaned["document_type"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=5,
						)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_pie_chart_RAW_document_type.png" ), 
							bbox_inches="tight",
							)
	plt.clf()

	fig = plt.figure(figsize=(10,4))
	axs = fig.add_subplot(121)
	patches, _ = axs.pie(df_unq["document_type"].value_counts(),
											colors=clrs,
											wedgeprops=dict(width=0.8, 
																			edgecolor="#1c1c",
																			linewidth=0.5,
																			),
											)
	
	axs.axis('equal')
	axs.set_title(f"Unique NLF document type\n{df_unq['document_type'].shape[0]}/{df['document_type'].shape[0]}")
	
	ax2 = fig.add_subplot(122)
	ax2.axis("off")

	ax2.legend(patches,
						[ f"{l} {v:.2f}" for l, v in zip(	df_unq["document_type"].value_counts(normalize=True).index, 
																						df_unq["document_type"].value_counts(normalize=True).values,
																					)
						],
						loc="center",
						frameon=False,
						fontsize=9,
						)
	plt.savefig(os.path.join( RES_DIR, f"{fname}_pie_chart_unique_document_type.png" ), 
							bbox_inches="tight",
							)
	plt.clf()

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
	plt.title(f"Document Type Cloud\n{fname}", color="white")
	plt.margins(x=0, y=0)
	plt.tight_layout(pad=0) 

	plt.savefig(os.path.join( RES_DIR, f"{fname}_doc_type_cloud.png" ), )
	plt.clf()

	df_count_dt = df_unq.groupby(df_unq["document_type"])[["user_ip", "query_word", "ocr_term",]].count().reset_index()
	print(df_count_dt)

	print("#"*150)
	
	fig, axs = plt.subplots()
	df_count_dt.set_index("document_type").plot( rot=0,
								ax=axs,
								kind='bar',
								xlabel="Unique Document Type", 
								ylabel="Count",
								title=f"{df_count_dt.shape[0]} Unique Document Type | USERS | QUERY WORDS | OCR TERMS\n{fname}",
								color=clrs,
								alpha=0.6,
								)
	for container in axs.containers:
		axs.bar_label(container)
	plt.legend(	loc="upper left", 
							frameon=False,
							#title=f"Top-{Nu} Users",
							ncol=df_count_dt.shape[0], 
							)

	plt.savefig(os.path.join( RES_DIR, f"{fname}_unq_doc_type.png" ), )
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

	p.axes.set_title(f"Top {N} Users / {df_tmp.shape[0]}\n{fname}", 
									#fontsize=18,
									)
	plt.ylabel("Counts", 
						#fontsize = 20,
						)
	plt.xlabel("\nUser Name", 
							#fontsize=20,
							)
	# plt.yscale("log")
	plt.xticks(rotation=90)
	for container in p.containers:
			p.bar_label(container,
									label_type="center",
									padding=6,
									size=15,
									color="black",
									rotation=90,
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

def plot_hourly_activity(df, fname, RES_DIR, Nu=10):
	df_count = df.groupby(df["timestamp"].dt.hour)[["user_ip", "query_word", "ocr_term",]].count().reset_index()

	fig, axs = plt.subplots()
	
	df_count.set_index("timestamp").plot( rot=0, 
								ax=axs,
								kind='bar',
								xlabel="Hour (o'clock)", 
								ylabel="Activity",
								title=f"24-Hour Activity\n{fname}",
								color=clrs,
								alpha=0.6,
								)
	
	df_count.set_index("timestamp").plot(	kind='line', 
								rot=0, 
								ax=axs, 
								marker="*", 
								linestyle="-", 
								linewidth=0.5,
								color=clrs,
								label=None,
								legend=False,
								xlabel="Hour (o'clock)", 
								ylabel="Activity",
								)

	plt.legend(	["Users", "Query Words", "OCR Terms"],
							loc="best", 
							frameon=False,
							ncol=df_count.shape[0],
							)
	
	
	plt.savefig(os.path.join( RES_DIR, f"{fname}_24h_activity.png" ), )
	plt.clf()

	#time_window = df_count["timestamp"].shape[0]/4
	time_window = int(24 / 4)
	
	print(df_count.columns)
	for col in df_count.columns[1:]:
			fig, axs = plt.subplots()
			print(col)
			df_count[col].plot(	kind='line', 
																	rot=0, 
																	ax=axs, 
																	marker="*", 
																	linestyle="-", 
																	linewidth=0.5,
																	color=clrs,
																	label=None,
																	legend=False,
																	xlabel="Hour (o'clock)", 
																	ylabel="Activity",
																	title=f"Mean($\mu$) & Standard Deviation ($\sigma$): {col}\n{fname}"
																)

			for i in range(4):
				print(i)
				print(i*time_window , (i+1)*time_window)
				df_sliced = df_count[i*time_window: ((i+1)*time_window)+1]
				print(df_sliced)

				axs.plot(df_sliced["timestamp"], 
								[df_sliced.mean()[col] for t in df_sliced["timestamp"]],
								color="red",
								linestyle="-",
								linewidth=0.7,
								label=f"($\mu$ = {df_sliced.mean()[col]:.1f})\t{i*time_window}:{(i+1)*time_window} o'clock",
								)

				axs.fill_between(df_sliced["timestamp"], 
								[ df_sliced.mean()[col] - 3 * df_sliced.std()[col] for t in df_sliced["timestamp"]],
								[ df_sliced.mean()[col] + 3 * df_sliced.std()[col] for t in df_sliced["timestamp"]],
								alpha=0.1,
								color=my_cols[i],
								label=f"+/- 3$\sigma$\t{i*time_window}:{(i+1)*time_window} o'clock",
								)

				axs.fill_between(df_sliced["timestamp"], 
								[ df_sliced.mean()[col] - 1 * df_sliced.std()[col] for t in df_sliced["timestamp"]],
								[ df_sliced.mean()[col] + 1 * df_sliced.std()[col] for t in df_sliced["timestamp"]],
								alpha=0.14,
								color=my_cols[i],
								label=f"+/- 1$\sigma$\t{i*time_window}:{(i+1)*time_window} o'clock",
								)


				#print(df_sliced_grouped)
				
				print("-"*60)
			
				axs.legend(	#["Users", "Query Words", "OCR Terms"],
										loc="upper left", 
										frameon=False,
										#ncol=df_count.shape[0],
										)
				plt.savefig(os.path.join( RES_DIR, f"{fname}_{col}_mean_std.png" ), )


	#
	
	uu, uc = np.unique(df["user_ip"], return_counts=True)
	#print(uu.shape[0], uu, uc)

	print(f"\n>> Sorting Top {Nu} Users / {df.shape[0]} | {fname}")
	uc_sorted_idx = np.argsort(-uc)
	user_ung = uu[uc_sorted_idx][:Nu]
	user_counts = uc[uc_sorted_idx][:Nu]
	#print(user_ung.shape[0], user_ung, user_counts)


	#

def plot_usr_doc_type(df, fname, RES_DIR, Nu=10):
	df_tmp = df.assign(document_type=df['document_type'].str.split(',')).explode('document_type')
	
	print(df_tmp["document_type"].value_counts())
	print("#"*150)

	df_count = df.groupby(df["document_type"])[["user_ip", "query_word", "ocr_term",]].count().reset_index()
	print(df_count)
	print("#"*150)

	df_count_usr = df_tmp.groupby(df_tmp["user_ip"])[["document_type"]].count().reset_index().sort_values(by="document_type", ascending=False)
	print(df_count_usr)
	print("#"*150)


	print(df_tmp[df_tmp["document_type"]=="NEWSPAPER"][["user_ip","document_type"]])
	print("/"*150)
	
	# TOP-N users:	
	df_topN_users = df_tmp["user_ip"].value_counts().head(Nu).rename_axis('user_ip').reset_index(name='occurrence')
	print(df_topN_users)

	fig, axs = plt.subplots()
	#users:
	dfgr_dt_usr = df.groupby('document_type')['user_ip']#.sort_values(by="user_ip", ascending=False)
	print(len(dfgr_dt_usr.get_group("JOURNAL")))
	print(">"*10)
	print(dfgr_dt_usr.get_group("JOURNAL"))

	df_doc_type_vs_users = dfgr_dt_usr.count() # change to pandas df
	df_doc_type_vs_users.plot( rot=90,
								ax=axs,
								kind='bar',
								xlabel="Document Type", 
								ylabel="Count",
								title=f"User Intrests of {df_doc_type_vs_users.shape[0]} Raw Document Type\n{fname}",
								color=clrs,
								alpha=0.6,
								)
	axs.bar_label(axs.containers[0])
	plt.savefig(os.path.join( RES_DIR, f"{fname}_usr_vs_doc_type.png" ), )
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
	print("%"*120)
	"""
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
	"""
	print(df[df.select_dtypes(include=[object]).columns].describe().T)
	print("%"*120)
	
	# missing features:
	#plot_missing_features(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# 24h activity:
	#plot_hourly_activity(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# users:
	#plot_user(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# language
	plot_language(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# doc_type
	plot_doc_type(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# users vs document_type:
	#plot_usr_doc_type(df, fname=QUERY_FILE, RES_DIR=result_directory)

	# query words & terms:
	#plot_word(df, fname=QUERY_FILE, RES_DIR=result_directory)
	#plot_ocr_term(df, fname=QUERY_FILE, RES_DIR=result_directory)

if __name__ == '__main__':
	os.system('clear')
	main()