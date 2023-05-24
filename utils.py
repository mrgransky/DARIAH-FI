import os
import sys
import contextlib
import torch
import faiss
import subprocess
import urllib
import requests
import joblib
import pickle
import dill
import itertools
import re
import json
import argparse
import datetime
import glob
import webbrowser
import string
import time
import logging
#import warnings
#warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from natsort import natsorted
from collections import Counter
from typing import List, Set, Dict, Tuple

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

#import spacy
from colorama import Fore, Style, Back
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Colormap as cm
import matplotlib.ticker as ticker
import matplotlib

#matplotlib.use("Agg")

sz=16
params = {
		'figure.figsize':	(sz*1.0, sz*0.5),  # W, H
		'figure.dpi':		200,
		#'figure.autolayout': True,
		#'figure.constrained_layout.use': True,
		'legend.fontsize':	sz*0.8,
		'axes.labelsize':	sz*1.0,
		'axes.titlesize':	sz*1.0,
		'xtick.labelsize':	sz*0.8,
		'ytick.labelsize':	sz*0.8,
		'lines.linewidth' :	sz*0.1,
		'lines.markersize':	sz*0.4,
		"markers.fillstyle": "full",
		'font.size':		sz*1.0,
		'font.family':		"serif",
		'legend.title_fontsize':'small'
	}
pylab.rcParams.update(params)
clrs = ["#ff2eee",
				'#16b3fd',
				'#0eca11',
				"#ffee32",
				"#ee0038",
				"#a99",
				"#742",
				"#4aaaa5",
				"#742802",
				'#0ef',
				"#ffb563",
				'#771',
				'#d72448', 
				'#7ede2333',
				"#416",
				"#031eee",
				'#a0ee2c44',
				'#864b',
				"#a91449",
				'#1f77b4',
				'#e377c2',
				'#bcbd22',
				'#688e',
				"#100874",
				"#931e00",
				"#a98d19",
				'#1eeeee7f',
				'#007749',
				"#d6df",
				"#918450",
				'#900fcc99',
				'#17becf',
				"#e56699",
				"#265",
				'#7f688e',
				'#d62789',
				'#99f9',
				'#d627',
				"#006cf789",
				"#7eee88", 
				"#10e4",
				"#f095",
				"#a6aa1122",
				"#ee5540",
				'#25e682', 
				"#e4d10888",
				"#0000ff",
				"#102d",
			]

usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket",
				}

NLF_DATASET_PATH = usr_[os.environ['USER']]
userName = os.path.expanduser("~")
dataset_path = os.path.join( NLF_DATASET_PATH, f"datasets" )
dpath = os.path.join( NLF_DATASET_PATH, f"NLF_Pseudonymized_Logs" )
#dpath = os.path.join( NLF_DATASET_PATH, f"no_ip_logs" )

rpath = os.path.join( NLF_DATASET_PATH, f"results" )
dfs_path = os.path.join( NLF_DATASET_PATH, f"dataframes")
#dfs_path = os.path.join( NLF_DATASET_PATH, f"temp_dataframes")

def get_tokens_byUSR(sp_mtrx, df_usr_tk, bow, user="ip1025",):
	matrix = sp_mtrx.toarray()
	sp_type = "Normalized" if matrix.max() == 1.0 else "Original" 

	user_idx = int(df_usr_tk.index[df_usr_tk['user_ip'] == user].tolist()[0])

	#print(f"\n\n>> user_idx: {user_idx} - ")
	
	#tk_indeces_sorted_no_0 = np.where(matrix[user_idx, :] != 0, matrix[user_idx, :], np.nan).argsort()[:(matrix[user_idx, :] != 0).sum()]
	#print(tk_indeces_sorted_no_0[-50:])
	#tks_name = [k for idx in tk_indeces_sorted_no_0 for k, v in bow.items() if v==idx]
	#tks_value_all = matrix[user_idx, tk_indeces_sorted_no_0]
	
	tk_dict = dict( sorted( df_usr_tk.loc[user_idx , "user_token_interest" ].items(), key=lambda x:x[1], reverse=True ) )
	tk_dict = {k: v for k, v in tk_dict.items() if v!=0}

	"""
	with open(f"temp_{user_idx}_raw_dict_.json", "w") as fw:
		json.dump(df_usr_tk.loc[user_idx , "user_token_interest" ], fw, indent=4, ensure_ascii=False)

	with open("temp_sorted_discending.json", "w") as fw:
		json.dump(tk_dict, fw, indent=4, ensure_ascii=False)
	
	with open("temp_sorted_no0.json", "w") as fw:
		json.dump(tk_dict, fw, indent=4, ensure_ascii=False)
	"""

	tks_name = list(tk_dict.keys())
	tks_value_all = list(tk_dict.values())
	
	#print(tks_name[:50])
	#print(tks_value_all[:50])

	assert len(tks_name) == len(tks_value_all), f"found {len(tks_name)} tokens names & {len(tks_value_all)} tokens values"

	print(f"Retrieving all {len(tks_name)} Tokens for {user} @ idx: {user_idx} | {sp_type} Sparse Matrix".center(120, ' '))

	tks_value_separated = list()
	# qcol_list = ["Search PHRs", "Snippet HWs", "Snippet Appr", "Content HWs", "Content PRTs", "Content Appr",]
	for col in ["usrInt_qu_tk", "usrInt_sn_hw_tk", "usrInt_sn_tk", "usrInt_cnt_hw_tk", "usrInt_cnt_pt_tk", "usrInt_cnt_tk", ]:
		#print(col)
		oneTK_separated_vals = list()
		for tkn in tks_name:
			#print(tkn)
			#print(df_usr_tk.loc[user_idx , col].get(tkn))
			oneTK_separated_vals.append( df_usr_tk.loc[user_idx , col].get(tkn) )
		#print(oneTK_separated_vals)
		tks_value_separated.append(oneTK_separated_vals)
		#print("-"*80)
	#print(len(tks_name), len(tks_value_all), len(tks_value_separated), len(tks_value_separated[0]))
	print(f"Found {len(tks_name)} tokens names, {len(tks_value_all)} tokens values (total) & {len(tks_value_separated)} tokens values (separated) | {user}")
	return tks_name, tks_value_all, tks_value_separated

def get_users_byTK(sp_mtrx, df_usr_tk, bow, token="häst", ):
	matrix = sp_mtrx.toarray()
	sp_type = "Normalized" if matrix.max() == 1.0 else "Original" 
	tkIdx = bow.get(token)

	usr_indeces_sorted_no_0 = np.where(matrix[:, tkIdx] != 0, matrix[:, tkIdx], np.nan).argsort()[:(matrix[:, tkIdx] != 0).sum()] # ref: https://stackoverflow.com/questions/40857349/np-argsort-which-excludes-zero-values

	usrs_value_all = matrix[usr_indeces_sorted_no_0, tkIdx]
	usrs_name = [df_usr_tk.loc[idx, 'user_ip'] for idx in usr_indeces_sorted_no_0 ]
	print(f"Retrieving all {len(usrs_name)} Users by (Token '{token}' idx: {tkIdx}) {sp_type} Sparse Matrix".center(120, '-'))
	
	usrs_value_separated = list()
	for col in ["usrInt_qu_tk", "usrInt_sn_hw_tk", "usrInt_sn_tk", "usrInt_cnt_hw_tk", "usrInt_cnt_pt_tk", "usrInt_cnt_tk", ]:
		oneUSR_separated_vals = list()
		for usr in usrs_name:
			user_idx = int(df_usr_tk.index[df_usr_tk['user_ip'] == usr].tolist()[0])
			#print(f"tk: {token:<15}{usr:<10}@idx: {user_idx:<10}{col:<20}{df_usr_tk.loc[user_idx , col].get(token)}")
			oneUSR_separated_vals.append( df_usr_tk.loc[user_idx , col].get(token) )
		#print(oneUSR_separated_vals)
		usrs_value_separated.append(oneUSR_separated_vals[::-1])
		#print("#"*100)
	#print("-"*80)
	#print(usrs_value_separated)
	#print(len(usrs_name), len(usrs_value_all), len(usrs_value_separated), len(usrs_value_separated[0]))
	print(f"Found {len(usrs_name)} userIPs, {len(usrs_value_all)} all userIPs values & {len(usrs_value_separated)} separated userIPs values | token: {token}")
	return usrs_name[::-1], usrs_value_all[::-1], usrs_value_separated#[::-1]

def get_filename_prefix(dfname):
	fprefix = "_".join(dfname.split("/")[-1].split(".")[:-2]) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	return fprefix

def plot_heatmap(mtrx, name_="user-based", RES_DIR=""):
	st_t = time.time()
	hm_title = f"{name_} similarity heatmap".capitalize()
	print(f"{hm_title.center(70,'-')}")

	print(type(mtrx), mtrx.shape, mtrx.nbytes)
	#RES_DIR = make_result_dir(infile=args.inputDF)

	f, ax = plt.subplots()

	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	im = ax.imshow(mtrx, 
								cmap="viridis",#"magma", # https://matplotlib.org/stable/tutorials/colors/colormaps.html
								)
	cbar = ax.figure.colorbar(im,
														ax=ax,
														label="Similarity",
														orientation="vertical",
														cax=cax,
														ticks=[0.0, 0.5, 1.0],
														)

	ax.set_ylabel(f"{name_.split('-')[0].capitalize()}")
	#ax.set_yticks([])
	#ax.set_xticks([])
	ax.xaxis.tick_top()
	ax.tick_params(axis='x', labelrotation=90, labelsize=10.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=10.0)
	plt.suptitle(f"{hm_title}\n{mtrx.shape[0]} Unique Elements")
	#print(os.path.join( RES_DIR, f'{name_}_similarity_heatmap.png' ))
	plt.savefig(os.path.join( RES_DIR, f"{name_}_similarity_heatmap.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)
	print(f"Done".center(70, "-"))

def get_similarity_df(df, sprs_mtx, method="user-based", result_dir=""):
	method_dict = {"user-based": "user_ip", 
								"item-based": "title_issue_page",
								"token-based": "something_TBD",
								}
	print(f"Getting {method} similarity of {type(sprs_mtx)} : {sprs_mtx.shape}")

	similarity = cosine_similarity( sprs_mtx )
	#similarity = linear_kernel(sprs_mtx)
	
	plot_heatmap(mtrx=similarity.astype(np.float32), 
							name_=method,
							RES_DIR=result_dir,
							)

	sim_df = pd.DataFrame(similarity,#.astype(np.float32), 
												index=df[method_dict.get(method)].unique(),
												columns=df[method_dict.get(method)].unique(),
												)
	#print(sim_df.shape)
	#print(sim_df.info(verbose=True, memory_usage="deep"))
	#print(sim_df.head(25))
	#print("><"*60)

	return sim_df

def get_snippet_hw_counts(results_list):
	return [ len(el.get("terms")) if el.get("terms") else 0 for ei, el in enumerate(results_list) ]

def get_content_hw_counts(results_dict):
	hw_count = 0
	if results_dict.get("highlighted_term"):
		hw_count = len(results_dict.get("highlighted_term"))
	return hw_count

def get_search_title_issue_page(results_list):
	return [f'{el.get("bindingTitle")}_{el.get("issue")}_{el.get("pageNumber")}' for ei, el in enumerate(results_list)]

def get_content_title_issue_page(results_dict):
	return f'{results_dict.get("title")}_{results_dict.get("issue")}_{results_dict.get("page")[0]}'

def get_sparse_mtx(df):
	print(f"Sparse Matrix (USER-ITEM): {df.shape}".center(100, '-'))
	print(list(df.columns))
	print(df.dtypes)
	print(f">> Checking positive indices?")
	assert np.all(df["user_index"] >= 0)
	assert np.all(df["nwp_tip_index"] >= 0)
	print(f">> Done!")

	sparse_mtx = csr_matrix( ( df["implicit_feedback"], (df["user_index"], df["nwp_tip_index"]) ), dtype=np.int8 ) # num, row, col
	#csr_matrix( ( data, (row, col) ), shape=(3, 3))
	##########################Sparse Matrix info##########################
	print("#"*110)
	print(f"Sparse: {sparse_mtx.shape} : |elem|: {sparse_mtx.shape[0]*sparse_mtx.shape[1]}")
	print(f"<> Non-zeros vals: {sparse_mtx.data}")# Viewing stored data (not the zero items)
	#print(sparse_mtx.toarray()[:25, :18])
	print(f"<> |Non-zero vals|: {sparse_mtx.count_nonzero()}") # Counting nonzeros
	print("#"*110)
	##########################Sparse Matrix info##########################
	return sparse_mtx

def print_df_detail(df, fname="unkonwn"):
	print(f"{fname} | DF: {df.shape}".center(150, ' '))

	print(df.info(verbose=True, memory_usage="deep"))
	"""
	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 50):
		print(df[["nwp_content_results", "search_query_phrase", "search_results" ]].head(10))
	"""
	#print(df[["nwp_content_results", "search_query_phrase", "search_results" ]].head(10))

	with pd.option_context('display.max_rows', None, 'display.max_colwidth', 1500):
		"""
		print(df[["user_ip",
							"timestamp",
							#"search_query_phrase", 
							#"search_results",
							#"search_referer",
						]
					].head(50)
				)
		print("#"*80)
		print(df[["user_ip",
							"timestamp",
							#"search_query_phrase", 
							#"search_results",
							#"search_referer",
						]
					].tail(50)
				)
		#print("#"*80)
		#print(df.timestamp.value_counts()) # find duplicates of time
		print("#"*80)
		"""
		print(df.user_ip.value_counts().sort_values())

	return

	print(f"|search results| = {len(list(df.loc[1, 'search_results'][0].keys()))}\t",
				f"{list(df.loc[1, 'search_results'][0].keys())}", 
			)
	print("<>"*120)

	#print(json.dumps(df["search_results"][1][0], indent=2, ensure_ascii=False))
	print(json.dumps(df.loc[1, "search_results"][0], indent=2, ensure_ascii=False))

	print("#"*150)

	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1500):
		print(df[[#nwp_content_results", 
							"nwp_content_referer",
							]
						].head(10))

	print("<>"*100)
	print(list(df["nwp_content_results"][4].keys()))
	
	print("#"*150)
	print(json.dumps(df["nwp_content_results"][4], indent=2, ensure_ascii=False))
	
	print("DONE".center(80, "-"))

def make_result_dir(infile=""):
	f = get_filename_prefix(dfname=infile) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	res_dir = os.path.join(rpath, f)
	make_folder(folder_name=res_dir)
	return res_dir

def rest_api_sof(params={}):	
	params = {'query': 						["Rusanen"], 
						'publicationPlace': ["Iisalmi", "Kuopio"], 
						'lang': 						["FIN"], 
						'orderBy': 					["DATE_DESC"], 
						'formats': 					["NEWSPAPER"], 
						}

	print(f"REST API: {params}")

	subprocess.call(['bash',
									'sof.sh',
									f'myFORMATS={params.get("formats", "")}',
									f'myQUERY={",".join(params.get("query"))}',
									f'myORDERBY={",".join(params.get("orderBy", ""))}',
									f'myLANG={params.get("lang", "")}',
									f'myPubPlace={params.get("publicationPlace", "")}',
									]
								)

def rest_api(params={}):
	
	# TODO: url must be provided!
	params = {'query': ["Rusanen"], 
						'publicationPlace': ["Iisalmi", "Kuopio"], 
						'lang': ["FIN"], 
						'orderBy': ["DATE_DESC"], 
						'formats': ["NEWSPAPER"], 
						'resultMode': ["TEXT_WITH_THUMB"], 
						'page': ['75']}

	#print("#"*65)
	print(f"REST API: {params}")
	#print("#"*65)
	#return

	subprocess.call(['bash', 
									'my_file.sh',
									#'query_retreival.sh',
									#'QUERY=kantasonni',
									f'DOC_TYPE={params.get("formats", "")}',
									f'QUERY={",".join(params.get("query"))}',
									f'ORDER_BY={",".join(params.get("orderBy", ""))}',
									f'LANGUAGES={params.get("lang", "")}',
									f'PUB_PLACE={params.get("publicationPlace", "")}', 
									f'REQUIRE_ALL_KEYWORDS={params.get("requireAllKeywords", "")}',
									f'QUERY_TARGETS_METADATA={params.get("qMeta", "")}',
									f'QUERY_TARGETS_OCRTEXT={params.get("qOcr", "")}',
									f'AUTHOR={params.get("author", "")}', 
									f'COLLECTION={params.get("collection", "")}',
									#f'DISTRICT={}',
									f'START_DATE={params.get("startDate", "")}',
									f'END_DATE={params.get("endDate", "")}',
									f'FUZZY_SEARCH={params.get("fuzzy", "")}',
									f'HAS_ILLUSTRATION={params.get("hasIllustrations", "")}',
									#f'IMPORT_START_DATE={}',
									f'IMPORT_TIME={params.get("importTime", "")}',
									f'INCLUDE_AUTHORIZED_RESULTS={params.get("showUnauthorizedResults", "")}',#TODO: negation required?!
									f'PAGES={params.get("pages", "")}', 
									#f'PUBLICATION={}',
									f'PUBLISHER={params.get("publisher", "")}', 
									#f'SEARCH_FOR_BINDINGS={}',
									f'SHOW_LAST_PAGE={params.get("showLastPage", "")}', 
									f'TAG={params.get("tag", "")}',
									]
								)

	return
	json_file = f"newspaper_info_query_{params.get('query')}"
	f = open(json_file)
	data = json.load(f)
	#print(len(data)) # 7
	#print(type(data)) # <class 'dict'>
	print(f'>> How many results: {data.get("totalResults")}')
	print(list(data.keys()))
	print(len(data.get("rows")))
	
	print(type(data.get("rows"))) # <class 'list'>
	print()
	SEARCH_RESULTS = data.get("rows")[35*20+0 : 35*20+20]
	print(json.dumps(SEARCH_RESULTS, indent=1, ensure_ascii=False))
	print("#"*100)
	print(len(SEARCH_RESULTS))
	
	# remove json file:
	print(f">> removing {json_file} ...")
	os.remove(json_file)
	
	return SEARCH_RESULTS

def get_df_no_ip_logs(infile="", TIMESTAMP=None):
	file_path = os.path.join(dpath, infile)

	#print(f">> Reading {file_path} ...")
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (?P<status>\d{3}) (.*) "([^"]*)" "(.*?)" (.*)' # original working!
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^"]+)" "(.*?)" (.*)' # original working!
	ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^\"]*)" "(.*?)" (.*)' # checked with all log files!
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "(?:-|.*(http://\D.*))" "(.*?)" (.*)'
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "(?:|-|.*(://\D.*))" "(.*?)" (.*)'
	cleaned_lines = []

	with open(file_path, mode="r") as f:
		for line in f:
			##print(line)
			matched_line = re.match(ACCESS_LOG_PATTERN, line)
			#print (matched_line)
			l = matched_line.groups()
			#print(l)
			cleaned_lines.append({
				"timestamp": 										l[0].replace(":", " ", 1), # original: 01/Feb/2017:12:34:51 +0200
				"client_request_line": 					l[1],
				"status": 											l[2],
				"bytes_sent": 									l[3],
				"referer": 											l[4],
				"user_agent": 									l[5],
				"session_id": 									l[6],
				#"query_word":										np.nan,
				#"term":													np.nan,
				#"page_ocr":													np.nan,
				#"fuzzy":												np.nan,
				#"has_metadata":									np.nan,
				#"has_illustration":							np.nan,
				#"show_unauthorized_results":		np.nan,
				#"pages":												np.nan,
				#"import_time":									np.nan,
				#"collection":										np.nan,
				#"author":												np.nan,
				#"keyword":											np.nan,
				#"publication_place":						np.nan,
				#"language":											np.nan,
				#"document_type":								np.nan,
				#"show_last_page":								np.nan,
				#"order_by":											np.nan,
				#"publisher":										np.nan,
				#"start_date":										np.nan,
				#"end_date":											np.nan,
				#"require_all_keywords":					np.nan,
				#"result_type":									np.nan,
				})
	
	df = pd.DataFrame.from_dict(cleaned_lines)

	# with pandas:
	df.timestamp = pd.to_datetime(df.timestamp)
	#df = df.replace("null", "-", regex=True).replace("-", pd.NA, regex=True).replace(r'^\s*$', pd.NA, regex=True)
	
	# with numpy:
	df = df.replace("-", np.nan, regex=True).replace(r'^\s*$', np.nan, regex=True)
	df = df.dropna(axis=0)
	df = df.reset_index(drop=True)
	
	if TIMESTAMP:
		print(f"\t\t\twithin timeframe: {TIMESTAMP[0]} - {TIMESTAMP[1]}")
		df_ts = df[ df.timestamp.dt.strftime('%H:%M:%S').between(TIMESTAMP[0], TIMESTAMP[1]) ]		
		df_ts = df_ts.reset_index(drop=True)
		return df_ts

	return df

def get_df_pseudonymized_logs(infile="", TIMESTAMP=None):
	file_path = os.path.join(dpath, infile)
	ACCESS_LOG_PATTERN = '(.*?) - - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^\"]*)" "(.*?)" (.*)' # checked with all log files!
	cleaned_lines = []

	with open(file_path, mode="r") as f:
		for line in f:
			#print(line)
			matched_line = re.match(ACCESS_LOG_PATTERN, line)
			#print (matched_line)
			l = matched_line.groups()
			#print(l)
			
			cleaned_lines.append({
				"user_ip": 							l[0],
				"timestamp": 						l[1].replace(":", " ", 1), # original: 01/Feb/2017:12:34:51 +0200
				"client_request_line": 	l[2],
				"status": 							l[3],
				"bytes_sent": 					l[4],
				"referer": 							l[5],
				"user_agent": 					l[6],
				"session_id": 					l[7],
				#"query_word":										np.nan,
				#"term":													np.nan,
				#"page_ocr":													np.nan,
				#"fuzzy":												np.nan,
				#"has_metadata":									np.nan,
				#"has_illustration":							np.nan,
				#"show_unauthorized_results":		np.nan,
				#"pages":												np.nan,
				#"import_time":									np.nan,
				#"collection":										np.nan,
				#"author":												np.nan,
				#"keyword":											np.nan,
				#"publication_place":						np.nan,
				#"language":											np.nan,
				#"document_type":								np.nan,
				#"show_last_page":								np.nan,
				#"order_by":											np.nan,
				#"publisher":										np.nan,
				#"start_date":										np.nan,
				#"end_date":											np.nan,
				#"require_all_keywords":					np.nan,
				#"result_type":									np.nan,
				})
	
	df = pd.DataFrame.from_dict(cleaned_lines)

	# with pandas:
	df.timestamp = pd.to_datetime(df.timestamp)
		
	#print(f">> Raw DF: {df.shape}")
	#print(df.isna().sum())
	#print("-"*50)

	#print(f">> Replacing space + bad urls + empty fields with np.nan :")
	# with numpy:
	#df = df.replace("-", np.nan, regex=True).replace(r'^\s*$', np.nan, regex=True).replace(r'http://+', np.nan, regex=True)
	#df = df.replace(r'-|^\s*$|http://+', np.nan, regex=True)
	#df = df.replace(r'-|^\s*$|http://[0-9]+|https://[0-9]+', np.nan, regex=True)
	df["referer"] = df["referer"].replace(r'-|^\s*$|http://[0-9]+|https://[0-9]+', np.nan, regex=True)

	# check nan:
	#print(f">> Secondary checcking for None/Null values: {df.shape}")
	#print(df.isna().sum())
	#print("-"*50)

	#print(f">> Dropping None for referer & user_ip:")
	df = df.dropna(subset=['user_ip', 'referer'])
	#print(f">> After droping None of user_ip & referer: {df.shape}")
	"""
	print(f">> Before Duplicate removal:") # youtube tutorial on drop dups: https://www.youtube.com/watch?v=xi0vhXFPegw (time: 15:50)
	print(f"\tuser & referer dups: {df[df.duplicated(subset=['user_ip', 'referer'])].shape[0]}")
	print(f"\tuser & timestamps dups: {df[df.duplicated(subset=['user_ip', 'timestamp'])].shape[0]}")
	print(f"\tuser & referer & timestamps dups: {df[df.duplicated(subset=['user_ip', 'referer', 'timestamp'])].shape[0]}")
	"""
	df['prev_time'] = df.groupby(['referer','user_ip'])['timestamp'].shift()
	th = datetime.timedelta(days=0, seconds=0, minutes=5)
	df = df[df['prev_time'].isnull() | df['timestamp'].sub(df['prev_time']).gt(th)]
	df = df.drop(['prev_time'], axis=1)

	df = df.reset_index(drop=True)

	if TIMESTAMP:
		print(f"\t\t\tobtain slice of DF: {df.shape} within timeframe: {TIMESTAMP[0]} - {TIMESTAMP[1]}")
		df_ts = df[ df.timestamp.dt.strftime('%H:%M:%S').between(TIMESTAMP[0], TIMESTAMP[1]) ]		
		df_ts = df_ts.reset_index(drop=True)
		return df_ts

	return df

def checking_(url):
	#print(f"\t\tValidation & Update")
	try:
		r = requests.get(url)
		r.raise_for_status()
		#print(f">> HTTP family: {r.status_code} => Exists: {r.ok}")
		#print(r.headers)
		#print()
		return r
	except requests.exceptions.HTTPError as ehttp: # not 200 : not ok!
		#print(url)
		print(f"\t{ehttp}\t{ehttp.response.status_code}")
		return
		#pass
	except (requests.exceptions.Timeout,
					requests.exceptions.ConnectionError, 
					requests.exceptions.RequestException, 
					requests.exceptions.TooManyRedirects,
					requests.exceptions.InvalidSchema,
					Exception, 
					ValueError, 
					TypeError, 
					EOFError, 
					RuntimeError,
					) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args} | {url}")
		return
	except Exception as e:
		logging.exception(e)
		return

def make_folder(folder_name="MUST_BE_RENAMED"):
	if not os.path.exists(folder_name): 
		#print(f"\n>> Creating DIR:\n{folder_name}")
		os.makedirs( folder_name )

def save_vocab(vb, fname:str=""):
	print(f"<<=!=>> Saving {len(vb)} BoWs:\n{fname} ...")
	with open(fname, "w") as fw:
		json.dump(vb, fw, indent=4, ensure_ascii=False)

def save_pickle(pkl, fname:str=""):
	dump_file_name = fname
	print(f"<<<=!=>>> Saving {type(pkl)} might take a while...\n{dump_file_name}")
	st_t = time.time()
	with open(dump_file_name , "wb" ) as f:
		#joblib.dump(pkl, f, compress='lz4', protocol=pickle.HIGHEST_PROTOCOL) # df_preprocessed.lz4 must be rmoved and saved again with this package!
		dill.dump(pkl, f) # df_preprocessed.lz4 must be rmoved and saved again with this package!
	fsize_dump = os.stat( dump_file_name ).st_size / 1e6
	print(f"<Elapsed_t: {time.time()-st_t:.3f}> | {fsize_dump:.2f} MB".center(110, " "))

def load_pickle(fpath:str):
	print(f"\nfile: {fpath} exists, loading...")
	st_t = time.time()
	with open(fpath, "rb") as f:
		#pkl = joblib.load(f)
		pkl = dill.load(f)
	fsize = os.stat( fpath ).st_size / 1e6
	print(f"Elapsed_t: {time.time()-st_t:.2f} s | {type(pkl)} | {fsize:.2f} MB".center(110, " "))
	return pkl

def get_parsed_url_parameters(inp_url):
	#print(f"\nParsing {inp_url}")
	
	p_url = urllib.parse.urlparse(inp_url)
	#print(parsed_url)

	#print(f">> Explore url parameters ...")
	params = urllib.parse.parse_qs( p_url.query, keep_blank_values=True)
	#print(parameters)
	return p_url, params

def get_query_phrase(inp_url):
	#print(f"\nParsing {inp_url}")
	
	p_url = urllib.parse.urlparse(inp_url)
	#print(parsed_url)

	#print(f">> Explore url parameters ...")
	params = urllib.parse.parse_qs( p_url.query, keep_blank_values=True)
	#print(parameters)
	return params.get("query")

def just_test_for_expected_results(df):
	df_cleaned = df.dropna(axis=0, how="any", subset=["query_word"]).reset_index(drop=True)
	print("#"*100)
	idx = np.random.choice(df_cleaned.shape[0]+1)
	print(f"\n>> search results of sample: {idx}")
	
	with pd.option_context('display.max_colwidth', 500):
		print(df_cleaned.loc[idx, ["user_ip", "query_word", "referer"]])

	one_result = df_cleaned.loc[idx, "search_results"]
	
	#print(json.dumps(one_result, indent=1, ensure_ascii=False))
	for k, v in one_result.items():
		print(k)
		print(one_result.get(k).get("newspaper_snippet"))
		print(len(one_result.get(k).get("newspaper_snippet_highlighted_words")), one_result.get(k).get("newspaper_snippet_highlighted_words"))
		print()
		print(one_result.get(k).get("newspaper_content_ocr"))
		print(len(one_result.get(k).get("newspaper_content_ocr_highlighted_words")), one_result.get(k).get("newspaper_content_ocr_highlighted_words"))
		print("-"*100)

def get_concat_df(dir_path: str=dfs_path):
	# loop over all files.dump located at:
	# dir_path: /scratch/project_2004072/Nationalbiblioteket/datasets/
	for files in glob.glob(os.path.join(dir_path, "*.dump")):
		print(files, files.endswith(".dump"))
	print(f">> Concatinating files.dump located at: {dir_path}", end=" ")
	with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
		dfs = [load_pickle(f) for f in glob.glob(os.path.join(dir_path, "*.dump")) ]
	print(len(dfs))
	"""
	df_concat=pd.concat(dfs,
										 #ignore_index=True,
										 ).sort_values("timestamp", ignore_index=True)
	return df_concat
	"""