import os
import subprocess
import urllib
import requests
import joblib
import re
import json
import argparse
import datetime
import glob
import webbrowser
import string
import time

import numpy as np
import pandas as pd
from natsort import natsorted

usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket",
				}

NLF_DATASET_PATH = usr_[os.environ['USER']]

dpath = os.path.join( NLF_DATASET_PATH, f"NLF_Pseudonymized_Logs" )
#dpath = os.path.join( NLF_DATASET_PATH, f"no_ip_logs" )
#dpath = os.path.join( NLF_DATASET_PATH, f"broken" )

rpath = os.path.join( NLF_DATASET_PATH, f"results" )
dfs_path = os.path.join( NLF_DATASET_PATH, f"dataframes" )

def make_result_dir(infile=""):
	f = infile.split("/")[-1]
	#print(f)
	f = f[:f.rfind(".log")]
	#print(f)
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

def make_folder(folder_name="MUST_BE_RENAMED"):
	if not os.path.exists(folder_name): 
		#print(f"\n>> Creating DIR:\n{folder_name}")
		os.makedirs( folder_name )

def save_(df, infile="", saving_path="", save_csv=False, save_parquet=True):
	dfs_dict = {
		f"{infile}":	df,
	}

	dump_file_name = os.path.join(dfs_path, f"{infile}.dump")
	print(f"\n>> Saving {dump_file_name} ...")
	joblib.dump(	dfs_dict, 
								dump_file_name,
								compress='lz4', # zlib more info: https://joblib.readthedocs.io/en/latest/auto_examples/compressors_comparison.html#sphx-glr-auto-examples-compressors-comparison-py
								)
	fsize_dump = os.stat( dump_file_name ).st_size / 1e6
	print(f"\t\t{fsize_dump:.1f} MB")

	if save_csv:
		csv_file_name = os.path.join(dfs_path, f"{infile}.csv")
		print(f"\n>> Saving {csv_file_name} ...")
		df.to_csv(csv_file_name, index=False)
		fsize_csv = os.stat( csv_file_name ).st_size / 1e6
		print(f"\t\t{fsize_csv:.1f} MB")

	if save_parquet:
		parquet_file_name = os.path.join(dfs_path, f"{infile}.parquet")
		print(f"\n>> Saving {parquet_file_name} ...")
		df.to_parquet(parquet_file_name)
		fsize_parquet = os.stat( parquet_file_name ).st_size / 1e6
		print(f"\t\t{fsize_parquet:.1f} MB")
		print(f">>>> To Read\tDF = pd.read_parquet({parquet_file_name})")

def load_df(infile=""):
	#fpath = os.path.join(dfs_path, f"{infile}.dump")
	fpath = infile
	fsize = os.stat( fpath ).st_size / 1e9
	print(f"Loading {fpath} | {fsize:.3f} GB")
	st_t = time.time()
	df_dict = joblib.load(fpath)
	#print(list(df_dict.keys()))# dict:{key(nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log) : value (df)}
	print(f"\t\tElapsed_t: {time.time() - st_t:.2f} s")
	df = df_dict[list(df_dict.keys())[0]]
	return df	

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
