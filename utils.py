import os
import urllib
import requests
import joblib
import re
import numpy as np
import pandas as pd

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
	
	#df = df.replace("null", "-", regex=True).replace("-", pd.NA, regex=True).replace(r'^\s*$', pd.NA, regex=True)
	
	# with numpy:
	df = df.replace("-", np.nan, regex=True).replace(r'^\s*$', np.nan, regex=True).replace(r'http://\d+', np.nan, regex=True)
	df = df.dropna(axis=0)
	df = df.reset_index(drop=True)
	
	if TIMESTAMP:
		print(f"\t\t\twithin timeframe: {TIMESTAMP[0]} - {TIMESTAMP[1]}")
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
		return r
	except requests.exceptions.ConnectionError as ec:
		#print(url)
		print(f">> {url}\tConnection Exception: {ec}")
		pass
	except requests.exceptions.Timeout as et:
		#print(url)
		print(f">> {url}\tTimeout Exception: {et}")
		pass
	except requests.exceptions.HTTPError as ehttp: # not 200 : not ok!
		#print(url)
		print(f"HTTP Exception: {ehttp}\t{ehttp.response.status_code}")
		pass
	except requests.exceptions.RequestException as e:
		#print(url)
		print(f">> {url}\tALL Exception: {e}")
		pass

def make_folders():
	if not os.path.exists(rpath): 
		#print(f"\n>> Creating DIR:\n{rpath}")
		os.makedirs( rpath )

	if not os.path.exists(dfs_path): 
		#print(f"\n>> Creating DIR:\n{dfs_path}")
		os.makedirs( dfs_path )

def save_(df, infile=""):
	dfs_dict = {
		f"{infile}":	df,
	}
	
	dump_file_name = os.path.join(dfs_path, f"{infile}.dump")
	csv_file_name = os.path.join(dfs_path, f"{infile}.csv")
	
	print(f">> Saving {csv_file_name} ...")
	df.to_csv(csv_file_name, index=False)
	fsize_csv = os.stat( csv_file_name ).st_size / 1e6
	print(f"\t\t{fsize_csv:.1f} MB")

	print(f">> Saving {dump_file_name} ...")
	joblib.dump(	dfs_dict, 
								dump_file_name,
								compress='lz4', # zlib more info: https://joblib.readthedocs.io/en/latest/auto_examples/compressors_comparison.html#sphx-glr-auto-examples-compressors-comparison-py
								)
	fsize_dump = os.stat( dump_file_name ).st_size / 1e6
	print(f"\t\t{fsize_dump:.1f} MB")

def get_parsed_url_parameters(inp_url):
	#print(f"\nParsing {inp_url}")
	p_url = urllib.parse.urlparse(inp_url)
	#print(parsed_url)

	#print(f">> Explore url parameters ...")
	params = urllib.parse.parse_qs( p_url.query, keep_blank_values=True)
	#print(parameters)

	return p_url, params

def get_np_ocr(ocr_url):
	parsed_url, parameters = get_parsed_url_parameters(ocr_url)
	txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
	#print(f">> page-X.txt available?\t{txt_pg_url}\t")
	text_response = checking_(txt_pg_url)
	if text_response is not None: # 200
		#print(f"\t\t\tYES >> loading...\n")
		return text_response.text