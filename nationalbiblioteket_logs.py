import os
import re
import datetime
import glob
import urllib
from urllib3 import Timeout
import requests
import webbrowser
import string
import sys
import joblib
import time
import argparse

from bs4 import BeautifulSoup
from natsort import natsorted
import numpy as np
import pandas as pd

# Apache access log format:
# 
# %h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i\
# Ex)
# 172.16.0.3 - - [25/Sep/2002:14:04:19 +0200] "GET / HTTP/1.1" 401 - "" "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.1) Gecko/20020827"
# https://regex101.com/r/xDfSqj/4

# two files with corrupted links: removed entirely the rows:

# fname = "nike6.docworks.lib.helsinki.fi_access_log.2017-02-02.log"
# - - [02/Feb/2017:06:42:13 +0200] "GET /images/KK_BLUE_mark_small.gif HTTP/1.1" 206 2349 "http://digi.kansalliskirjasto.fi/"Kansalliskirjasto" "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)" 5
	
# fname = "nike5.docworks.lib.helsinki.fi_access_log.2017-02-02.log"
# - - [02/Feb/2017:06:42:13 +0200] "GET /images/KK_BLUE_mark_small.gif HTTP/1.1" 206 2349 "http://digi.kansalliskirjasto.fi/"Kansalliskirjasto" "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)" 16

parser = argparse.ArgumentParser(description='National Library of Finland (NLF)')
parser.add_argument('--query', default=9, type=int) # smallest

args = parser.parse_args()

usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket",
				}

NLF_DATASET_PATH = usr_[os.environ['USER']]

dpath = os.path.join( NLF_DATASET_PATH, f"no_ip_logs" )
#dpath = os.path.join( NLF_DATASET_PATH, f"broken" )

rpath = os.path.join( NLF_DATASET_PATH, f"results" )
dfs_path = os.path.join( NLF_DATASET_PATH, f"dataframes" )

def make_folders():
	if not os.path.exists(rpath): 
		#print(f"\n>> Creating DIR:\n{rpath}")
		os.makedirs( rpath )

	if not os.path.exists(dfs_path): 
		#print(f"\n>> Creating DIR:\n{dfs_path}")
		os.makedirs( dfs_path )

def convert_date(INP_DATE):
	months_dict = {
		"Jan": "01", 
		"Feb": "02", 
		"Mar": "03", 
		"Apr": "04", 
		"May": "05", 
		"Jun": "06", 
		"Jul": "07", 
		"Aug": "08", 
		"Sep": "09", 
		"Oct": "10", 
		"Nov": "11", 
		"Dec": "12", 
		}

	d_list = INP_DATE.split("/")
	##print(d_list)
	d_list[1] = months_dict.get(d_list[1])
	MODIDFIED_DATE = '/'.join(d_list)
	##print(MODIDFIED_DATE)

	yyyy_mm_dd = datetime.datetime.strptime(MODIDFIED_DATE, "%d/%m/%Y").strftime("%Y-%m-%d")
	return yyyy_mm_dd

def checking_(url):
	#print(f"\t\tValidation & Update")
	try:
		r = requests.get(url)
		r.raise_for_status()
		#print(r.status_code, r.ok)
		return r
	except requests.exceptions.ConnectionError as ec:
		print(url)
		print(f"Connection Exception: {ec}")
		pass
	except requests.exceptions.Timeout as et:
		print(url)
		print(f"Timeout Exception: {et}")
		pass
	except requests.exceptions.HTTPError as ehttp: # not 200 : not ok!
		print(url)
		print(f"HTTP Exception: {ehttp}\t{ehttp.response.status_code}")
		pass
	except requests.exceptions.RequestException as e:
		print(url)
		print(f"ALL Exception: {e}")
		pass

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
			##print(l)

			#dt_tz = l[0].replace("[", "").replace("]", "")
			#DDMMYYYY = dt_tz[:dt_tz.find(":")]
			#YYYYMMDD = convert_date( DDMMYYYY )
			#HMS = dt_tz[dt_tz.find(":")+1:dt_tz.find(" ")]
			#TZ = dt_tz[dt_tz.find(" ")+1:]

			cleaned_lines.append({
				#"timestamp": 									l[0], # original: 01/Feb/2017:12:34:51 +0200
				"timestamp": 										l[0].replace(":", " ", 1),
				"client_request_line": 					l[1],
				"status": 											l[2],
				"bytes_sent": 									l[3],
				"referer": 											l[4],
				"user_agent": 									l[5],
				"session_id": 									l[6],
				"query_word":										np.nan,
				"term":													np.nan,
				"OCR":													np.nan,
				"fuzzy":												np.nan,
				"has_metadata":									np.nan,
				"has_illustration":							np.nan,
				"show_unauthorized_results":		np.nan,
				"pages":												np.nan,
				"import_time":									np.nan,
				"collection":										np.nan,
				"author":												np.nan,
				"keyword":											np.nan,
				"publication_place":						np.nan,
				"language":											np.nan,
				"document_type":								np.nan,
				"show_last_page":								np.nan,
				"order_by":											np.nan,
				"publisher":										np.nan,
				"start_date":										np.nan,
				"end_date":											np.nan,
				"require_all_keywords":					np.nan,
				"result_type":									np.nan,
				#"date": 								YYYYMMDD,
				#"date": 								pd.to_datetime(l[0].replace(":", " ", 1) ).to_period('D'),
				#"date": 								pd.to_datetime(l[0].replace(":", " ", 1) ).strftime('%Y-%m-%d'),
				#"time": 								HMS,
				#"time":									pd.to_datetime(l[0].replace(":", " ", 1) ).strftime('%H:%M:%S'),
				#"timezone": 						TZ,
				})

	df = pd.DataFrame.from_dict(cleaned_lines)

	# with pandas:
	df.timestamp = pd.to_datetime(df.timestamp)
	df = df.replace("null", "-", regex=True).replace("-", pd.NA, regex=True).replace(r'^\s*$', pd.NA, regex=True)
	
	# with numpy:
	#df = df.replace("-", pd.NA, regex=True).replace(r'^\s*$', np.nan, regex=True)
	
	if TIMESTAMP:
		#print(f"\t\t\twithin timeframe: {TIMESTAMP[0]} - {TIMESTAMP[1]}")
		df_ts = df[ df.timestamp.dt.strftime('%H:%M:%S').between(TIMESTAMP[0], TIMESTAMP[1]) ]		
		df_ts = df_ts.reset_index(drop=True)
		return df_ts

	return df

def single_query(file_="", ts=None, browser_show=False):
	#print(f">> Analyzing a single query of {file_}")
	df = get_df_no_ip_logs(infile=file_, TIMESTAMP=ts)	
	#print(f"df: {df.shape}")
	
	"""
	#print(list(df.columns))
	#print("-"*180)
	#print(df.info(verbose=True, memory_usage="deep"))
	#print("-"*180)
	#print(f"\n\n>> NaN referer: {df['referer'].isna().sum()} / {df.shape[0]}\n\n")
	#print("#"*180)
	
	#print( df.head(40) )
	#print("-"*130)
	#print( df.tail(40) )
	"""
	#print(f">> Generating a sample query...")
	qlink = int(np.random.randint(0, high=df.shape[0]+1, size=1))
	#qlink = 52 # no "page="" in query
	#qlink = 1402 # google.fi # no "page="" in query
	#qlink = 5882 # ERROR due to login credintial
	#qlink = 277939 # MARC is not available for this page.
	#qlink = 231761 # 5151 # shows text content with ocr although txt_ocr returns False
	#qlink = 104106 # 55901 # indirect search from google => no "page="" in query
	#qlink = 227199 # 349904 # 6462 #  # with txt ocr available & "page="" in query
	#qlink = 103372 # broken link
	#qlink = 158 # short link
	#qlink = 21888 # broken connection
	#qlink = 30219 #30219 # 15033 #  # "" nothing in url
	#qlink = 96624 # both referer and user_agent pd.NA
	#qlink = 10340 # read timeout error in puhti when updating with session & head ?
	#print(df.iloc[qlink])
	single_url = df["referer"][qlink]
	#print(f">> Q: {qlink} : {single_url}")

	if single_url is pd.NA:
		#print(f">> no link is available! => exit")
		return

	r = checking_(single_url)
	if r is None:
		return

	single_url = r.url

	if browser_show:
		#print(f"\topening in browser... ")
		webbrowser.open(single_url, new=2)
	
	#print(f"Parsing {single_url}")
	parsed_url = urllib.parse.urlparse(single_url)
	#print(parsed_url)
	
	#print(f">> Explore url parameters ...")
	parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
	#print(parameters)
	
	#print("#"*130)
	#print(f">> 'search' in path: {parsed_url.path} ?")
	SRCH_PARAM = True if "search" in parsed_url.path else False
	#print(SRCH_PARAM)

	#print(f">> 'page=' in query: {parsed_url.query} ?")
	PG_PARAM = True if "page=" in parsed_url.query else False
	#print(PG_PARAM)
	#print("#"*130)

	# query_word & term:
	df.loc[qlink, "query_word"] = ",".join(parameters.get("query")) if parameters.get("query") else np.nan
	df.loc[qlink, "term"] = ",".join(parameters.get("term")) if parameters.get("term") else np.nan

	if PG_PARAM and not SRCH_PARAM: # possible OCR existence
		txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
		#print(f">> page-X.txt available?\t{txt_pg_url}\t")

		text_response = checking_(txt_pg_url)
		if text_response is not None:
			#print(f"\t\t\tYES >> loading...\n")
			df.loc[qlink, "OCR"] = text_response.text
	
	#print(list(df.columns))
	#print(df.shape)
	#print(df.isna().sum())
	#print(df.info(verbose=True, memory_usage="deep"))
	save_(df, infile=f"SINGLEQuery_timestamp_{ts}_{file_}")

def all_queries(file_="", ts=None):
	#print(f">> Analyzing queries of {file_} ...")
	df = get_df_no_ip_logs(infile=file_, TIMESTAMP=ts)	
	#print(f"df: {df.shape}")
	
	"""
	#print(list(df.columns))
	#print("-"*180)
	#print(df.info(verbose=True, memory_usage="deep"))
	#print("-"*180)
	#print(f"\n\n>> NaN referer: {df['referer'].isna().sum()} / {df.shape[0]}\n\n")
	#print("#"*180)
	
	#print( df.head(40) )
	#print("-"*130)
	#print( df.tail(40) )
	"""
	#print(df)
	#print("#"*100)
	def analyze_(df):
		in_url = df.referer
		#print(f"URL: {in_url}")

		if in_url is pd.NA:
			##print(f">> no link is available! => exit")
			return df
		
		r = checking_(in_url)
		if r is None:
			return df

		in_url = r.url

		#print(f"\nParsing {in_url}")
		parsed_url = urllib.parse.urlparse(in_url)
		#print(parsed_url)

		#print(f">> Explore url parameters ...")
		parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
		#print(parameters)
		
		#print(f">> Search Page?")
		SRCH_PARAM = True if "search" in parsed_url.path else False
		#print(SRCH_PARAM)

		#print(f">> Page Parameter?")
		PG_PARAM = True if "page=" in parsed_url.query else False
		#print(PG_PARAM)

		# all features:
		df["query_word"] = ",".join(parameters.get("query")) if parameters.get("query") else np.nan
		df["term"] = ",".join(parameters.get("term")) if parameters.get("term") else np.nan
		# ORC updating in the next for loop
		df["fuzzy"] = ",".join(parameters.get("fuzzy")) if parameters.get("fuzzy") else np.nan
		df["has_metadata"] = ",".join(parameters.get("lang")) if parameters.get("lang") else np.nan
		df["has_illustration"] = ",".join(parameters.get("hasIllustrations")) if parameters.get("hasIllustrations") else np.nan
		df["show_unauthorized_results"] = ",".join(parameters.get("showUnauthorizedResults")) if parameters.get("showUnauthorizedResults") else np.nan
		df["pages"] = ",".join(parameters.get("pages")) if parameters.get("pages") else np.nan
		df["import_time"] = ",".join(parameters.get("importTime")) if parameters.get("importTime") else np.nan
		df["collection"] = ",".join(parameters.get("collection")) if parameters.get("collection") else np.nan
		df["author"] = ",".join(parameters.get("author")) if parameters.get("author") else np.nan
		df["keyword"] = ",".join(parameters.get("tag")) if parameters.get("tag") else np.nan
		df["publication_place"] = ",".join(parameters.get("publicationPlace")) if parameters.get("publicationPlace") else np.nan
		df["language"] = ",".join(parameters.get("lang")) if parameters.get("lang") else np.nan
		df["document_type"] = ",".join(parameters.get("formats")) if parameters.get("formats") else np.nan
		df["show_last_page"] = ",".join(parameters.get("showLastPage")) if parameters.get("showLastPage") else np.nan
		df["order_by"] = ",".join(parameters.get("orderBy")) if parameters.get("orderBy") else np.nan
		df["publisher"] = ",".join(parameters.get("publisher")) if parameters.get("publisher") else np.nan
		df["start_date"] = ",".join(parameters.get("startDate")) if parameters.get("startDate") else np.nan
		df["end_date"] = ",".join(parameters.get("endDate")) if parameters.get("endDate") else np.nan
		df["require_all_keywords"] = ",".join(parameters.get("requireAllKeywords")) if parameters.get("requireAllKeywords") else np.nan
		df["result_type"] = ",".join(parameters.get("resultType")) if parameters.get("resultType") else np.nan

		# OCR extraction:
		if PG_PARAM and not SRCH_PARAM:
			txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
			#print(f">> page-X.txt available?\t{txt_pg_url}\t")
			
			text_response = checking_(txt_pg_url)
			
			if text_response is not None: # 200
				#print(f"\t\t\tYES >> loading...\n")
				#return text_response.text
				df["OCR"] = text_response.text
				
		return df
	
	check_urls = lambda INPUT_DF: analyze_(INPUT_DF)
	df = pd.DataFrame( df.apply( check_urls, axis=1, ) )

	print("*"*205)
	print(df.head(60))
	print("#"*150)
	print(df.info(verbose=True, memory_usage="deep"))
	print("#"*150)
	print(f"NaN OCR: {df['OCR'].isna().sum()} / {df.shape[0]}")
	print(f"NaN query_word: {df['query_word'].isna().sum()} / {df.shape[0]}")
	print(f"NaN term: {df['term'].isna().sum()} / {df.shape[0]}")
	cols = list(df.columns)
	print(len(cols), cols, df.shape)
	print("#"*150)
	save_(df, infile=file_)

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

def get_log_files():
	log_files_dir = natsorted( glob.glob( os.path.join(dpath, "*.log") ) )
	##print(log_files_dir)

	log_files_date = [lf[ lf.rfind(".2")+1: lf.rfind(".") ] for lf in log_files_dir]
	##print(len(log_files_date), log_files_date)
	log_files = [lf[ lf.rfind("/")+1: ] for lf in log_files_dir]
	##print(log_files)

	return log_files

def get_query_log(QUERY=0):
	print(f">> Given log file index: {QUERY}")
	log_files_dir = natsorted( glob.glob( os.path.join(dpath, "*.log") ) )
	##print(log_files_dir)

	log_files_date = [lf[ lf.rfind(".2")+1: lf.rfind(".") ] for lf in log_files_dir]
	##print(len(log_files_date), log_files_date)
	all_log_files = [lf[ lf.rfind("/")+1: ] for lf in log_files_dir]
	##print(log_files)
	query_log_file = all_log_files[QUERY] 
	print(f"\t\t {query_log_file}")

	return query_log_file

def run():
	"""	
	single_query(file_=get_query_log(QUERY=args.query), 
							browser_show=True, 
							#ts=["23:52:00", "23:59:59"],
							)
	"""
	# run all log files using array in batch
	all_queries(file_=get_query_log(QUERY=args.query),
							ts=["23:40:00", "23:59:39"],
							)
	
	#print(f"\t\tCOMPLETED!")
	
if __name__ == '__main__':
	os.system('clear')
	make_folders()
	run()