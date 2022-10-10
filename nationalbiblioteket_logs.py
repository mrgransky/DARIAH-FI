import os
import re
import datetime
import glob
import urllib
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
parser.add_argument('--query', default=6, type=int) # smallest

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

if not os.path.exists(rpath): 
	print(f"\n>> Creating DIR:\n{rpath}")
	os.makedirs( rpath )

if not os.path.exists(dfs_path): 
	print(f"\n>> Creating DIR:\n{dfs_path}")
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
	#print(d_list)
	d_list[1] = months_dict.get(d_list[1])
	MODIDFIED_DATE = '/'.join(d_list)
	#print(MODIDFIED_DATE)

	yyyy_mm_dd = datetime.datetime.strptime(MODIDFIED_DATE, "%d/%m/%Y").strftime("%Y-%m-%d")
	return yyyy_mm_dd

def update_url(INP_URL):
	print(f">> Updating ...")
	#session = requests.Session()  # so connections are recycled
	#r = session.head(INP_URL, allow_redirects=True)
	try:
		#r = requests.get(INP_URL, timeout=120) # wait 120s for (connection, read)
		r = requests.get(INP_URL, timeout=None) # wait forever for (connection, read)
		updated_url = r.url
		history_url = r.history

		return updated_url, history_url

	except requests.exceptions.Timeout:
		print(f">> Timeout Exception! => return original url")
		return INP_URL, None

	#updated_url = r.url
	#history_url = r.history

	#return updated_url, history_url
	
def broken_connection(url):
	try:
		requests.get(url, timeout=None)
		return False
	except requests.exceptions.ConnectionError:
		print ("Broken url => Connection Error!!!!")
		return True

def get_df_no_ip_logs(infile="", TIMESTAMP=None):
	file_path = os.path.join(dpath, infile)

	print(f">> Reading {file_path} ...")
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (?P<status>\d{3}) (.*) "([^"]*)" "(.*?)" (.*)' # original working!
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^"]+)" "(.*?)" (.*)' # original working!
	ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^\"]*)" "(.*?)" (.*)' # checked with all log files!
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "(?:-|.*(http://\D.*))" "(.*?)" (.*)'
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "(?:|-|.*(://\D.*))" "(.*?)" (.*)'
	cleaned_lines = []
	with open(file_path, mode="r") as f:
		for line in f:
			#print(line)
			matched_line = re.match(ACCESS_LOG_PATTERN, line)
			#print (matched_line)
			l = matched_line.groups()
			#print(l)

			#dt_tz = l[0].replace("[", "").replace("]", "")
			#DDMMYYYY = dt_tz[:dt_tz.find(":")]
			#YYYYMMDD = convert_date( DDMMYYYY )
			#HMS = dt_tz[dt_tz.find(":")+1:dt_tz.find(" ")]
			#TZ = dt_tz[dt_tz.find(" ")+1:]

			cleaned_lines.append({
				#"timestamp": 						l[0], # original: 01/Feb/2017:12:34:51 +0200
				"timestamp": 						l[0].replace(":", " ", 1),
				"client_request_line": 	l[1],
				"status": 							l[2],
				"bytes_sent": 					l[3],
				"referer": 							l[4],
				"user_agent": 					l[5],
				"session_id": 					l[6], #TODO: must be check if key is a right name!
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
	df = df.replace("-", pd.NA, regex=True).replace(r'^\s*$', pd.NA, regex=True)
	
	# with numpy:
	#df = df.replace("-", pd.NA, regex=True).replace(r'^\s*$', np.nan, regex=True)
	
	if TIMESTAMP:
		print(f"\t\t\twithin timeframe: {TIMESTAMP[0]} - {TIMESTAMP[1]}")
		df_ts = df[ df.timestamp.dt.strftime('%H:%M:%S').between(TIMESTAMP[0], TIMESTAMP[1]) ]		
		df_ts = df_ts.reset_index(drop=True)
		return df_ts

	return df

def single_query(file_="", ts=None, browser_show=False):
	ocr_txt = np.nan

	print(f">> Analyzing a single query of {file_}")
	df = get_df_no_ip_logs(infile=file_, TIMESTAMP=ts)	
	print(f"df: {df.shape}")
	
	"""
	print(list(df.columns))
	print("-"*180)
	print(df.info(verbose=True, memory_usage="deep"))
	print("-"*180)
	print(f"\n\n>> NaN referer: {df['referer'].isna().sum()} / {df.shape[0]}\n\n")
	print("#"*180)
	
	print( df.head(40) )
	print("-"*130)
	print( df.tail(40) )
	"""
	print(f">> Generating a sample query...")
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
	print(df.iloc[qlink])
	s_url = df["referer"][qlink] #if df["referer"][qlink].notnull() else None
	print(f">> Q: {qlink} : {s_url}")

	if s_url is pd.NA:
		print(f">> no link is available! => exit")
		return 0

	if broken_connection(s_url):
		print(f"\t\tBROKEN => exit")
		return 0

	print(f">> updating url ...")
	s_url, h_url = update_url(s_url)
	print(f"{h_url}\n{s_url}")

	if browser_show:
		print(f"\topening in browser... ")
		webbrowser.open(s_url, new=2)
	
	print(f">> Parsing url ...")
	parsed_url = urllib.parse.urlparse(s_url)
	print(parsed_url)
	
	print(f">> Explore url parameters ...")
	parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
	print(parameters)
	
	print("#"*130)
	print(f">> 'search' in path: {parsed_url.path} ?")
	SRCH_PARAM = True if "search" in parsed_url.path else False
	print(SRCH_PARAM)

	print(f">> 'page=' in query: {parsed_url.query} ?")
	PG_PARAM = True if "page=" in parsed_url.query else False
	print(PG_PARAM)
	print("#"*130)

	if PG_PARAM and not SRCH_PARAM: # possible OCR existence
		txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
		print(f">> page-X.txt available?\t{txt_pg_url}\t")
		txt_resp = requests.get(txt_pg_url)
		
		print(f"\t\t\t{txt_resp.status_code} => {txt_resp.ok}") # 200: ok, 400: bad_request, 403: forbidden, 404: not_found

		if txt_resp.ok: # 200
			print(f"\t\t\tYES >> loading...\n")
			ocr_txt = txt_resp.text
			#print(ocr_txt)
	
	df.loc[qlink, "OCR"] = ocr_txt
	print(list(df.columns))
	print(df.shape)
	print(df.isna().sum())
	print(df.info(verbose=True, memory_usage="deep"))
	save_(df, infile=f"SINGLEQuery_timestamp_{ts}_{file_}")

def all_queries(file_="", ts=None):
	ocr_txt = np.nan

	print(f">> Analyzing queries of {file_} ...")
	df = get_df_no_ip_logs(infile=file_, TIMESTAMP=ts)	
	print(f"df: {df.shape}")
	
	"""
	print(list(df.columns))
	print("-"*180)
	print(df.info(verbose=True, memory_usage="deep"))
	print("-"*180)
	print(f"\n\n>> NaN referer: {df['referer'].isna().sum()} / {df.shape[0]}\n\n")
	print("#"*180)
	
	print( df.head(40) )
	print("-"*130)
	print( df.tail(40) )
	"""
	def analyze_(in_url):
		print(f"URL: {in_url}")

		if in_url is pd.NA:
			#print(f">> no link is available! => exit")
			return 0
		
		if broken_connection(in_url):
			#print(f"\t\tBROKEN => exit")
			return 0

		in_url, h_url = update_url(in_url)
		print(f"{h_url}\n{in_url}")

		print(f">> Parsing url ...")
		parsed_url = urllib.parse.urlparse(in_url)
		print(parsed_url)

		print(f">> Explore url parameters ...")
		parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
		print(parameters)
		
		print(f">> Search Page?")
		SRCH_PARAM = True if "search" in parsed_url.path else False
		print(SRCH_PARAM)

		print(f">> Page Parameter?")
		PG_PARAM = True if "page=" in parsed_url.query else False
		print(PG_PARAM)

		if PG_PARAM and not SRCH_PARAM:
			txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
			print(f">> page-X.txt available?\t{txt_pg_url}\t")
			txt_resp = requests.get(txt_pg_url)

			print(f"\t\t\t{txt_resp.status_code} => {txt_resp.ok}") # 200: ok, 400: bad_request, 403: forbidden, 404: not_found

			if txt_resp.ok: # 200
				print(f"\t\t\tYES >> loading...\n")
				return txt_resp.text

	check_urls = lambda x: analyze_(x)
	
	# cleaning
	#df["ocr_text"] = pd.DataFrame( df.referer.apply( check_urls ) )
	df["OCR"] = pd.DataFrame( df.referer.apply( check_urls ) )
	#df.loc[qlink, "OCR"] = ocr_txt

	print(list(df.columns))
	print("#"*150)
	print(df.head(50))
	print("#"*150)
	print(df.info(verbose=True, memory_usage="deep"))
	print("#"*150)
	print(f"\n\n>> NaN referer: {df['OCR'].isna().sum()} / {df.shape[0]}\n\n")
	save_(df, infile=file_)

def save_(df, infile=""):
	dfs_dict = {
		f"{infile}":	df,
	}
	
	dump_file_name = os.path.join(dfs_path, f"{infile}.dump")
	print(f">> Saving {dump_file_name} ...")
	print(f"\tSaving...")
	
	

	joblib.dump(	dfs_dict, 
								dump_file_name,
								#os.path.join( dfs_path, f"{fname}" ),
								compress='lz4', # zlib more info: https://joblib.readthedocs.io/en/latest/auto_examples/compressors_comparison.html#sphx-glr-auto-examples-compressors-comparison-py
								)
	#fsize = os.stat( os.path.join( dfs_path, f"{fname}" ) ).st_size / 1e6
	fsize = os.stat( dump_file_name ).st_size / 1e6

	print(f"\t\t{fsize:.1f} MB")

def get_log_files():
	log_files_dir = natsorted( glob.glob( os.path.join(dpath, "*.log") ) )
	#print(log_files_dir)

	log_files_date = [lf[ lf.rfind(".2")+1: lf.rfind(".") ] for lf in log_files_dir]
	#print(len(log_files_date), log_files_date)
	log_files = [lf[ lf.rfind("/")+1: ] for lf in log_files_dir]
	#print(log_files)

	return log_files

def get_query_log(QUERY=0):
	print(f">> Given log file index: {QUERY}")
	log_files_dir = natsorted( glob.glob( os.path.join(dpath, "*.log") ) )
	#print(log_files_dir)

	log_files_date = [lf[ lf.rfind(".2")+1: lf.rfind(".") ] for lf in log_files_dir]
	#print(len(log_files_date), log_files_date)
	all_log_files = [lf[ lf.rfind("/")+1: ] for lf in log_files_dir]
	#print(log_files)
	query_log_file = all_log_files[QUERY] 
	print(f"\t\t {query_log_file}")

	return query_log_file

def run():
	"""
	single_query(file_=get_query_log(QUERY=args.query), 
							browser_show=False, 
							#ts=["23:52:00", "23:59:59"],
							)
	"""

	# run all log files using array in batch
	all_queries(file_=get_query_log(QUERY=args.query),
							#ts=["23:52:00", "23:59:59"],
							)

	

	print(f"\t\tCOMPLETED!")
	
if __name__ == '__main__':
	os.system('clear')
	run()