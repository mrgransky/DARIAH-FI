import os
import re
import datetime
import glob
import urllib
import requests
import webbrowser

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


usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket/no_ip_logs',
				#'alijanif':	'/scratch/project_2004072/Nationalbiblioteket/broken',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket/no_ip_logs",
				#"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket/broken",
				}

dpath = usr_[os.environ['USER']]

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

def get_df_no_ip_logs(infile=""):
	print(f"\n\n>> Loading {infile} ...")
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^\"]+)" "(.*?)" (\d+)'
	ACCESS_LOG_PATTERN = '- - (?P<time>\[.*?\]) "(.*?)" (?P<status>\d{3}) (.*) "([^"]*)" "(.*?)" (.*)'
	#ACCESS_LOG_PATTERN = '- - (?P<time>\[.*?\]) "(.*?)" (?P<status>\d{3}) (.*) "([^\"]+)" "(.*?)" (.*)'

	cleaned_lines = []
	with open(infile, mode="r") as f:
		for line in f:
			#print(line)
			l = re.match(ACCESS_LOG_PATTERN, line).groups()

			dt_tz = l[0].replace("[", "").replace("]", "")
			
			DDMMYYYY = dt_tz[:dt_tz.find(":")]
			YYYYMMDD = convert_date( DDMMYYYY )
			HMS = dt_tz[dt_tz.find(":")+1:dt_tz.find(" ")]
			TZ = dt_tz[dt_tz.find(" ")+1:]

			cleaned_lines.append({
				"timestamp": 						l[0],
				"client_request_line": 	l[1],
				"status": 							l[2],
				"bytes_sent": 					l[3],
				"referer": 							l[4],
				"user_agent": 					l[5],
				"session_id": 					l[6], #TODO: must be check if key is a right name!
				"date": 								YYYYMMDD,
				"time": 								HMS,
				"timezone": 						TZ,
				})
	
	return pd.DataFrame.from_dict(cleaned_lines)

def get_single_ocr_text(df, browser_show=False):
	print(f">> Analyze single df: {df.shape}")
	
	print(f">> cleaning urls...")

	# with pandas:
	df['referer'] = df['referer'].replace("-", pd.NA, regex=True)
	df['user_agent'] = df['user_agent'].replace("-", pd.NA, regex=True)
	
	# with numpy:
	#df['referer'] = df['referer'].replace("-", np.nan, regex=True)
	#df['user_agent'] = df['user_agent'].replace("-", np.nan, regex=True)

	print(list(df.columns))
	print("-"*180)
	print(df.info(verbose=True, memory_usage="deep"))
	print("-"*180)
	print(f"\n\n>> NaN referer: {df['referer'].isna().sum()} / {df.shape[0]}\n\n")
	print("#"*180)
	
	print( df.head(40) )
	print("-"*130)
	print( df.tail(40) )

	print(f"\n>> Generating a sample query...")
	#qlink = int(np.random.randint(0, high=df.shape[0]+1, size=1))
	qlink = 52
	#qlink = 1402 # google.fi
	qlink = 6462 # 349904 # 227199  #  # with txt ocr available
	#qlink = 389518 # 4383 # no link!
	#qlink = 5882 # ERROR due to login credintial
	#qlink = 277939 # MARC is not available for this page.
	#qlink = 231761 # shows text content with ocr although txt_ocr returns False

	s_url = df["referer"][qlink] #if df["referer"][qlink].notnull() else None
	print(f"\n>> Q: {qlink} : {s_url}")

	if s_url is pd.NA:
		print(f">> no link is available! => exit")
		return 0

	print(f">> Manipulate header with user-agent...")

	if browser_show:
		print(f"\topening in browser... ")
		webbrowser.open(s_url, new=2)
	
	print(f"\n>> Parsing url ...")
	parsed_url = urllib.parse.urlparse(s_url)
	print(parsed_url)
	
	print(f">> Explore url parameters ...")
	parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
	print(parameters)
	
	print(f">> Search Parameter?")
	SRCH_PARAM = True if "search" in parsed_url.path else False
	print(SRCH_PARAM)

	print(f">> Page Parameter?")
	PG_PARAM = True if "page" in parsed_url.query else False
	print(PG_PARAM)

	print("#"*130)

	if PG_PARAM and not SRCH_PARAM:
		print(f">> Generating txt page ...")
		txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
		txt_resp = requests.get(txt_pg_url)

		print(f">> Page: {txt_pg_url} exists? {txt_resp.status_code}") # 200: ok, 400: bad_request, 403: forbidden, 404: not_found

		print(f"\n>> Loading txt page...\n")
		print(txt_resp.text)

		title_info_url = s_url+"&marc=true"
		title_info_resp = requests.get(title_info_url)
		print(f"\n>> Loading title information page...\n")
		print(f">> Page: {title_info_url} exists? {title_info_resp.status_code}") # 200: ok, 400: bad_request, 403: forbidden, 404: not_found
		print(title_info_resp.text)

	"""
	if txt_resp.status_code==200:
		print(f">> Loading ...")
		print(txt_resp.text)

		#page = urllib.request.urlopen( s_url )
		#html_orig = page.read().decode("utf-8")
		#html_prty = BeautifulSoup(html_orig, "html.parser")
		#print(html_orig)
		#print(html_prty.get_text())
	else:
		print(f">> Loading xlsx file...")
	"""

def get_ocr_texts(df):
	print(f">> Extracting OCR text from df: {df.shape}")
	
	df['referer'] = df['referer'].replace("-", pd.NA, regex=True)
	#df['referer'] = df['referer'].replace("-", np.nan, regex=True)
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
		#print(f"\n>> input url: {in_url}")
		
		if in_url is pd.NA:
			#print(f">> no link is available! => exit")
			return 0
		
		#print(f"\n>> Parsing url ...")
		parsed_url = urllib.parse.urlparse(in_url)
		#print(parsed_url)

		#print(f">> Explore url parameters ...")
		parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
		#print(parameters)
		
		#print(f">> Search Page?")
		SRCH_PARAM = True if "search" in parsed_url.path else False
		#print(SRCH_PARAM)

		#print(f">> Page Parameter?")
		PG_PARAM = True if "page" in parsed_url.query else False
		#print(PG_PARAM)

		if PG_PARAM and not SRCH_PARAM:
			#print(f">> Generating txt page ...")
			txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
			txt_resp = requests.get(txt_pg_url)

			#print(f">> Page: {txt_pg_url} exists? {txt_resp.status_code}") # 200: ok, 400: bad_request, 403: forbidden, 404: not_found

			if txt_resp.status_code==200:
				#print(f"\n>> Loading txt page...\n")
				#print(txt_resp.text)
				return txt_resp.text

	check_urls = lambda x:analyze_(x)
	
	# cleaning
	df["ocr_text"] = pd.DataFrame( df.referer.apply( check_urls ) )
	print(list(df.columns))
	print("#"*150)
	print(df.head(50))
	print("#"*150)
	print(df.info(verbose=True, memory_usage="deep"))
	print("#"*150)
	print(f"\n\n>> NaN referer: {df['ocr_text'].isna().sum()} / {df.shape[0]}\n\n")
	

def run():
	
	# working with single log file:
	fname = "nike5.docworks.lib.helsinki.fi_access_log.2017-02-01.log"	
	#fname = "nike6.docworks.lib.helsinki.fi_access_log.2017-02-02.log"

	df = get_df_no_ip_logs(infile=os.path.join(dpath, fname))

	#get_single_ocr_text(df, browser_show=False)
	get_ocr_texts(df)
	
	"""
	log_files = natsorted( glob.glob( os.path.join(dpath, "*.log") ) )
	log_files_date = [lf[ lf.rfind(".2")+1: lf.rfind(".") ] for lf in log_files]
	print(len(log_files_date), log_files_date)

	for f in log_files:
		df = get_df_no_ip_logs( infile=f )
		print(df.shape)
		#print("-"*130)

		#print( df.head(40) )
		#print("-"*130)

		#print( df.tail(40) )
		print(f"\t\tCOMPLETED!")
	"""

if __name__ == '__main__':
	os.system('clear')
	run()