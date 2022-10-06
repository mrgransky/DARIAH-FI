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

def update_url(INP_URL):
	session = requests.Session()  # so connections are recycled
	resp = session.head(INP_URL, allow_redirects=True)
	#print(resp.url)
	updated_url = resp.url
	return updated_url

def broken_connection(url):
	try:
		requests.get(url)
		return False
	except requests.exceptions.ConnectionError:
		#print ("Failed to open url")
		return True

def get_df_no_ip_logs(infile=""):
	print(f">> Reading {infile} ...")
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (?P<status>\d{3}) (.*) "([^"]*)" "(.*?)" (.*)' # original working!
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^"]+)" "(.*?)" (.*)' # original working!
	ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^\"]*)" "(.*?)" (.*)' # checked with all log files!
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "(?:-|.*(http://\D.*))" "(.*?)" (.*)'
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "(?:|-|.*(://\D.*))" "(.*?)" (.*)'
	
	cleaned_lines = []
	with open(infile, mode="r") as f:
		for line in f:
			#print(line)
			matched_line = re.match(ACCESS_LOG_PATTERN, line)
			#print (matched_line)
			l = matched_line.groups()
			#print(l)

			dt_tz = l[0].replace("[", "").replace("]", "")
			
			DDMMYYYY = dt_tz[:dt_tz.find(":")]
			YYYYMMDD = convert_date( DDMMYYYY )
			HMS = dt_tz[dt_tz.find(":")+1:dt_tz.find(" ")]
			TZ = dt_tz[dt_tz.find(" ")+1:]

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
	df['referer'] = df['referer'].replace("-", pd.NA, regex=True).replace(r'^\s*$', pd.NA, regex=True)
	df['user_agent'] = df['user_agent'].replace("-", pd.NA, regex=True).replace(r'^\s*$', pd.NA, regex=True)
	
	# with numpy:
	#df['referer'] = df['referer'].replace("-", np.nan, regex=True).replace(r'^\s*$', np.nan, regex=True)
	#df['user_agent'] = df['user_agent'].replace("-", np.nan, regex=True).replace(r'^\s*$', np.nan, regex=True)
	
	return df

def get_single_ocr_text(df, browser_show=True):
	print(f">> Analyze single df: {df.shape}")
	
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
	#qlink = 

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
	s_url = update_url(s_url)
	print(f"\t>> {s_url}")

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
			#print(txt_resp.text)
				

		#title_info_url = s_url+"&marc=true"
		#title_info_resp = requests.get(title_info_url)
		#print(f"\n>> Loading title information page...\n")
		#print(f">> Page: {title_info_url} exists? {title_info_resp.status_code}") # 200: ok, 400: bad_request, 403: forbidden, 404: not_found
		#print(title_info_resp.text)

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

		print(f">> updating url ...")
		in_url = update_url(in_url)
		print(f"\t>> {in_url}")

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
	#fname = "nike5.docworks.lib.helsinki.fi_access_log.2017-02-01.log"	
	#fname = "nike6.docworks.lib.helsinki.fi_access_log.2017-02-02.log"

	#fname = "nike5.docworks.lib.helsinki.fi_access_log.2017-02-07.log"	# smallest 
	#df = get_df_no_ip_logs(infile=os.path.join(dpath, fname))

	#get_single_ocr_text(df, browser_show=False)
	#get_ocr_texts(df)
	
	#"""
	log_files = natsorted( glob.glob( os.path.join(dpath, "*.log") ) )
	log_files_date = [lf[ lf.rfind(".2")+1: lf.rfind(".") ] for lf in log_files]
	#print(len(log_files_date), log_files_date)

	for f in log_files:
		df = get_df_no_ip_logs( infile=f )
		print(df.shape)
		#print("-"*130)

		#print( df.head(40) )
		#print("-"*130)

		#print( df.tail(40) )
		print(f"\t\tCOMPLETED!")
	#"""

if __name__ == '__main__':
	os.system('clear')
	run()