import os
import re
import glob
import webbrowser
import string
import time
import argparse

from natsort import natsorted
import numpy as np
import pandas as pd

from url_scraping import *
from utils import *

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
parser.add_argument('--query', default=0, type=int) # smallest
args = parser.parse_args()


def single_query(file_="", ts=None, browser_show=False):
	print(f">> Analyzing a single query of {file_}")
	st_t = time.time()
	#df = get_df_no_ip_logs(infile=file_, TIMESTAMP=ts)
	df = get_df_pseudonymized_logs(infile=file_, TIMESTAMP=ts)

	elapsed_t = time.time() - st_t
	print(f">> Elapsed time: {elapsed_t:.2f} sec\tINITIAL df: {df.shape}\tavg search/s: {df.shape[0]/(24*60*60):.3f}")
	print("-"*100)
	
	#print(df.head(30))
	#print("-"*150)
	#print(df.tail(30))

	print(f">> Generating a sample query...")
	#qlink = int(np.random.randint(0, high=df.shape[0]+1, size=1))
	qlink = np.random.choice(df.shape[0]+1)

	#qlink = 43786 # with quey word!
	########### file Q=6 ########### 
	#qlink = 1368231 # does not query words but results are given!
	#qlink = 91218 # does not query words
	#qlink = 54420 # going directly to OCR
	#qlink = 72120 # parsing elapsed time: 11.73 sec varies stochastically!

	########### file Q=0 ########### 
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
	s = time.time()
	
	single_url = df["referer"][qlink]
	print(f">> Q: {qlink} : {single_url}")

	r = checking_(single_url)
	if r is None:
		return

	single_url = r.url

	if browser_show:
		#print(f"\topening in browser... ")
		webbrowser.open(single_url, new=2)
	
	print(f"\n>> Parsing cleanedup URL: {single_url}")

	parsed_url, parameters = get_parsed_url_parameters(single_url)
	print(f"\n>> Parsed url:")
	print(parsed_url)

	print(f"\n>> Explore parsed url parameters:")
	print(parameters)

	# check for queries -> REST API:
	if parameters.get("query"):
		my_query_word = ",".join( parameters.get("query") )
		df.loc[qlink, "query_word"] = my_query_word
		print("#"*65)
		print(f"\tEXECUTE BASH REST API for {my_query_word}")
		print("#"*65)

	# term(ONLY IF OCR page):
	if parameters.get("term") and parameters.get("page"):
		df.loc[qlink, "ocr_term"] = ",".join(parameters.get("term"))
		txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
		print(f">> page-X.txt available?\t{txt_pg_url}\t")

		text_response = checking_(txt_pg_url)
		if text_response is not None:
			print(f"\t\t\tYES >> loading...\n")
			df.loc[qlink, "ocr_text"] = text_response.text
	
	print(f"\n\n>> Parsing Completed!\tElapsed time: {time.time()-s:.2f} sec\tFINAL df: {df.shape}")
	print(list(df.columns))
	print("-"*150)

	print(df.isna().sum())
	print(df.info(verbose=True, memory_usage="deep"))

	#print(df.head(30))
	#print("-"*150)
	#print(df.tail(30))
	#save_(df, infile=f"SINGLEQuery_timestamp_{ts}_{file_}")

def all_queries(file_="", ts=None):
	#print(f">> Analyzing queries of {file_} ...")
	st_t = time.time()
	#df = get_df_no_ip_logs(infile=file_, TIMESTAMP=ts)
	df = get_df_pseudonymized_logs(infile=file_, TIMESTAMP=ts)

	elapsed_t = time.time() - st_t
	print(f">> Elapsed time: {elapsed_t:.2f} sec\tINITIAL df: {df.shape}\tavg search/s: {df.shape[0]/(24*60*60):.3f}")
	print("-"*100)
	#save_(df, infile=file_)
	#return

	def analyze_(df):
		raw_url = df.referer
		#print(f"\n>> RAW URL: {raw_url}")

		r = checking_(raw_url)
		if r is None:
			return df

		in_url = r.url
		#print(f"\tUpdated URL: {in_url}")

		parsed_url, parameters = get_parsed_url_parameters(in_url)
		#print(f"Parsed: {parsed_url}")
		#print(f"Parameters: {parameters}")

		if parameters.get("fuzzy"): df["fuzzy"] = ",".join(parameters.get("fuzzy"))
		if parameters.get("qMeta"): df["has_metadata"] = ",".join(parameters.get("qMeta"))
		if parameters.get("hasIllustrations"): df["has_illustration"] = ",".join(parameters.get("hasIllustrations"))
		if parameters.get("showUnauthorizedResults"): df["show_unauthorized_results"] = ",".join(parameters.get("showUnauthorizedResults"))
		if parameters.get("pages"): df["pages"] = ",".join(parameters.get("pages"))
		if parameters.get("importTime"): df["import_time"] = ",".join(parameters.get("importTime"))
		if parameters.get("collection"): df["collection"] = ",".join(parameters.get("collection"))
		if parameters.get("author"): df["author"] = ",".join(parameters.get("author"))
		if parameters.get("tag"): df["keyword"] = ",".join(parameters.get("tag"))
		if parameters.get("publicationPlace"): df["publication_place"] = ",".join(parameters.get("publicationPlace"))
		if parameters.get("lang"): df["language"] = ",".join(parameters.get("lang"))
		if parameters.get("formats"): df["document_type"] = ",".join(parameters.get("formats"))
		if parameters.get("showLastPage"): df["show_last_page"] = ",".join(parameters.get("showLastPage"))
		if parameters.get("orderBy"): df["order_by"] = ",".join(parameters.get("orderBy"))
		if parameters.get("publisher"): df["publisher"] = ",".join(parameters.get("publisher"))
		if parameters.get("startDate"): df["start_date"] = ",".join(parameters.get("startDate"))
		if parameters.get("endDate"): df["end_date"] = ",".join(parameters.get("endDate"))
		if parameters.get("requireAllKeywords"): df["require_all_keywords"] = ",".join(parameters.get("requireAllKeywords"))
		if parameters.get("resultType"): df["result_type"] = ",".join(parameters.get("resultType"))
		
		# web scraping for 20 search results: 
		if parameters.get("query"):
			my_query_word = ",".join(parameters.get("query"))
			df["query_word"] = my_query_word
			#print("#"*65)
			#print(f"\tEXECUTE BASH REST API for {my_query_word}")
			#print("#"*65)
			#df["search_results"] = get_all_search_details(in_url)
		
		# OCR extraction:
		if parameters.get("term") and parameters.get("page"):
			df["ocr_term"] = ",".join(parameters.get("term"))
			df["ocr_page"] = ",".join(parameters.get("page"))
			txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
			#print(f">> page-X.txt available?\t{txt_pg_url}\t")		
			text_response = checking_(txt_pg_url)
			if text_response is not None: # 200
				#print(f"\t\t\tYES >> loading...\n")
				#return text_response.text
				df["ocr_text"] = text_response.text
		
		return df
	
	s = time.time()
	check_urls = lambda INPUT_DF: analyze_(INPUT_DF)
	df = pd.DataFrame( df.apply( check_urls, axis=1, ) )	
	print(f">> Parsing Completed!\tElapsed time: {time.time()-s:.2f} sec\tFINAL df: {df.shape}")
	print("*"*205)
	print(df.head(30))
	
	print("#"*150)
	print(df.info(verbose=True, memory_usage="deep"))
	print("#"*150)
	
	cols = list(df.columns)
	print(len(cols), cols)
	print("#"*150)

	print(df.head(30))
	print("-"*150)
	print(df.tail(30))

	save_(df, infile=file_, saving_path=dfs_path)

def get_query_log(QUERY=0):
	#print(f"\nGiven log file index: {QUERY}")
	log_files_dir = natsorted( glob.glob( os.path.join(dpath, "*.log") ) )
	#print(log_files_dir)

	#log_files_date = [lf[ lf.rfind(".2")+1: lf.rfind(".") ] for lf in log_files_dir]
	#print(len(log_files_date), log_files_date)
	
	all_log_files = [lf[ lf.rfind("/")+1: ] for lf in log_files_dir]
	#print(all_log_files)
	query_log_file = all_log_files[QUERY] 
	print(f"\nQ: {QUERY}\t{query_log_file}")

	return query_log_file

def run():
	make_folder(folder_name=dfs_path)
	"""
	single_query(file_=get_query_log(QUERY=args.query), 
							#browser_show=True, 
							#ts=["23:52:00", "23:59:59"],
							)
	"""
	# run all log files using array in batch
	
	all_queries(file_=get_query_log(QUERY=args.query),
							#ts=["00:00:00", "00:02:59"],
							)
	#print(f"\t\tCOMPLETED!")

def run_all():
	for q in range(70):
		all_queries(file_=get_query_log(QUERY=q),)


if __name__ == '__main__':
	os.system('clear')
	run()
	#run_all()