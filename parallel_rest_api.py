import numpy as np
import urllib
import time
import re
import aiohttp
import asyncio
from typing import List, Set, Dict, Tuple

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

SEARCH_QUERY_DIGI_URL: str = "https://digi.kansalliskirjasto.fi/search?requireAllKeywords=true&query="
DIGI_HOME_PAGE_URL : str = "https://digi.kansalliskirjasto.fi"
headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}
payload = {
	"authors": [],
	"collections": [],
	"districts": [],
	"endDate": None,
	"exactCollectionMaterialType": "false",
	"formats": [],
	"fuzzy": "false",
	"hasIllustrations": "false",
	"importStartDate": None,
	"importTime": "ANY",
	"includeUnauthorizedResults": "false",
	"languages": [],
	"orderBy": "RELEVANCE",
	"pages": "",
	"publicationPlaces": [],
	"publications": [],
	"publishers": [],
	"queryTargetsMetadata": "false",
	"queryTargetsOcrText": "true",
	"searchForBindings": "false",
	"showLastPage": "false",
	"startDate": None,
	"tags": [],
}
MY_QUERY_PHRASE : str = "referensbibliotek"
my_list = ['finland', 'öppen', 'komma', 'svensk', 'fartyg', 'regering', 'meddela', 'ryssland', 'övrig', 'london', 'hjältegrav', 'runeberg', 'engelsk', 'vicehär', 'hålla', 'finnas', 'tyskland', 'blott', 'suppleant', 'skriva', 'plats', 'fråga', 'kvinna', 'ärkebiskop', 'färgblindhet', 'stockholm', 'jordfästning', 'hjält', 'finländsk', 'england', 'helsingfors', 'koffert', 'hjälp', 'person', 'besättning', 'imperialism', 'arbetare', 'ledare', 'sänka', 'förening', 'bombardera', 'flykting', 'intresse', 'köpenhamn', 'korrespondent', 'redan', 'procession', 'gammal', 'kyrka', 'folkhjälp', 'medlem', 'släkting', 'värld', 'sphinx', 'styrelse', 'representant', 'sverige', 'neutral', 'understöd', 'vapenbroder', 'vänna', 'soldat', 'förhållande', 'lämna', 'besluta', 'yttervärld', 'sända', 'arbete', 'befolkning', 'maskin', 'följande', 'sydvästkust', 'människa', 'eriksgata', 'förvaltningsråd', 'växla', 'endast', 'insamling', 'belgien', 'vacker', 'ångare', 'strid', 'nämligen', 'utrikesminister', 'spansk', 'kreditanstalt', 'besiktning', 'allmän', 'militär', 'flygare', 'naturligtvis', 'utlåning', 'ingenjör', 'främst', 'börja', 'norrie', 'teatern', 'hembygd', 'amerik', 'hemtrakt']

def filter_zero_elements(numbers, words):
	"""
	Filters elements with zero values from two lists and returns the filtered lists.

	Args:
			numbers: A list of numbers.
			words: A list of words.

	Returns:
			A tuple containing two lists:
					- numbers_list: A list of numbers with zeros removed.
					- words_list: A list of words with corresponding indices to the filtered numbers.
	"""

	numbers_list = [num for num, word in zip(numbers, words) if num != 0]
	words_list = [word for num, word in zip(numbers, words) if num != 0]
	return numbers_list, words_list

def get_num_NLF_pages(INPUT_QUERY: str="query", INPUT_TOKEN: str="token"):
	st_t = time.time()
	URL = f"{SEARCH_QUERY_DIGI_URL}" + urllib.parse.quote_plus(INPUT_QUERY + " " + INPUT_TOKEN)
	print(f"{URL:<150}", end=" ")
	parsed_url = urllib.parse.urlparse(URL)
	parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
	offset_pg = ( int( re.search(r'page=(\d+)', URL).group(1) )-1)*20 if re.search(r'page=(\d+)', URL) else 0
	search_page_request_url = f"{DIGI_HOME_PAGE_URL}/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	payload["query"] = parameters.get('query')[0] if parameters.get('query') else ""
	payload["requireAllKeywords"] = parameters.get('requireAllKeywords')[0] if parameters.get('requireAllKeywords') else "false"
	try:
		r = session.post(
			url=search_page_request_url,
			json=payload,
			headers=headers,
		)
		r.raise_for_status()  # Raise HTTPError for bad status codes
		res = r.json()
		TOTAL_NUM_NLF_RESULTs = res.get("totalResults")
		print(f"Found NLF tot_page(s): {TOTAL_NUM_NLF_RESULTs:<10} in {time.time()-st_t:.1f} sec")
	except requests.exceptions.RequestException as e:
		print(f"<!> Error: {e}")
		return
	return TOTAL_NUM_NLF_RESULTs

async def get_recommendation_num_NLF_pages_async(session, INPUT_QUERY: str="global warming", REC_TK: str="pollution"):
	st_t = time.time()
	URL = f"{SEARCH_QUERY_DIGI_URL}" + urllib.parse.quote_plus(INPUT_QUERY + " " + REC_TK)
	# print(f"{URL:<150}", end=" ")
	parsed_url = urllib.parse.urlparse(URL)
	parameters = urllib.parse.parse_qs(parsed_url.query, keep_blank_values=True)
	offset_pg = (int(re.search(r'page=(\d+)', URL).group(1)) - 1) * 20 if re.search(r'page=(\d+)', URL) else 0
	search_page_request_url = f"{DIGI_HOME_PAGE_URL}/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	payload["query"] = parameters.get('query')[0] if parameters.get('query') else ""
	payload["requireAllKeywords"] = parameters.get('requireAllKeywords')[0] if parameters.get('requireAllKeywords') else "false"
	try:
		async with session.post(
			url=search_page_request_url,
			json=payload,
			headers=headers,
		) as response:
				response.raise_for_status()
				res = await response.json()
				TOTAL_NUM_NLF_RESULTs = res.get("totalResults")
				# print(f"Found NLF tot_page(s): {TOTAL_NUM_NLF_RESULTs:<10} in {time.time() - st_t:.1f} sec")
				return TOTAL_NUM_NLF_RESULTs
	except aiohttp.ClientError as e:
		print(f"<!> Error: {e}")
		return None

async def get_num_NLF_pages_asynchronous_run(qu: str="global warming", TOKENs_list: List[str]=["tk1", "tk2"]):
	async with aiohttp.ClientSession() as session:
		tasks = [
			NUMBER_OF_PAGES
			for tk in TOKENs_list
			if (
				(NUMBER_OF_PAGES:=get_recommendation_num_NLF_pages_async(session, INPUT_QUERY=qu, REC_TK=tk))
			)
		]
		num_NLF_pages = await asyncio.gather(*tasks)
		# # Filter out None and 0 values
		# num_NLF_pages = [pages for pages in num_NLF_pages if pages not in [None, 0]]
		return num_NLF_pages


# OLD implementation: inefficient
start_time = time.time()
TOKENs_num_NLF_pages = [
	# NUMBER_OF_PAGEs
	get_num_NLF_pages(INPUT_QUERY=MY_QUERY_PHRASE, INPUT_TOKEN=tk)
	for tk in my_list
	# if(
	# 	(NUMBER_OF_PAGEs:=get_num_NLF_pages(INPUT_QUERY=MY_QUERY_PHRASE, INPUT_TOKEN=tk))
	# )
]
print(f"TOTAL Elapsed_t: {time.time()-start_time:.1f} | Initial TK list: {len(my_list)} | {len(TOKENs_num_NLF_pages)} PAGE(s)")
print(TOKENs_num_NLF_pages)
print("#"*100)

# asynchronous implementation: efficient
start_time_async = time.time()
TOKENs_num_NLF_pages_async = asyncio.run(get_num_NLF_pages_asynchronous_run(qu=MY_QUERY_PHRASE, TOKENs_list=my_list))
print(f"TOTAL Elapsed_t: {time.time() - start_time_async:.1f} | Initial TK list: {len(my_list)} | {len(TOKENs_num_NLF_pages_async)} PAGE(s)")
print(TOKENs_num_NLF_pages_async)

print("#"*100)
print(my_list)
print(np.all(TOKENs_num_NLF_pages==TOKENs_num_NLF_pages_async))

# # Example usage
# numbers = [0, 10, 11, 22, 0, 5, 8, 9, 1, 0]
# words = ['finland', 'öppen', 'komma', 'svensk', 'fartyg', 'regering', 'meddela', 'ryssland', 'övrig', 'london']

# numbers_list, words_list = filter_zero_elements(numbers, words)

# print(f"Numbers without zeros: {numbers_list}")
# print(f"Corresponding words: {words_list}")