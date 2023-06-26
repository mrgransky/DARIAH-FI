from utils import *

def scrap_clipping_page(URL):
	#print(f"Scraping clipping page: {URL}")
	parsed_url, parameters = get_parsed_url_parameters(URL)
	#print(f"Parameters:\n{json.dumps(parameters, indent=2, ensure_ascii=False)}")
	#print()
	#print(f"parsed_url : {parsed_url}")

	offset_pg=(int(parameters.get('page')[0])-1)*20 if "page=" in URL else 0
	clipping_pg_api = f"https://digi.kansalliskirjasto.fi/rest/article-search/search-by-type?offset={offset_pg}&count=20"
	payload = {	"categoryIds": parameters.get('categoryId') if parameters.get('categoryId') else [],
							"collections": parameters.get('collection') if parameters.get('collection') else [],
							"endDate": parameters.get('endDate')[0] if parameters.get('endDate') else None,
							"exactCollectionMaterialType": "false", # TODO: must be investigated!!
							"fuzzy": parameters.get('fuzzy')[0] if parameters.get('fuzzy') else "false",
							"generalTypes": parameters.get('formats') if parameters.get('formats') else [],
							"includeCollected": parameters.get('includeCollected')[0] if parameters.get('includeCollected') else "false",
							"keywords": parameters.get('keyword') if parameters.get('keyword') else [],
							"onlyCollected": "false", # TODO: must be investigated!!
							"orderBy": parameters.get('orderBy')[0] if parameters.get('orderBy') else "CREATED_DESC",
							"query": parameters.get('query')[0] if parameters.get('query') else "",
							"queryTargetsMetadata": parameters.get('qMeta')[0] if parameters.get('qMeta') else "false",
							"queryTargetsOcrText": parameters.get('qOcr')[0] if parameters.get('qOcr') else "true",
							"requireAllKeywords": parameters.get('requireAllKeywords')[0]  if parameters.get('requireAllKeywords') else "false",
							"startDate": parameters.get('startDate')[0] if parameters.get('startDate') else None,
							"subjectIds": parameters.get('subjectId') if parameters.get('subjectId') else [],
							"titles": parameters.get('title') if parameters.get('title') else [],
							}
	headers = {	'Content-type': 'application/json',
							'Accept': 'application/json; text/plain; */*', 
							'Cache-Control': 'no-cache',
							'Connection': 'keep-alive',
							'Pragma': 'no-cache',
							}

	try:
		st_t = time.time()
		r = requests.post(url=clipping_pg_api, 
											json=payload, 
											headers=headers,
											)
		res = r.json()
		#print(res.keys())
		CLIPPING_RESULTS = res.get("rows")
		#print(f"\t\tFound {len(CLIPPING_RESULTS)} clipping result(s) | Elapsed_t: {time.time()-st_t:.2f} s")
		#print(json.dumps(CLIPPING_RESULTS, indent=2, ensure_ascii=False))
	except (requests.exceptions.Timeout,
					requests.exceptions.ConnectionError, 
					requests.exceptions.RequestException, 
					requests.exceptions.TooManyRedirects,
					requests.exceptions.InvalidSchema,
					ValueError, 
					TypeError, 
					EOFError, 
					RuntimeError,
					json.JSONDecodeError,
					json.decoder.JSONDecodeError,
					Exception, 
				) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}")
		return
	return CLIPPING_RESULTS

def scrap_collection_page(URL):
	#print(f"Scraping collection page: {URL}")
	st_t = time.time()
	parsed_url, parameters = get_parsed_url_parameters(URL)
	#print(f"Parsed url:\n{json.dumps(parameters, indent=2, ensure_ascii=False)}")
	#print()
	#print(f"parsed_url : {parsed_url}")

	offset_pg=(int(parameters.get('page')[0])-1)*20 if "page=" in URL else 0
	collection_pg_api = f"https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset={offset_pg}&count=200"
	#print(collection_pg_api)
	payload = {	"authors": parameters.get('author') if parameters.get('author') else [],
							"collections": parameters.get('collection') if parameters.get('collection') else [],
							"districts": [], # TODO: must be investigated!!
							"endDate": parameters.get('endDate')[0] if parameters.get('endDate') else None,
							"exactCollectionMaterialType": "false", # TODO: must be investigated!!
							"formats": parameters.get('formats') if parameters.get('formats') else [],
							"fuzzy": parameters.get('fuzzy')[0] if parameters.get('fuzzy') else "false",
							"hasIllustrations": parameters.get('hasIllustrations')[0] if parameters.get('hasIllustrations') else "false",
							"importStartDate": parameters.get('importStartDate')[0] if parameters.get('importStartDate') else None,
							"importTime": parameters.get('importTime')[0] if parameters.get('importStartDate') else "ANY",
							"includeUnauthorizedResults": parameters.get('showUnauthorizedResults')[0] if parameters.get('showUnauthorizedResults') else "false",
							"languages": parameters.get('lang') if parameters.get('lang') else [],
							"orderBy": parameters.get('orderBy')[0] if parameters.get('orderBy') else "DATE_DESC",
							"pages": parameters.get('pages')[0]  if parameters.get('pages') else "",
							"publicationPlaces": parameters.get('publicationPlace') if parameters.get('publicationPlace') else [],
							"publications": parameters.get('title') if parameters.get('title') else [],
							"publishers": parameters.get('publisher') if parameters.get('publisher') else [],
							"query": parameters.get('query')[0] if parameters.get('query') else "",
							"queryTargetsMetadata": parameters.get('qMeta')[0] if parameters.get('qMeta') else "false",
							"queryTargetsOcrText": parameters.get('qOcr')[0] if parameters.get('qOcr') else "true",
							"requireAllKeywords": parameters.get('requireAllKeywords')[0]  if parameters.get('requireAllKeywords') else "false",
							"searchForBindings": parameters.get('searchForBindings')[0]  if parameters.get('searchForBindings') else "false",
							"showLastPage": parameters.get('showLastPage')[0]  if parameters.get('showLastPage') else "false",
							"startDate": parameters.get('startDate')[0] if parameters.get('startDate') else None,
							"tags": parameters.get('tag') if parameters.get('tag') else [],
							}
	headers = {	'Content-type': 'application/json',
							'Accept': 'application/json; text/plain; */*', 
							'Cache-Control': 'no-cache',
							'Connection': 'keep-alive',
							'Pragma': 'no-cache',
							}

	try:
		r = requests.post(url=collection_pg_api, 
											json=payload, 
											headers=headers,
											)
		res = r.json()
		#print(res.keys())
		COLLECTION_RESULTS = res.get("rows")
		#print(f"\t\tFound {len(COLLECTION_RESULTS)} search result(s) | Elapsed_t: {time.time()-st_t:.2f} s")
		#print(json.dumps(COLLECTION_RESULTS, indent=2, ensure_ascii=False))
	except (requests.exceptions.Timeout,
					requests.exceptions.ConnectionError, 
					requests.exceptions.RequestException, 
					requests.exceptions.TooManyRedirects,
					requests.exceptions.InvalidSchema,
					ValueError, 
					TypeError, 
					EOFError, 
					RuntimeError,
					json.JSONDecodeError,
					json.decoder.JSONDecodeError,
					Exception, 
					) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}")
		return
	return COLLECTION_RESULTS

def scrap_search_page(URL):
	# print(f"<> Scraping: {URL}")
	st_t = time.time()
	parsed_url, parameters = get_parsed_url_parameters(URL)
	#print(f"Parsed url:\n{json.dumps(parameters, indent=2, ensure_ascii=False)}")

	# print(f"{parsed_url}")
	offset_pg=( int( re.search(r'page=(\d+)', URL).group(1) )-1)*20 if re.search(r'page=(\d+)', URL) else 0
	search_pg_api = f"https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	
	payload = {	"authors": parameters.get('author') if parameters.get('author') else [],
							"collections": parameters.get('collection') if parameters.get('collection') else [],
							"districts": [], # TODO: must be investigated!!
							"endDate": parameters.get('endDate')[0] if parameters.get('endDate') else None,
							"exactCollectionMaterialType": "false", # TODO: must be investigated!!
							"formats": parameters.get('formats') if parameters.get('formats') else [],
							"fuzzy": parameters.get('fuzzy')[0] if parameters.get('fuzzy') else "false",
							"hasIllustrations": parameters.get('hasIllustrations')[0] if parameters.get('hasIllustrations') else "false",
							"importStartDate": parameters.get('importStartDate')[0] if parameters.get('importStartDate') else None,
							"importTime": parameters.get('importTime')[0] if parameters.get('importStartDate') else "ANY",
							"includeUnauthorizedResults": parameters.get('showUnauthorizedResults')[0] if parameters.get('showUnauthorizedResults') else "false",
							"languages": parameters.get('lang') if parameters.get('lang') else [],
							"orderBy": parameters.get('orderBy')[0] if parameters.get('orderBy') else "IMPORT_DATE",
							"pages": parameters.get('pages')[0]  if parameters.get('pages') else "",
							"publicationPlaces": parameters.get('publicationPlace') if parameters.get('publicationPlace') else [],
							"publications": parameters.get('title') if parameters.get('title') else [],
							"publishers": parameters.get('publisher') if parameters.get('publisher') else [],
							"query": parameters.get('query')[0] if parameters.get('query') else "",
							"queryTargetsMetadata": parameters.get('qMeta')[0] if parameters.get('qMeta') else "false",
							"queryTargetsOcrText": parameters.get('qOcr')[0] if parameters.get('qOcr') else "true",
							"requireAllKeywords": parameters.get('requireAllKeywords')[0]  if parameters.get('requireAllKeywords') else "false",
							"searchForBindings": parameters.get('searchForBindings')[0]  if parameters.get('searchForBindings') else "false",
							"showLastPage": parameters.get('showLastPage')[0]  if parameters.get('showLastPage') else "false",
							"startDate": parameters.get('startDate')[0] if parameters.get('startDate') else None,
							"tags": parameters.get('tag') if parameters.get('tag') else [],
							}
	
	headers = {	'Content-type': 'application/json',
							'Accept': 'application/json; text/plain; */*', 
							'Cache-Control': 'no-cache',
							'Connection': 'keep-alive',
							'Pragma': 'no-cache',
							}

	try:
		r = requests.post(url=search_pg_api, 
											json=payload, 
											headers=headers,
											)

		#print(r.headers)
		#print(r.status_code)

		res = r.json()
		# a list of up to 20 results, each of which contains: 
		#print(res.keys()): ['bindingId', 'bindingTitle', 'publicationId', 'generalType', 'authorized', 'authors', 'pageNumber', 'language', 'publisher', 'issue', 'importDate', 'dateAccuracy', 'placeOfPublication', 'textHighlights', 'terms', 'score', 'url', 'thumbnailUrl', 'date']
		SEARCH_RESULTS = res.get("rows") 
		#print(f"\t\tFound {len(SEARCH_RESULTS)} search result(s) | Elapsed_t: {time.time()-st_t:.2f} s")
		#print(json.dumps(SEARCH_RESULTS, indent=2, ensure_ascii=False))
	except (requests.exceptions.Timeout,
					requests.exceptions.ConnectionError, 
					requests.exceptions.RequestException, 
					requests.exceptions.TooManyRedirects,
					requests.exceptions.InvalidSchema,
					ValueError, 
					TypeError, 
					EOFError, 
					RuntimeError,
					json.JSONDecodeError,
					json.decoder.JSONDecodeError,
					Exception, 
					) as e:
		print(f"{type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}")
		return
	return SEARCH_RESULTS

def scrap_ocr_page_content(URL):
	print(f"Scraping newspaper content page: {URL}")

	if "&page=" in URL:
		up_url = URL
	else:
		up_url = f"{URL}&page=1"

	#print(f"\tUpdated: {up_url}")
	parsed_url, parameters = get_parsed_url_parameters(up_url)
	#print(f"Parsed url | OCR extraction: {parameters}")
	#print(f"parsed_url : {parsed_url}")
	
	api_url = f"https://digi.kansalliskirjasto.fi/rest/binding-search/ocr-hits/{parsed_url.path.split('/')[-1]}"
	try:
		hgltd_wrds = [d.get("text") for d in requests.get(api_url, params=parameters).json()]
	except json.JSONDecodeError as jve:
		print(f"JSON empty response:\n{jve}")
		hgltd_wrds = []
	api_nwp = f"https://digi.kansalliskirjasto.fi/rest/binding?id={parsed_url.path.split('/')[-1]}"
	nwp_info = requests.get(api_nwp).json()
	
	#print(list(nwp_info.get("bindingInformation").keys()))
	#print(list(nwp_info.get("bindingInformation").get("citationInfo").keys()))
	#print(nwp_info.get("bindingInformation").get("citationInfo").get("refWorksLanguage"))
	#print(nwp_info.get("bindingInformation").get("citationInfo").get("refWorksOutputLanguage")) # English (30)
	#print()
	
	title = nwp_info.get("bindingInformation").get("publicationTitle") # Uusi Suometar 
	doc_type = nwp_info.get("bindingInformation").get("generalType") # NEWSPAER
	issue = nwp_info.get("bindingInformation").get("issue") # 63
	publisher = nwp_info.get("bindingInformation").get("citationInfo").get("publisher") # Uuden Suomettaren Oy
	pub_date = nwp_info.get("bindingInformation").get("citationInfo").get("localizedPublishingDate") # 16.03.1905
	pub_place = nwp_info.get("bindingInformation").get("citationInfo").get("publishingPlace") # Helsinki, Suomi
	lang = nwp_info.get("bindingInformation").get("citationInfo").get("refWorksLanguage") # English

	txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
	# ocr_api_url = f"https://digi.kansalliskirjasto.fi/rest/binding/ocr-data?bindingId={parsed_url.path.split('/')[-1]}&page={parameters.get('page')[0]}&oldOcr=false"
	# print(f">> ocr_api_url: {ocr_api_url}")
	text_response = checking_(txt_pg_url)
	if text_response: # 200
		txt = text_response.text
	else:
		txt = None

	return title, doc_type, issue, publisher, pub_date, pub_place, lang, parameters.get("term"), hgltd_wrds, parameters.get("page"), txt

def scrap_newspaper_content_page(URL):
	print(f"URL: {URL:<150}", end="")
	NWP_CONTENT_RESULTS = {}
	up_url = URL if re.search(r'page=(\d+)', URL) else f"{URL}&page=1"
	# print(f">> Updated: {up_url}")
	st_t = time.time()
	parsed_url, parameters = get_parsed_url_parameters(up_url)
	if (checking_(up_url) is None or not parameters):
		return
	NWP_CONTENT_RESULTS["parsed_term"] = parameters.get("term")
	NWP_CONTENT_RESULTS["page"] = parameters.get("page")
	# print(f"parsed_url : {parsed_url}")
	# print(json.dumps(parameters, indent=2, ensure_ascii=False))
	txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
	# print(f"<> page-X.txt: {txt_pg_url}")
	rsp_txt = checking_(txt_pg_url)
	if rsp_txt:
		try:
			NWP_CONTENT_RESULTS["text"] = rsp_txt.text
		except(requests.exceptions.Timeout,
						requests.exceptions.ConnectionError, 
						requests.exceptions.RequestException, 
						requests.exceptions.TooManyRedirects,
						requests.exceptions.InvalidSchema,
						json.decoder.JSONDecodeError,
						json.JSONDecodeError,
						ValueError, 
						TypeError, 
						EOFError, 
						RuntimeError,
						Exception, 
				) as e:
			print(f"<!> {e}")

	api_url = f"https://digi.kansalliskirjasto.fi/rest/binding-search/ocr-hits/{parsed_url.path.split('/')[-1]}"
	rs_api_url = checking_(url=api_url, prms=parameters)
	if rs_api_url:
		try:
			# hgltd_wrds = [d.get("text") for d in rs_api_url.json()]
			NWP_CONTENT_RESULTS["highlighted_term"] = [d.get("text") for d in rs_api_url.json()]
		except (json.JSONDecodeError,
						json.decoder.JSONDecodeError,
						Exception,
					) as e:
			print(f"<!ERR!> HWs: {e}")
			# hgltd_wrds = []

	api_nwp = f"https://digi.kansalliskirjasto.fi/rest/binding?id={parsed_url.path.split('/')[-1]}"
	rsp_api_nwp = checking_(url=api_nwp, prms=None)
	if rsp_api_nwp:
		try:
			nwp_info = rsp_api_nwp.json()
			NWP_CONTENT_RESULTS["title"] = nwp_info.get("bindingInformation").get("publicationTitle") # Uusi Suometar 
			NWP_CONTENT_RESULTS["document_type"] = nwp_info.get("bindingInformation").get("generalType") # NEWSPAER
			NWP_CONTENT_RESULTS["issue"] = nwp_info.get("bindingInformation").get("issue") # 63
			NWP_CONTENT_RESULTS["publisher"] = nwp_info.get("bindingInformation").get("citationInfo").get("publisher") # Uuden Suomettaren Oy
			NWP_CONTENT_RESULTS["publication_date"] = nwp_info.get("bindingInformation").get("citationInfo").get("localizedPublishingDate") # 16.03.1905
			NWP_CONTENT_RESULTS["publication_place"] = nwp_info.get("bindingInformation").get("citationInfo").get("publishingPlace") # Helsinki, Suomi
			NWP_CONTENT_RESULTS["language"] = nwp_info.get("bindingInformation").get("citationInfo").get("refWorksLanguage") # English
		except (requests.exceptions.Timeout,
						requests.exceptions.ConnectionError, 
						requests.exceptions.RequestException, 
						requests.exceptions.TooManyRedirects,
						requests.exceptions.InvalidSchema,
						json.decoder.JSONDecodeError,
						json.JSONDecodeError,
						ValueError, 
						TypeError, 
						EOFError, 
						RuntimeError,
						Exception, 
					) as e:
			print(f"<!> {type(e).__name__} line {e.__traceback__.tb_lineno} in {__file__}: {e.args}")
	print(f"\tElapsed_t: {time.time()-st_t:.3f} s")
	return NWP_CONTENT_RESULTS

if __name__ == '__main__':
	os.system("clear")
	# clippings:
	url = 'https://digi.kansalliskirjasto.fi/clippings?query=economic%20crisis&fuzzy=true&formats=NEWSPAPER&formats=JOURNAL&startDate=2023-02-21&endDate=2023-02-28&categoryId=7&categoryId=6&subjectId=6&subjectId=9&subjectId=11&title=fk20100613&title=fk20100478&keyword=%20%20%20%20turvattomuus&keyword=%20Alma%20Josefina%20Jakovaara&collection=24&collection=475&collection=477&orderBy=RELEVANCE&includeCollected=true&resultMode=THUMB'
	scrap_clipping_page(URL=url)