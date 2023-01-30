import os
import sys
import requests
import json
import logging
import pandas as pd
import numpy as np
from utils import *

import time
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import exceptions

def get_all_search_details(URL):
	st_t = time.time()

	SEARCH_RESULTS = {}
	
	options = Options()
	options.headless = True

	#options.add_argument("--remote-debugging-port=9230") # alternative: 9222
	options.add_argument("--remote-debugging-port=9222")
	options.add_argument("--no-sandbox")
	options.add_argument("--disable-gpu")
	options.add_argument("--disable-dev-shm-usage")
	options.add_argument("--disable-extensions")
	options.add_experimental_option("excludeSwitches", ["enable-automation"])
	options.add_experimental_option('useAutomationExtension', False)
	
	driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
	
	driver.get(URL)
	print(f"Scraping {driver.current_url}")
	try:
		medias = WebDriverWait(driver, 
													timeout=10,
													).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'result-row'))) # alternative for 'result-row': 'media'
		for media_idx, media_elem in enumerate(medias):
			#print(f">> Result: {media_idx}")
			outer_html = media_elem.get_attribute('outerHTML')
			
			#print(media_elem.text)
			#print()
			result = scrap_newspaper(outer_html)
			SEARCH_RESULTS[f"result_{media_idx}"] = result
			#print("-"*120)
	except (exceptions.StaleElementReferenceException,
					exceptions.NoSuchElementException,
					exceptions.TimeoutException,
					exceptions.WebDriverException,
					exceptions.SessionNotCreatedException,
					exceptions.InvalidArgumentException,
					exceptions.InvalidSessionIdException,
					exceptions.InsecureCertificateException,
					ValueError,
					TypeError,
					EOFError,
					AttributeError,
					RuntimeError,
					Exception,
					) as e:
		print(f"\t<!> Selenium: {type(e).__name__} line {e.__traceback__.tb_lineno} of {__file__}: {e.args}")
		return
	print(f"\t\t\tFound {len(medias)} media(s) => {len(SEARCH_RESULTS)} search result(s) | Elapsed_t: {time.time()-st_t:.2f} s")
	#print(json.dumps(SEARCH_RESULTS, indent=2, ensure_ascii=False))
	return SEARCH_RESULTS

def scrap_newspaper(HTML):
	query_newspaper = dict.fromkeys([
		"newspaper_title",
		"newspaper_issue", 
		"newspaper_date", 
		"newspaper_publisher", 
		"newspaper_publication_place", 
		"newspaper_page", 
		"newspaper_import_date",
		"newspaper_thumbnail",
		"newspaper_snippet",
		"newspaper_snippet_highlighted_words",
		"newspaper_content_ocr",
		"newspaper_content_ocr_highlighted_words",
		"newspaper_link",
		"newspaper_document_type",
		])

	my_parser = "lxml"
	#my_parser = "html.parser"
	soup = BeautifulSoup(HTML, my_parser)
	#print(soup.prettify())
	#return None

	all_newspaper_info = get_np_info(INP_SOUP=soup)
	#print(len(all_newspaper_info), all_newspaper_info)

	np_title = soup.find("div", class_="main-link-title link-colors")
	np_issue_date = soup.find("span", class_="font-weight-bold")

	pg_snippet = soup.find("div", class_="search-highlight-fragment ng-star-inserted")	
	pg_imported_date = soup.find("div", class_="import-date ng-star-inserted")
	thumbnail = soup.find("img")
	pg_link = soup.find("a")
	
	if thumbnail: query_newspaper["newspaper_thumbnail"] = "https://digi.kansalliskirjasto.fi" + thumbnail.get("src")
	if pg_link: query_newspaper["newspaper_link"] = "https://digi.kansalliskirjasto.fi" + pg_link.get("href")
	if np_title: query_newspaper["newspaper_title"] = np_title.text
	if pg_imported_date: query_newspaper["newspaper_import_date"] = pg_imported_date.text
	if pg_snippet:
		query_newspaper["newspaper_snippet"] = pg_snippet.text
		query_newspaper["newspaper_snippet_highlighted_words"] = [tag.text for tag in pg_snippet.findChildren('em' , recursive=False)]
	
	if all_newspaper_info[-1]: query_newspaper["newspaper_page"] = all_newspaper_info[-1].split()[1] # remove sivu, sida page: ex) sivu 128 => 128
	#if all_newspaper_info[1]: query_newspaper["newspaper_issue"] = all_newspaper_info[1]
	#if all_newspaper_info[2]: query_newspaper["newspaper_date"] = all_newspaper_info[2]
	if all_newspaper_info[3]: query_newspaper["newspaper_publisher"] = all_newspaper_info[3]
	if all_newspaper_info[4]: query_newspaper["newspaper_publication_place"] = all_newspaper_info[4]
	
	# OCR Content:
	if pg_link: 
		ttl, dtyp, issue, publisher, pub_date, pub_place, lang, trm, hw, pg, txt = scrap_ocr_page_content(query_newspaper["newspaper_link"])
		query_newspaper["newspaper_content_ocr"] = txt
		query_newspaper["newspaper_content_ocr_highlighted_words"] = hw
		query_newspaper["newspaper_issue"] = issue
		query_newspaper["newspaper_date"] = pub_date
		query_newspaper["newspaper_document_type"] = dtyp
		query_newspaper["newspaper_language"] = lang
	return query_newspaper

def get_np_info(INP_SOUP):	
	selectors = [
		'span.badge.badge-secondary.ng-star-inserted',                                             #badge 
		'span.font-weight-bold span.ng-star-inserted:has(span[translate])',                        #issue 
		'span.font-weight-bold span.ng-star-inserted:last-child',                                  #date 
		'span.font-weight-bold ~ span.ng-star-inserted:-soup-contains(", ")',                      #publisher 
		'span.font-weight-bold ~ span.ng-star-inserted:-soup-contains(", ") + span.ng-star-inserted:-soup-contains(", ")', #city 
		'span.font-weight-bold ~ span.ng-star-inserted:-soup-contains(":")',                       #page 
	]
	desired_list = [ None if s[0] is None else ( s[0].get_text(' ', strip=True)[2:] if '-soup-contains' in s[1] else s[0].get_text(' ', strip=True)) for s in [ ( INP_SOUP.select_one(sel), sel) for sel in selectors] ]
	return desired_list

def query_rest_api():
	url = 'https://digi.kansalliskirjasto.fi/search?page=5&query=Starast&formats=NEWSPAPER'
	#url = 'https://digi.kansalliskirjasto.fi/search?query=Starast&formats=NEWSPAPER' # without page: page=1
	parsed_url, parameters = get_parsed_url_parameters(url)
	print(f"Parameters: {parameters}")
	print(f"parsed_url : {parsed_url}")
	if parameters.get('page'):
		#print(type(parameters.get('page')[0]), parameters.get('page')[0])
		pg = str((int(parameters.get('page')[0]) - 1) * 20 )
	else:
		pg = '0'
	#print(pg)
	#api_search_results_url = f"https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset={pg}&count={'20'}"
	
	api_search_results_url = "https://digi.kansalliskirjasto.fi/search"
	print(api_search_results_url)
	nwp_info = requests.get(api_search_results_url, params=parameters).json()
	
	print(len(nwp_info.get("rows")))

def scrap_ocr_page_content(URL):
	#print(f"\n>> NWP: {URL}")
	if "page=" in URL:
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
	text_response = checking_(txt_pg_url)
	if text_response: # 200
		txt = text_response.text
	else:
		txt = None

	return title, doc_type, issue, publisher, pub_date, pub_place, lang, parameters.get("term"), hgltd_wrds, parameters.get("page"), txt
		

if __name__ == '__main__':
	os.system("clear")
	#url = 'https://digi.kansalliskirjasto.fi/search?query=sj%C3%A4lvst%C3%A4ndighetsdag&formats=NEWSPAPER&formats=JOURNAL&formats=PRINTING&formats=BOOK&formats=MANUSCRIPT&formats=MAP&formats=MUSIC_NOTATION&formats=PICTURE&formats=CARD_INDEX&orderBy=RELEVANCE'
	#url = 'https://digi.kansalliskirjasto.fi/search?page=36&query=kantasonni&formats=NEWSPAPER&orderBy=RELEVANCE'
	url = 'https://digi.kansalliskirjasto.fi/search?query=economic%20crisis&orderBy=RELEVANCE'
	#url = 'https://digi.kansalliskirjasto.fi/search?query=sj%C3%A4lvst%C3%A4ndighetsdag&formats=JOURNAL&orderBy=RELEVANCE' # <6 : 4
	#url = 'https://digi.kansalliskirjasto.fi/search?page=62&query=Katri%20ikonen%20&orderBy=DATE'
	#url = 'https://digi.kansalliskirjasto.fi/search?query=%22TAAVI%20ELIAKSENPOIKA%22%20AND%20%22SIPPOLA%22&formats=NEWSPAPER&orderBy=RELEVANCE' # no result is returned! => timeout!
	get_all_search_details(URL=url)
	#get_np_info()
	#query_rest_api()
	#scrap_ocr_page_content(URL='https://digi.kansalliskirjasto.fi/sanomalehti/binding/2247833?term=självständighetsdagen&term=Självständighetsdagen')