import os
import sys
import requests
import json
import logging
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils import *

def get_all_search_details(URL):
	SEARCH_RESULTS = {}
	options = Options()
	options.add_argument("--remote-debugging-port=9222")
	options.add_argument("--disable-extensions")
	options.addArguments("--no-sandbox")
	options.add_experimental_option("excludeSwitches", ["enable-automation"])
	options.add_experimental_option('useAutomationExtension', False)
	options.headless = True
	
	driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
	
	driver.get(URL)
	print(f"Scraping {driver.current_url}")
	#print(driver.page_source)
	try:
		#medias = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'media')))
		medias = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'result-row')))
	except Exception as e:
		print(f">> {type(e).__name__} line {e.__traceback__.tb_lineno} of {__file__}: {e.args}")
		#logging.error(e, exc_info=True)
		return
	except:
		print(f">> General Exception: {URL}")
		return

	#print(f">> Found {len(medias)} search results!")
	#print(medias)
	#print("#"*180)

	for media_idx, media_elem in enumerate(medias):
		#print(f">> Result: {media_idx}")
		outer_html = media_elem.get_attribute('outerHTML')
		
		#print(media_elem.text)
		#print()
		result = scrap_newspaper(outer_html)
		SEARCH_RESULTS[f"result_{media_idx}"] = result
		#print("-"*120)
	#print(SEARCH_RESULTS)
	#print(json.dumps(SEARCH_RESULTS, indent=1, ensure_ascii=False))
	#df = pd.DataFrame.from_dict(SEARCH_RESULTS, orient='index').reset_index()
	#print(df)
	#print("-"*120)
	return SEARCH_RESULTS

def scrap_newspaper(HTML):
	query_newspaper = dict.fromkeys([	
		"newspaper_page", 
		"newspaper_issue", 
		"newspaper_date", 
		"newspaper_publisher", 
		"newspaper_city", 
		"newspaper_thumbnail",
		"newspaper_import_date",
		"newspaper_highlight",
		"newspaper_link",
		"newspaper_content_ocr",
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

	pg_highlight = soup.find("div", class_="search-highlight-fragment ng-star-inserted")
	pg_imported_date = soup.find("div", class_="import-date ng-star-inserted")	
	thumbnail = soup.find("img")
	pg_link = soup.find("a")
	
	if thumbnail: query_newspaper["newspaper_thumbnail"] = "https://digi.kansalliskirjasto.fi" + thumbnail.get("src")
	if pg_link: query_newspaper["newspaper_link"] = "https://digi.kansalliskirjasto.fi" + pg_link.get("href")
	if np_title: query_newspaper["newspaper_title"] = np_title.text
	if pg_imported_date: query_newspaper["newspaper_import_date"] = pg_imported_date.text
	if pg_highlight: query_newspaper["newspaper_highlight"] = pg_highlight.text
	
	if all_newspaper_info[-1]: query_newspaper["newspaper_page"] = all_newspaper_info[-1].split()[1] # remove sivu, sida page: ex) sivu 128 => 128
	if all_newspaper_info[1]: query_newspaper["newspaper_issue"] = all_newspaper_info[1]
	if all_newspaper_info[2]: query_newspaper["newspaper_date"] = all_newspaper_info[2]
	if all_newspaper_info[3]: query_newspaper["newspaper_publisher"] = all_newspaper_info[3]
	if all_newspaper_info[4]: query_newspaper["newspaper_city"] = all_newspaper_info[4]
	
	if pg_link: query_newspaper["newspaper_content_ocr"] = get_np_ocr(query_newspaper["newspaper_link"])

	#print(query_newspaper)
	#print("#"*80)
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

if __name__ == '__main__':
	os.system("clear")
	# ['nro', '4', 'Näköispainos','6.12.1939']:
	#url = 'https://digi.kansalliskirjasto.fi/search?query=sj%C3%A4lvst%C3%A4ndighetsdag&formats=NEWSPAPER&formats=JOURNAL&formats=PRINTING&formats=BOOK&formats=MANUSCRIPT&formats=MAP&formats=MUSIC_NOTATION&formats=PICTURE&formats=CARD_INDEX&orderBy=RELEVANCE'
	#url = 'https://digi.kansalliskirjasto.fi/search?query=economic%20crisis&orderBy=RELEVANCE'
	#url = 'https://digi.kansalliskirjasto.fi/search?formats=JOURNAL' # without page content and city print(len(all_newspaper_info), all_newspaper_info) # < 6!!!
	#url = 'https://digi.kansalliskirjasto.fi/search?formats=JOURNAL'
	url = 'https://digi.kansalliskirjasto.fi/search?query=sj%C3%A4lvst%C3%A4ndighetsdag&formats=JOURNAL&orderBy=RELEVANCE' # <6 : 4
	#url = 'https://digi.kansalliskirjasto.fi/search?query=%22TAAVI%20ELIAKSENPOIKA%22%20AND%20%22SIPPOLA%22&formats=NEWSPAPER&orderBy=RELEVANCE' # no result is returned! => timeout!
	get_all_search_details(URL=url)
	#get_np_info()
