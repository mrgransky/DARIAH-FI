import os
import requests
import json

from bs4 import BeautifulSoup

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def run_req_html():
	#session = HTMLSession()
	#r = session.get(url)

	print(r.status_code, r.html)
	print()

	tst = r.html.find('app-digiweb ng-version="9.1.12"', first=True)
	print(tst)

def checking_(url):
		try:
				r = requests.get(url)
				r.raise_for_status()
				print(r.status_code, r.ok)
				return r
		except requests.exceptions.ConnectionError as ec:
				print(f">> Connection Exception: {ec}")
				pass
		except requests.exceptions.Timeout as et:
				print(f">> Timeout Exception: {et}")
				pass
		except requests.exceptions.HTTPError as ehttp: # not 200 : not ok!
				print(f">> HTTP Exception: {ehttp}")
				print(ehttp.response.status_code)
				pass
		except requests.exceptions.RequestException as e:  # This is the correct syntax
				#raise SystemExit(e)
				print(f">> ALL Exception: {e}")
				pass

def get_contents(url="", name=""):
	my_parser = "lxml"
	#my_parser = "html.parser"
	
	source = requests.get(url).text
	soup = BeautifulSoup(source, my_parser)
	
	print(f">> results:\n{soup.find_all(name)}")

	doc = soup.select(name)
	print(f">> Looking for\t>>{name}<<\tfound: {len(doc)} !")

	for idx, val in enumerate(doc):
		print(idx, val.text)

	print("#"*150)

def run_beautiful_soup(INP_URL):
	if checking_(INP_URL) is None:
		return

	print("*"*20)
	css_selector = "head"
	get_contents(url=INP_URL, name=css_selector)

def run_selenium(URL):
	SEARCH_RESULTS = {}
	options = Options()
	options.add_argument("--remote-debugging-port=9222"),
	options.headless = True
	
	driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
	
	driver.get(URL)
	print(f">> Loading {driver.current_url} ...")
	#print(driver.page_source)
	#pt = "//section[@class='container mb-5 ng-star-inserted']"
	"""
	pt = "//app-digiweb/ng-component/section/div/div/app-binding-search-results/div/div"
	medias = driver.find_elements(By.XPATH, pt)
	"""

	medias = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'media')))
	#print(f">> Found {len(medias)} search results!")
	#print(medias)
	#print("#"*180)

	for media_idx, media_elem in enumerate(medias):
		#print(f">> Result: {media_idx}")
		outer_html = media_elem.get_attribute('outerHTML')
		
		#print(media_elem.text)
		#print()
		result = analyze_html(outer_html)
		SEARCH_RESULTS[f"search_result_{media_idx}"] = result
		#print("-"*120)
	#print(SEARCH_RESULTS)
	print(json.dumps(SEARCH_RESULTS, indent=1, ensure_ascii=False))

def analyze_html(HTML):

	#print(HTML)
	my_parser = "lxml"
	#my_parser = "html.parser"
	soup = BeautifulSoup(HTML, my_parser)
	#print(soup.prettify())

	"""
	print("#"*80)
	matches = soup.find_all("em")
	print(matches)
	print("#"*80)
	"""

	newspaper = soup.find_all("span", class_="ng-star-inserted")
	all_newspaper_info = [elem.text.replace(',', '').replace(':', '').split() for elem in newspaper]

	highlighted_text = soup.find("div", class_="search-highlight-fragment ng-star-inserted").text
	imported_date = soup.find("div", class_="import-date ng-star-inserted").text
	thumbnail = "https://digi.kansalliskirjasto.fi"+soup.find("img")["src"]

	query_newspaper = {	"newspaper_page": " ".join(all_newspaper_info[0]),#all_newspaper_info[0],
											"newspaper_issue":" ".join(all_newspaper_info[1]), 
											"newspaper_date": " ".join(all_newspaper_info[2]), 
											"newspaper_sth": " ".join(all_newspaper_info[3]),  #TODO: ask about the  
											"newspaper_city":" ".join(all_newspaper_info[4]), 
											"newspaper_thumbnail": thumbnail,
											"newspaper_import_date": imported_date,
											"newpaper_highlight": highlighted_text,
											}
	#print(query_newspaper)
	#print("#"*30)
	return query_newspaper




if __name__ == '__main__':
	os.system("clear")
	#url = 'https://digi.kansalliskirjasto.fi/search?query=sj%C3%A4lvst%C3%A4ndighetsdag&formats=NEWSPAPER&formats=JOURNAL&formats=PRINTING&formats=BOOK&formats=MANUSCRIPT&formats=MAP&formats=MUSIC_NOTATION&formats=PICTURE&formats=CARD_INDEX&orderBy=RELEVANCE'
	url = 'https://digi.kansalliskirjasto.fi/search?query=economic%20crisis&orderBy=RELEVANCE'
	#url = 'https://www.neuralnine.com/books/'
	#run_beautiful_soup(url)
	run_selenium(URL=url)