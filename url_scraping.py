import requests
import json
import os
from requests_html import HTML, HTMLSession
from bs4 import BeautifulSoup

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


def run_req_html():
	#session = HTMLSession()
	#r = session.get(url)

	print(r.status_code, r.html)
	print()

	tst = r.html.find('app-digiweb ng-version="9.1.12"', first=True)
	print(tst)

def get_contents(url="", name=""):
	source = requests.get(url).text
	soup = BeautifulSoup(source, 'lxml')

	doc = soup.select(name)
	print(f">> Looking for {name}\tfound {len(doc)} !")

	for idx, val in enumerate(doc):
		print(idx, val.text)

	print("#"*150)
	


def run_beautiful_soup(INP_URL):
	if checking_(INP_URL) is None:
		return

	print("*"*20)
	css_selector = ".media"
	get_contents(url=INP_URL, name=css_selector)

if __name__ == '__main__':
	os.system('clear')
	url = 'https://digi.kansalliskirjasto.fi/search?query=%22Laihia%20Saari%22~8&formats=NEWSPAPER&orderBy=RELEVANCE'
	#run_req_html(url)
	run_beautiful_soup(url)