import requests
import json
import os
from requests_html import HTML, HTMLSession
from bs4 import BeautifulSoup

def broken_connection(url):
	try:
		requests.get(url)
		return False
	except requests.exceptions.ConnectionError:
		#print ("Failed to open url")
		return True

def run_req_html():
	#session = HTMLSession()
	#r = session.get(url)

	print(r.status_code, r.html)
	print()

	tst = r.html.find('app-digiweb ng-version="9.1.12"', first=True)
	print(tst)

def run_beautiful_soup(url):
	if broken_connection(url):
		return 0

	source = requests.get(url).text
	soup = BeautifulSoup(source, 'lxml')

	#print(soup.head.title) # not important!
	#print("#"*150)

	#print(soup.prettify())
	#print("#"*150)

	print(soup.body)
	print("#"*150)

	#print(dir(soup))
	#print("#"*150)

	print(soup.find('app-digiweb'))
	print("#"*150)

	print(soup.find_all("div"))
	print("#"*150)



if __name__ == '__main__':
	os.system('clear')
	url = 'https://digi.kansalliskirjasto.fi/search?query=%22Laihia%20Saari%22~8&formats=NEWSPAPER&orderBy=RELEVANCE'
	#run_req_html(url)
	run_beautiful_soup(url)