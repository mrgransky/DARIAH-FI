import requests
import json
import os
from requests_html import HTML, HTMLSession

os.system('clear')

url = 'https://digi.kansalliskirjasto.fi/search?query=%22Laihia%20Saari%22~8&formats=NEWSPAPER&orderBy=RELEVANCE'

session = HTMLSession()
r = session.get(url)

print(r.status_code, r.html)
print()

tst = r.html.find('app-digiweb ng-version="9.1.12"', first=True)
print(tst)



"""
section = r.html.find('.section', first=True)
print(section)

navigation_result = r.html.find('app-result-navigation', first=True)
print(navigation_result.html)


search_result_text = r.html.find('app-search-result-text-thumb', first=True)
print(search_result_text)
#print(search_result_text.html)
"""

