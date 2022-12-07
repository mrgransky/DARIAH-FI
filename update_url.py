import requests
import urllib


#raw_url = 'https://digi.kansalliskirjasto.fi/aikakausi/binding/498491?term=1864&term=SUOMI&language=fi'
#raw_url = 'https://twitter.com/i/user/2274951674'
raw_url = 'https://youtu.be/dQw4w9WgXcQ'
print(f">> Raw URL: {raw_url}")


#r = requests.get(raw_url)
r = requests.head(raw_url, allow_redirects=True)
r.raise_for_status()

print(f"HTTP family: {r.status_code}\tExists: {r.ok}\thistory:{r.history}")

updated_url = r.url
print(f">> Updated URL: {updated_url}")