import requests
import json
import datetime

def get_ip_info():
	"""
	Fetch and print current IP address, location, and ISP.
	"""
	try:
		response = requests.get('http://ip-api.com/json')
		data = response.json()
		ip_address = data['query']
		location = f"{data['city']}, {data['regionName']}, {data['country']}"
		isp = data['isp']
		lat, lon = data['lat'], data['lon']
		timezone = data['timezone']
		org = data['org'] # organization
		as_number = data['as']
		as_name = data.get('asname', None)
		mobile = data.get('mobile', False)
		proxy = data.get('proxy', False)
		print(f"IP Address: {ip_address} Location: {location} ISP: {isp}".center(170, "-"))
		print(f"(Latitude, Longitude): ({lat}, {lon}) Time Zone: {timezone} Organization: {org} AS Number: {as_number}, AS Name: {as_name} Mobile: {mobile}, Proxy: {proxy}")
		print("-"*170)
	except requests.exceptions.RequestException as e:
		print(f"Error: {e}")

def print_ip_info(addr=None):
		from urllib.request import urlopen
		import json
		if addr is None:
				url = 'https://ipinfo.io/json'
		else:
				url = 'https://ipinfo.io/' + addr + '/json'
		# if res==None, check your internet connection
		res = urlopen(url)
		data = json.load(res)
		for attr in data.keys():
				# print the data line by line
				print(attr, ' '*13 + '\t->\t', data[attr])

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	get_ip_info()
	print_ip_info()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))