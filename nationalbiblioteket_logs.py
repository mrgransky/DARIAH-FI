import os
import re
import datetime
import numpy as np
import pandas as pd

# Apache access log format:
# 
# %h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i\
# Ex)
# 172.16.0.3 - - [25/Sep/2002:14:04:19 +0200] "GET / HTTP/1.1" 401 - "" "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.1) Gecko/20020827"


# https://regex101.com/r/xDfSqj/4

usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket/no_ip_logs',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket/no_ip_logs",
				}

dpath = usr_[os.environ['USER']]


def convert_date(INP_DATE):
	months_dict = {
		"Jan": "01", 
		"Feb": "02", 
		"Mar": "03", 
		"Apr": "04", 
		"May": "05", 
		"Jun": "06", 
		"Jul": "07", 
		"Aug": "08", 
		"Sep": "09", 
		"Oct": "10", 
		"Nov": "11", 
		"Dec": "12", 
		}

	d_list = INP_DATE.split("/")
	#print(d_list)
	d_list[1] = months_dict.get(d_list[1])
	MODIDFIED_DATE = '/'.join(d_list)
	#print(MODIDFIED_DATE)

	yyyy_mm_dd = datetime.datetime.strptime(MODIDFIED_DATE, "%d/%m/%Y").strftime("%Y-%m-%d")
	return yyyy_mm_dd

def get_df_no_ip_logs(infile=""):
	print(f">> Loading {infile} ...")
	#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^\"]+)" "(.*?)" (\d+)'
	ACCESS_LOG_PATTERN = '- - (?P<time>\[.*?\]) "(.*?)" (?P<status>\d{3}) (.*) "([^\"]+)" "(.*?)" (.*)'

	cleaned_lines = []
	with open(infile, mode="r") as f:
		for line in f:
			l = re.match(ACCESS_LOG_PATTERN, line).groups()

			dt_tz = l[0].replace("[", "").replace("]", "")
			
			DDMMYYYY = dt_tz[:dt_tz.find(":")]
			YYYYMMDD = convert_date( DDMMYYYY )
			HMS = dt_tz[dt_tz.find(":")+1:dt_tz.find(" ")]
			TZ = dt_tz[dt_tz.find(" ")+1:]

			cleaned_lines.append({
				"timestamp": 						l[0],
				"client_request_line": 	l[1],
				"status": 							l[2],
				"bytes_sent": 					l[3],
				"referer": 							l[4],
				"user_agent": 					l[5],
				"session_id": 					l[6], #TODO: must be check if key is a right name!
				"date": 								YYYYMMDD,
				"time": 								HMS,
				"timezone": 						TZ,
				})
	
	return pd.DataFrame.from_dict(cleaned_lines)

if __name__ == '__main__':
	os.system('clear')
	fname = "nike6.docworks.lib.helsinki.fi_access_log.2017-02-01.log"

	df = get_df_no_ip_logs(infile=os.path.join(dpath, fname))
	print(df.shape)
	print("-"*130)

	print( df.head(40) )
	print("-"*130)

	print( df.tail(40) )





"""
#ACCESS_LOG_PATTERN = '- - \[(.*?)\] "(.*?)" (\\d{3}) (.*) "([^\"]+)" "(.*?)" (\d+)'
ACCESS_LOG_PATTERN = '- - (?P<time>\[.*?\]) "(.*?)" (?P<status>\d{3}) (.*) "([^\"]+)" "(.*?)" (.*)'

cleaned_lines = []
with open(infile, mode="r") as f:
	for line in f:
		l = re.match(ACCESS_LOG_PATTERN, line).groups()
		cleaned_lines.append({"time_received": l[0], 
													"client_request_line": l[1],
													"status": l[2],
													"bytes_sent": l[3],
													"referer": l[4],
													"user_agent": l[5],
													"session_id": l[6], #TODO: must be check if key is a right name!
													})



print(f">> Loading complete! no_logs: {len(cleaned_lines)}")

qidx = int( np.random.randint(0, len(cleaned_lines)+1, size=1) )
print(f">> q:{qidx}: {cleaned_lines[qidx]}")	

df = pd.DataFrame.from_dict(cleaned_lines)
print( df.head() )
"""

