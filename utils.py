import os
import urllib
import requests
import joblib

usr_ = {'alijani': '/lustre/sgn-data/vision', 
				'alijanif':	'/scratch/project_2004072/Nationalbiblioteket',
				"xenial": 	f"{os.environ['HOME']}/Datasets/Nationalbiblioteket",
				}

NLF_DATASET_PATH = usr_[os.environ['USER']]

dpath = os.path.join( NLF_DATASET_PATH, f"no_ip_logs" )
#dpath = os.path.join( NLF_DATASET_PATH, f"broken" )

rpath = os.path.join( NLF_DATASET_PATH, f"results" )
dfs_path = os.path.join( NLF_DATASET_PATH, f"dataframes" )

def checking_(url):
	#print(f"\t\tValidation & Update")
	try:
		r = requests.get(url)
		r.raise_for_status()
		#print(r.status_code, r.ok)
		return r
	except requests.exceptions.ConnectionError as ec:
		#print(url)
		print(f">> {url}\tConnection Exception: {ec}")
		pass
	except requests.exceptions.Timeout as et:
		#print(url)
		print(f">> {url}\tTimeout Exception: {et}")
		pass
	except requests.exceptions.HTTPError as ehttp: # not 200 : not ok!
		#print(url)
		print(f"HTTP Exception: {ehttp}\t{ehttp.response.status_code}")
		pass
	except requests.exceptions.RequestException as e:
		#print(url)
		print(f">> {url}\tALL Exception: {e}")
		pass

def make_folders():
	if not os.path.exists(rpath): 
		#print(f"\n>> Creating DIR:\n{rpath}")
		os.makedirs( rpath )

	if not os.path.exists(dfs_path): 
		#print(f"\n>> Creating DIR:\n{dfs_path}")
		os.makedirs( dfs_path )

def save_(df, infile=""):
	dfs_dict = {
		f"{infile}":	df,
	}
	
	dump_file_name = os.path.join(dfs_path, f"{infile}.dump")
	csv_file_name = os.path.join(dfs_path, f"{infile}.csv")
	
	print(f">> Saving {csv_file_name} ...")
	df.to_csv(csv_file_name, index=False)
	fsize_csv = os.stat( csv_file_name ).st_size / 1e6
	print(f"\t\t{fsize_csv:.1f} MB")

	print(f">> Saving {dump_file_name} ...")
	joblib.dump(	dfs_dict, 
								dump_file_name,
								compress='lz4', # zlib more info: https://joblib.readthedocs.io/en/latest/auto_examples/compressors_comparison.html#sphx-glr-auto-examples-compressors-comparison-py
								)
	fsize_dump = os.stat( dump_file_name ).st_size / 1e6
	print(f"\t\t{fsize_dump:.1f} MB")

def get_parsed_url_parameters(inp_url):
	#print(f"\nParsing {inp_url}")
	p_url = urllib.parse.urlparse(inp_url)
	#print(parsed_url)

	#print(f">> Explore url parameters ...")
	params = urllib.parse.parse_qs( p_url.query, keep_blank_values=True)
	#print(parameters)

	return p_url, params

def get_np_ocr(ocr_url):
	parsed_url, parameters = get_parsed_url_parameters(ocr_url)
	txt_pg_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}/page-{parameters.get('page')[0]}.txt"
	#print(f">> page-X.txt available?\t{txt_pg_url}\t")
		
	text_response = checking_(txt_pg_url)
		
	if text_response is not None: # 200
		#print(f"\t\t\tYES >> loading...\n")
		return text_response.text
