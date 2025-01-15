import requests
import json
import datetime
import torch

import numpy as np
import scipy.sparse as sp
import time
from numba import njit, prange

def get_customized_cosine_similarity(spMtx, query_vec, idf_vec, spMtx_norm, exponent: float=1.0):
	print(f"Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]})".center(130, "-"))
	print(
		f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype}\n"
		f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
		f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
		f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
	)
	st_t=time.time()
	################################### Vectorized Implementation ##########################################
	idf_squeezed = idf_vec.ravel()
	query_vec_squeezed = query_vec.ravel()
	quInterest = query_vec_squeezed * idf_squeezed # Element-wise multiplication
	quInterestNorm = np.linalg.norm(quInterest)
	
	idx_nonzeros = np.nonzero(quInterest)[0] # Get the indices of non-zero elements in quInterest
	quInterest_nonZeros = quInterest[idx_nonzeros] / quInterestNorm
	usrInterestNorm = spMtx_norm + np.float32(1e-18)
	
	# Extract only the necessary columns from the sparse matrix
	spMtx_nonZeros = spMtx[:, idx_nonzeros].tocsc()  # Converting to CSC for faster column slicing
	
	# Calculate user interest by element-wise multiplication with IDF
	spMtx_nonZeros = spMtx_nonZeros.multiply(idf_squeezed[idx_nonzeros])
	
	# Normalize user interests
	spMtx_nonZeros = spMtx_nonZeros.multiply(1 / usrInterestNorm[:, None])
	
	# Apply exponent if necessary
	if exponent != 1.0:
		spMtx_nonZeros.data **= exponent
	
	cs = spMtx_nonZeros.dot(quInterest_nonZeros) # Compute the cosine similarity scores
	
	print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(cs)} {cs.dtype} {cs.shape}".center(130, " "))
	return cs
	################################### Vectorized Implementation ##########################################

def get_customized_cosine_similarity_optimized(spMtx, query_vec, idf_vec, spMtx_norm, exponent: float = 1.0):
		print(f"[Optimized] Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]})".center(130, "-"))
		print(
				f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype}\n"
				f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
				f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
				f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
		)
		st_t = time.time()

		# Ensure inputs are in the correct format
		query_vec_squeezed = query_vec.ravel().astype(np.float32)
		idf_squeezed = idf_vec.ravel().astype(np.float32)
		spMtx_norm = spMtx_norm.astype(np.float32)

		# Compute quInterest and its norm
		quInterest = query_vec_squeezed * idf_squeezed
		quInterestNorm = np.linalg.norm(quInterest)

		# Get indices of non-zero elements in quInterest
		idx_nonzeros = np.nonzero(quInterest)[0]
		quInterest_nonZeros = quInterest[idx_nonzeros] / quInterestNorm

		# Normalize user interests
		usrInterestNorm = spMtx_norm + np.float32(1e-18)

		# Extract only the necessary columns from the sparse matrix
		spMtx_nonZeros = spMtx[:, idx_nonzeros].tocsc()  # Convert to CSC for faster column slicing

		# Apply IDF and normalize
		spMtx_nonZeros = spMtx_nonZeros.multiply(idf_squeezed[idx_nonzeros])
		spMtx_nonZeros = spMtx_nonZeros.multiply(1 / usrInterestNorm[:, None])

		# Apply exponent if necessary
		if exponent != 1.0:
				spMtx_nonZeros.data **= exponent

		# Compute cosine similarity scores
		cs = spMtx_nonZeros.dot(quInterest_nonZeros)

		print(f"Elapsed_t: {time.time() - st_t:.2f} s {type(cs)} {cs.dtype} {cs.shape}".center(130, " "))
		return cs

def get_device_with_most_free_memory():
	if torch.cuda.is_available():
		print(f"Available GPU(s) = {torch.cuda.device_count()}")
		max_free_memory = 0
		selected_device = 0
		for i in range(torch.cuda.device_count()):
			torch.cuda.set_device(i)
			free_memory = torch.cuda.mem_get_info()[0]
			if free_memory > max_free_memory:
				max_free_memory = free_memory
				selected_device = i
		device = torch.device(f"cuda:{selected_device}")
		print(f"Selected GPU: cuda:{selected_device} with {max_free_memory / 1024**3:.2f} GB free memory")
	else:
		device = torch.device("cpu")
		print("No GPU available, using CPU")
	return device

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
	device = get_device_with_most_free_memory()
	print(device)
	# Example data
	# n_users = 306357
	# n_features = 6504704

	n_users = int(3e+4)
	n_features = int(8e+5)

	spMtx = sp.random(n_users, n_features, density=0.01, format='csr', dtype=np.float32)
	query_vec = np.random.rand(1, n_features).astype(np.float32)
	idf_vec = np.random.rand(1, n_features).astype(np.float32)
	spMtx_norm = np.random.rand(n_users).astype(np.float32)

	# Compute cosine similarity
	cs = get_customized_cosine_similarity(spMtx, query_vec, idf_vec, spMtx_norm)
	print(cs)

	# Compute optimized cosine similarity
	cs_optimized = get_customized_cosine_similarity_optimized(spMtx, query_vec, idf_vec, spMtx_norm)
	print(cs_optimized)

	# Compare results
	print(np.allclose(cs, cs_optimized))

	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))