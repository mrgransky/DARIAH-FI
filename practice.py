import requests
import json
import datetime
import torch
import cupyx.scipy.sparse as cusparse
import numpy as np
import scipy.sparse as sp
import time
import cupy as cp
import argparse
from numba import njit, prange
from utils import *
import gc  # For explicit garbage collection

# run in the background using the following command in the terminal:
# $ nohup python -u practice.py -nu 1e+5 -nt 2e+6 -bs 1750 > logs/practice_results.out &

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
Files_DIR: str = "/media/volume" if USER == "ubuntu" else HOME
lmMethod: str="stanza"
nSPMs: int = 732 if USER == "ubuntu" else 2 # dynamic changing of nSPMs due to Rahti CPU memory issues!
DATASET_DIR: str = f"Nationalbiblioteket/compressed_concatenated_SPMs" if USER == "ubuntu" else f"datasets/compressed_concatenated_SPMs"
compressed_spm_file = os.path.join(Files_DIR, DATASET_DIR, f"concat_x{nSPMs}_lm_{lmMethod}.tar.gz")
fprefix: str = f"concatinated_{nSPMs}_SPMs_lm_{lmMethod}"
spm_files_dir = os.path.join(Files_DIR, DATASET_DIR, f"concat_x{nSPMs}_lm_{lmMethod}")
SEARCH_QUERY_DIGI_URL: str = "https://digi.kansalliskirjasto.fi/search?requireAllKeywords=true&query="
DIGI_HOME_PAGE_URL : str = "https://digi.kansalliskirjasto.fi"


def get_customized_cosine_similarity(spMtx, query_vec, idf_vec, spMtx_norm, exponent:float=1.0):
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

def get_customized_cosine_similarity_optimized(spMtx, query_vec, idf_vec, spMtx_norm, exponent:float=1.0):
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

def get_customized_cosine_similarity_gpu(spMtx, query_vec, idf_vec, spMtx_norm, exponent:float=1.0, batch_size:int=2048):
	try:
		print(f"[GPU Optimized] Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]}) batch_size={batch_size}".center(130, "-"))
		print(
			f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype} non_zeros={np.count_nonzero(query_vec)} (ratio={np.count_nonzero(query_vec) / query_vec.size})\n"
			f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
			f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
			f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
		)
		# Clear memory before starting
		cp.get_default_memory_pool().free_all_blocks()
		torch.cuda.empty_cache()
		device = cp.cuda.Device()
		print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode('utf-8')}")
		print(f"Initial Free GPU Memory: {device.mem_info[0] / 1024 ** 3:.2f} GB")
		st_t = time.time()
		# Convert inputs to GPU with memory management
		with cp.cuda.Device(0):
			query_vec_squeezed = cp.asarray(query_vec.ravel(), dtype=cp.float32)
			idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
			spMtx_norm_gpu = cp.asarray(spMtx_norm, dtype=cp.float32)
			# Convert sparse matrix efficiently
			spMtx_csr = spMtx.tocsr()
			spMtx_gpu = cp.sparse.csr_matrix(
				(
					cp.asarray(spMtx_csr.data, dtype=cp.float32),
				 	cp.asarray(spMtx_csr.indices),
				 	cp.asarray(spMtx_csr.indptr)
				),
				shape=spMtx_csr.shape
			)
			# Free CPU memory
			del spMtx_csr
			# Compute interest and normalization
			quInterest = query_vec_squeezed * idf_squeezed
			quInterestNorm = cp.linalg.norm(quInterest)
			idx_nonzeros = cp.nonzero(quInterest)[0]
			quInterest_nonZeros = quInterest[idx_nonzeros] / quInterestNorm
			usrInterestNorm = spMtx_norm_gpu + cp.float32(1e-4)
			# Free unnecessary arrays
			del query_vec_squeezed, quInterest
			# Initialize result array
			cs = cp.zeros(spMtx_gpu.shape[0], dtype=cp.float32)
			# Process in batches
			for i in range(0, spMtx_gpu.shape[0], batch_size):
					start_idx = i
					end_idx = min(i + batch_size, spMtx_gpu.shape[0])
					# Process batch
					spMtx_batch = spMtx_gpu[start_idx:end_idx, idx_nonzeros]
					spMtx_batch = spMtx_batch.multiply(idf_squeezed[idx_nonzeros])
					spMtx_batch = spMtx_batch.multiply(1 / usrInterestNorm[start_idx:end_idx, None])
					if exponent != 1.0:
							spMtx_batch.data **= exponent
					cs[start_idx:end_idx] = spMtx_batch.dot(quInterest_nonZeros)
					# Clean up batch memory
					del spMtx_batch
					cp.get_default_memory_pool().free_all_blocks()
					torch.cuda.empty_cache()
			# Get result and clean up
			result = cp.asnumpy(cs)
			# Final cleanup
			del cs, spMtx_gpu, idf_squeezed, spMtx_norm_gpu, quInterest_nonZeros, usrInterestNorm
			cp.get_default_memory_pool().free_all_blocks()
			torch.cuda.empty_cache()
		print(f"Elapsed_t: {time.time() - st_t:.2f} s".center(130, " "))
		return result
	except Exception as e:
			print(f"Error in cosine similarity calculation: {str(e)}")
			cp.get_default_memory_pool().free_all_blocks()
			torch.cuda.empty_cache()
			raise

# def get_customized_cosine_similarity_gpu(spMtx, query_vec, idf_vec, spMtx_norm, exponent:float=1.0, batch_size:int=512):
# 	print(f"[GPU Optimized] Customized Cosine Similarity (1 x nUsers={spMtx.shape[0]}) batch_size={batch_size}".center(130, "-"))
# 	print(
# 		f"Query: {query_vec.shape} {type(query_vec)} {query_vec.dtype} non_zeros={np.count_nonzero(query_vec)} (ratio={np.count_nonzero(query_vec) / query_vec.size})\n"
# 		f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
# 		f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
# 		f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
# 	)
# 	# Print GPU device information
# 	device = cp.cuda.Device()
# 	device_id = device.id
# 	device_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode('utf-8')
# 	print(f"Using GPU: {device_name}")
# 	print(f"Total GPU Memory: {device.mem_info[1] / 1024 ** 3:.2f} GB")
# 	print(f"Free GPU Memory: {device.mem_info[0] / 1024 ** 3:.2f} GB")

# 	st_t = time.time()
# 	# Convert inputs to CuPy arrays (float32)
# 	query_vec_squeezed = cp.asarray(query_vec.ravel(), dtype=cp.float32)
# 	idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
# 	spMtx_norm = cp.asarray(spMtx_norm, dtype=cp.float32)
# 	# Convert sparse matrix to CuPy CSR format
# 	spMtx_csr = spMtx.tocsr()
# 	spMtx_gpu = cp.sparse.csr_matrix(
# 			(cp.asarray(spMtx_csr.data, dtype=cp.float32), cp.asarray(spMtx_csr.indices), cp.asarray(spMtx_csr.indptr)),
# 			shape=spMtx_csr.shape
# 	)
# 	# Compute quInterest and its norm
# 	quInterest = query_vec_squeezed * idf_squeezed
# 	quInterestNorm = cp.linalg.norm(quInterest)
# 	# Get indices of non-zero elements in quInterest
# 	idx_nonzeros = cp.nonzero(quInterest)[0]
# 	quInterest_nonZeros = quInterest[idx_nonzeros] / quInterestNorm
# 	# Normalize user interests
# 	usrInterestNorm = spMtx_norm + cp.float32(1e-4)
# 	# Initialize result array
# 	cs = cp.zeros(spMtx_gpu.shape[0], dtype=cp.float32)
# 	# Process in batches to avoid memory overflow
# 	for i in range(0, spMtx_gpu.shape[0], batch_size):
# 			# Define batch range
# 			start_idx = i
# 			end_idx = min(i + batch_size, spMtx_gpu.shape[0])
# 			# Extract batch from sparse matrix
# 			spMtx_batch = spMtx_gpu[start_idx:end_idx, :]
# 			# Extract only the necessary columns from the batch
# 			spMtx_nonZeros = spMtx_batch[:, idx_nonzeros]
# 			# Apply IDF and normalize
# 			spMtx_nonZeros = spMtx_nonZeros.multiply(idf_squeezed[idx_nonzeros])
# 			spMtx_nonZeros = spMtx_nonZeros.multiply(1 / usrInterestNorm[start_idx:end_idx, None])
# 			# Apply exponent if necessary
# 			if exponent != 1.0:
# 					spMtx_nonZeros.data **= exponent
# 			# Compute cosine similarity scores for the batch
# 			cs_batch = spMtx_nonZeros.dot(quInterest_nonZeros)
# 			# Store batch results
# 			cs[start_idx:end_idx] = cs_batch
# 			# Free memory for the batch
# 			del spMtx_batch, spMtx_nonZeros, cs_batch
# 			cp.get_default_memory_pool().free_all_blocks()
# 			torch.cuda.empty_cache() # Clear CUDA cache
# 			# torch.cuda.synchronize() # Ensure all CUDA operations are complete
# 			# Print memory usage after each batch
# 			# print(f"Batch {i // batch_size + 1}: Free GPU Memory: {device.mem_info[0] / 1024 ** 3:.2f} GB")

# 	print(f"Elapsed_t: {time.time() - st_t:.2f} s {type(cs)} {cs.dtype} {cs.shape}".center(130, " "))
# 	return cp.asnumpy(cs)  # Convert result back to NumPy for compatibility

def get_customized_recsys_avg_vec(spMtx, cosine_sim, idf_vec, spMtx_norm):
	print(f"avgRecSys (1 x nTKs={spMtx.shape[1]})".center(130, "-"))
	st_t = time.time()
	#################################################Vectorized Version#################################################
	nUsers, nTokens = spMtx.shape
	avg_rec = np.zeros(nTokens, dtype=np.float32)
	idf_squeezed = idf_vec.ravel()
	non_zero_cosines = np.nonzero(cosine_sim)[0]
	non_zero_values = cosine_sim[non_zero_cosines]
	userInterestNorm = spMtx_norm + np.float32(1e-18)# avoid zero division
	print(
		f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
		f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
		f"CS {type(cosine_sim)} {cosine_sim.shape} {cosine_sim.dtype} NonZero(s): {non_zero_cosines.shape[0]}\n"
		f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
	)
	# Process only rows with non-zero cosine similarities
	spMtx_non_zero = spMtx[non_zero_cosines]
	
	# Element-wise multiplication with IDF vector
	userInterest = spMtx_non_zero.multiply(idf_squeezed).tocsr()
	
	# Normalize user interest vectors
	norm_factors = np.repeat(
		userInterestNorm[non_zero_cosines], 
		np.diff(userInterest.indptr)
	)
	userInterest.data /= norm_factors
	
	# Multiply by cosine similarities
	cosine_factors = np.repeat(
		non_zero_values, 
		np.diff(userInterest.indptr)
	)
	userInterest.data *= cosine_factors
	
	# Sum the weighted user interest vectors
	avg_rec = userInterest.sum(axis=0).A1
	
	# Normalize the result
	avg_rec /= np.sum(non_zero_values)
	
	print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(avg_rec)} {avg_rec.dtype} {avg_rec.shape}".center(130, " "))	
	return avg_rec

# def get_customized_recsys_avg_vec_gpu(spMtx, cosine_sim, idf_vec, spMtx_norm, batch_size=2048):
# 		print(f"[GPU optimized] avgRecSys (1 x nTKs={spMtx.shape[1]})".center(130, "-"))
# 		st_t = time.time()
		
# 		# Move data to GPU
# 		idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
# 		cosine_sim_gpu = cp.asarray(cosine_sim, dtype=cp.float32)
# 		spMtx_norm_gpu = cp.asarray(spMtx_norm, dtype=cp.float32)
		
# 		# Find non-zero cosine similarities
# 		non_zero_cosines = cp.nonzero(cosine_sim_gpu)[0]
# 		non_zero_values = cosine_sim_gpu[non_zero_cosines]
		
# 		print(
# 				f"spMtx {type(spMtx)} {spMtx.shape} {spMtx.dtype}\n"
# 				f"spMtxNorm: {type(spMtx_norm)} {spMtx_norm.shape} {spMtx_norm.dtype}\n"
# 				f"CS {type(cosine_sim)} {cosine_sim.shape} {cosine_sim.dtype} NonZero(s): {len(non_zero_cosines)}\n"
# 				f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype}"
# 		)
		
# 		# Convert sparse matrix to CuPy CSR format
# 		spMtx_csr = spMtx.tocsr()
# 		spMtx_gpu = cp.sparse.csr_matrix(
# 				(cp.asarray(spMtx_csr.data, dtype=cp.float32),
# 				 cp.asarray(spMtx_csr.indices),
# 				 cp.asarray(spMtx_csr.indptr)),
# 				shape=spMtx_csr.shape
# 		)
		
# 		# Initialize result array on GPU
# 		avg_rec = cp.zeros(spMtx.shape[1], dtype=cp.float32)
		
# 		# Process in batches
# 		for i in range(0, len(non_zero_cosines), batch_size):
# 				batch_indices = non_zero_cosines[i:i + batch_size]
# 				batch_values = non_zero_values[i:i + batch_size]
				
# 				# Extract batch from sparse matrix
# 				spMtx_batch = spMtx_gpu[batch_indices]
				
# 				# Apply IDF
# 				batch_result = spMtx_batch.multiply(idf_squeezed)
				
# 				# Normalize by user interest norm
# 				norm_factors = spMtx_norm_gpu[batch_indices] + cp.float32(1e-18)
# 				batch_result = batch_result.multiply(1.0 / norm_factors[:, None])
				
# 				# Multiply by cosine similarities
# 				batch_result = batch_result.multiply(batch_values[:, None])
				
# 				# Add to running sum
# 				avg_rec += batch_result.sum(axis=0).ravel()
				
# 				# Clean up memory
# 				del batch_result, spMtx_batch
# 				cp.get_default_memory_pool().free_all_blocks()
		
# 		# Normalize the result
# 		avg_rec /= cp.sum(non_zero_values)
		
# 		# Convert back to CPU
# 		result = cp.asnumpy(avg_rec)
		
# 		print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(result)} {result.dtype} {result.shape}".center(130, " "))
# 		return result

def get_customized_recsys_avg_vec_gpu(spMtx, cosine_sim, idf_vec, spMtx_norm, batch_size:int=2048):
	try:
		print(f"[GPU optimized] avgRecSys (1 x nTKs={spMtx.shape[1]})".center(130, "-"))
		st_t = time.time()
		# Clear memory before starting
		cp.get_default_memory_pool().free_all_blocks()
		torch.cuda.empty_cache()
		with cp.cuda.Device(0):
			# Move data to GPU efficiently
			idf_squeezed = cp.asarray(idf_vec.ravel(), dtype=cp.float32)
			cosine_sim_gpu = cp.asarray(cosine_sim, dtype=cp.float32)
			spMtx_norm_gpu = cp.asarray(spMtx_norm, dtype=cp.float32)
			# Find non-zero cosine similarities
			non_zero_cosines = cp.nonzero(cosine_sim_gpu)[0]
			non_zero_values = cosine_sim_gpu[non_zero_cosines]
			# Free unnecessary arrays
			del cosine_sim_gpu
			# Convert sparse matrix efficiently
			spMtx_csr = spMtx.tocsr()
			spMtx_gpu = cp.sparse.csr_matrix(
					(cp.asarray(spMtx_csr.data, dtype=cp.float32),
					 cp.asarray(spMtx_csr.indices),
					 cp.asarray(spMtx_csr.indptr)),
					shape=spMtx_csr.shape
			)
			del spMtx_csr
			# Initialize result array
			avg_rec = cp.zeros(spMtx.shape[1], dtype=cp.float32)
			# Process in smaller batches
			for i in range(0, len(non_zero_cosines), batch_size):
				batch_indices = non_zero_cosines[i:i + batch_size]
				batch_values = non_zero_values[i:i + batch_size]
				# Process batch
				spMtx_batch = spMtx_gpu[batch_indices]
				batch_result = spMtx_batch.multiply(idf_squeezed)
				
				norm_factors = spMtx_norm_gpu[batch_indices] + cp.float32(1e-18)
				batch_result = batch_result.multiply(1.0 / norm_factors[:, None])
				batch_result = batch_result.multiply(batch_values[:, None])
				
				avg_rec += batch_result.sum(axis=0).ravel()
				# Clean up batch memory
				del spMtx_batch, batch_result
				cp.get_default_memory_pool().free_all_blocks()
			
			# Normalize the result
			sum_non_zero_values = cp.sum(non_zero_values)
			avg_rec /= sum_non_zero_values
			# Convert back to CPU and clean up GPU memory
			result = cp.asnumpy(avg_rec)
			# Final cleanup
			del avg_rec, spMtx_gpu, idf_squeezed, spMtx_norm_gpu
			del non_zero_cosines, non_zero_values
			cp.get_default_memory_pool().free_all_blocks()
			torch.cuda.empty_cache()
		print(f"Elapsed_t: {time.time()-st_t:.2f} s {type(result)} {result.dtype} {result.shape}".center(130, " "))
		return result
	except Exception as e:
		print(f"Error in average recommendation calculation: {str(e)}")
		cp.get_default_memory_pool().free_all_blocks()
		torch.cuda.empty_cache()
		raise

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
		print(f"Selected GPU ({torch.cuda.get_device_name(device)}): cuda:{selected_device} with {max_free_memory / 1024**3:.2f} GB free memory")
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

def main():
	parser = argparse.ArgumentParser(description="Process some sparse matrix.")
	parser.add_argument('--batch_size', '-bs', type=int, default=2048, help='')
	parser.add_argument('--num_users', '-nu', type=lambda x: int(float(x)), default=int(1e+4), help='number of users')
	parser.add_argument('--num_tokens', '-nt', type=lambda x: int(float(x)), default=int(1e+6), help='number of tokens')

	args, unknown = parser.parse_known_args()
	print(args)

	n_users = int(args.num_users)
	n_features = int(args.num_tokens)
	print(f"Creating sparse matrix with {n_users} users and {n_features} features...")
	st_t = time.time()
	# Random sparse matrix:
	# spMtx = sp.random(n_users, n_features, density=0.01, format='csr', dtype=np.float32)
	# idf_vec = np.random.rand(1, n_features).astype(np.float32)
	# spMtx_norm = np.random.rand(n_users).astype(np.float32)
	# print(f"Sparse matrix created in {time.time() - st_t:.2f} seconds")

	spMtx = load_pickle(
		fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_spMtx_USERs_vs_TOKENs_*_nUSRs_x_*_nTOKs.gz' )[0]
	)
	n_users, n_features = spMtx.shape
	print(f"Sparse Matrix: {spMtx.shape}")
	idf_vec = load_pickle(
		fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_idf_vec_1_x_*_nTOKs.gz')[0]
	)

	spMtx_norm=load_pickle(
		fpath=glob.glob( spm_files_dir+'/'+f'{fprefix}'+'_shrinked_users_norm_1_x_*_nUSRs.gz')[0]
	)
	query_vec = np.random.rand(1, n_features).astype(np.float32)

	# Compute optimized cosine similarity using GPU acceleration[cupy]
	cs_optimized_gpu = get_customized_cosine_similarity_gpu(
		spMtx, 
		query_vec, 
		idf_vec, 
		spMtx_norm, 
		batch_size=args.batch_size,
	)
	print(cs_optimized_gpu[:10])

	# # Compute cosine similarity
	# cs = get_customized_cosine_similarity(spMtx, query_vec, idf_vec, spMtx_norm)
	# print(cs[:10])

	# # Compute optimized cosine similarity
	# cs_optimized = get_customized_cosine_similarity_optimized(spMtx, query_vec, idf_vec, spMtx_norm)
	# print(cs_optimized[:10])

	# print(np.allclose(cs, cs_optimized, atol=1e-2))
	# print(np.allclose(cs, cs_optimized_gpu, atol=1e-2))
	# print(np.allclose(cs_optimized, cs_optimized_gpu, atol=1e-2))

	# Compute Average Recommendation Vector:
	avgRecSys = get_customized_recsys_avg_vec(
		spMtx=spMtx,
		cosine_sim=cs_optimized_gpu,
		idf_vec=idf_vec,
		spMtx_norm=spMtx_norm,
	)
	print(avgRecSys[:10])

	# Compute [GPU optimized] Average Recommendation Vector:
	avgRecSys_optimized = get_customized_recsys_avg_vec_gpu(
		spMtx=spMtx,
		cosine_sim=cs_optimized_gpu,
		idf_vec=idf_vec,
		spMtx_norm=spMtx_norm,
		batch_size=args.batch_size,
	)
	print(avgRecSys_optimized[:10])

	# check if the two vectors are close to each other:
	print(np.allclose(avgRecSys, avgRecSys_optimized, atol=1e-2))

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	get_ip_info()
	print_ip_info()
	device = get_device_with_most_free_memory()
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))