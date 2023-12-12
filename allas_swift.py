import swiftclient
import rasterio
import geopandas as gpd
from rasterio.io import MemoryFile
import tempfile
import os
import tarfile
import json

"""
Example script for using Allas directly from a Python script with swift library
Created on 27.01.2020 by Johannes Nyman
"""

### 1. Establishing the Swift connection to Allas

# You need to run the following commands in Puhti to get the authentication to Allas active

"""
module load allas
allas-conf
"""

# These exist after running allas-conf
_authurl = os.environ['OS_STORAGE_URL']
_auth_token = os.environ['OS_AUTH_TOKEN']
_project_name = os.environ['OS_PROJECT_NAME']
_user = os.environ['OS_USERNAME']

# Various settings for connecting to Puhti
_auth_version = '3'
_os_options = {
		'user_domain_name': 'Default',
		'project_domain_name': 'Default',
		'project_name': _project_name
}

print(f"creating swift connection...")

# Creating the connection client
conn = swiftclient.Connection(
		user=_user,
		preauthurl=_authurl,
		preauthtoken=_auth_token,
		os_options=_os_options,
		auth_version=_auth_version
)

### 0.0 Looping through buckets and files inside
resp_headers, containers = conn.get_account()
for container in containers:
	print(f"bucket < {container['name']} > contains:")
	for data in conn.get_container(container['name'])[1]:
		print("\t" + container['name'] + "/" + data['name'])

### 1. Download a file from Allas to local filesystem
print(f"downloading file into local file...")
obj = 'dataframes_x2.tar'
container = 'buck_x2'
out_dir = "/scratch/project_2004072/Nationalbiblioteket/test_trash"
file_output = 'dataframes_x2_copy.tar'

headers, my_obj = conn.get_object(container, obj)

print(f"{type(headers)} {type(my_obj)}")

for k, v in headers.items():
	print(k, v)

print(f"{json.dumps(headers, indent=2, ensure_ascii=False)}")
print("#"*100)

# print(my_obj)
# print("#"*100)

# with open(os.path.join(out_dir, file_output), 'bw') as f:
# 	f.write(my_obj)
# print(f"done!")
# print(os.listdir(out_dir))


# ### 2. Writing a raster file to Allas using the Swift library
# print(f"writing...")
# fp = "<PATH-TO-LOCAL-TIF-FILE>"
# bucket_name = 'buck_x2'
# raster = rasterio.open(fp)
# input_data = raster.read()

# # The file is written to memory first and then uploaded to Allas
# with MemoryFile() as mem_file:
# 		with mem_file.open(**raster.profile) as dataset:
# 				dataset.write(input_data)
# 		conn.put_object(bucket_name, os.path.basename(fp), contents=mem_file)


# ### 3. Writing a vector file to Allas using the Swift library
# fp = "<PATH-TO-GPKG-FILE>"
# bucket_name = '<YOUR-BUCKET>'
# vector = gpd.read_file(fp)

# # The file is written to memory first and then uploaded to Allas
# tmp = tempfile.NamedTemporaryFile()
# vector.to_file(tmp, layer='test', driver="GPKG")
# tmp.seek(0) # Moving pointer to the beginning of temp file.
# conn.put_object(bucket_name, os.path.basename(fp) ,contents=tmp)