import re
import string
import os
import sys
import joblib
import time
import argparse
import glob

from natsort import natsorted
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

from utils import *

import matplotlib
matplotlib.use("Agg")

parser = argparse.ArgumentParser(description='National Library of Finland (NLF) Data Analysis')
parser.add_argument('--query', default=0, type=int) # smallest
args = parser.parse_args()

global result_directory

def main():
	print("#"*65)
	print(f"\tDATA ANALYSIS")
	print("#"*65)
	df = load_df(infile=get_query_dataframe(QUERY=args.query))
	print(df.shape)
	cols = list(df.columns)
	print(len(cols), cols)
	print("#"*150)

	print(df.head(10))
	print("-"*150)
	print(df.tail(10))

	print(df.isna().sum())
	print("-"*150)
	print(df.info(verbose=True, memory_usage="deep"))
	

def get_query_dataframe(QUERY=0):
	#print(f"\nGiven log file index: {QUERY}")
	all_dump_file_paths = natsorted( glob.glob( os.path.join(dfs_path, "*.dump") ) )
	#print(all_dump_file_paths)

	all_files = [dfile[dfile.rfind("/")+1:dfile.rfind(".dump")] for dfile in all_dump_file_paths]
	#print(all_files)

	query_dataframe_file = all_files[QUERY] 
	result_directory = os.path.join(rpath, query_dataframe_file)
	make_folder(folder_name=result_directory)
	print(f"Input Query: {QUERY}")

	return query_dataframe_file

if __name__ == '__main__':
	os.system('clear')
	main()