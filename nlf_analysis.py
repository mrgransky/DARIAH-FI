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

parser = argparse.ArgumentParser(description='National Library of Finland (NLF)')
parser.add_argument('--query', default=0, type=int) # smallest
args = parser.parse_args()

def main():
	print(f">> Data Analysis")
	df = load_df(infile=get_query_dataframe(QUERY=args.query))
	print(df.shape)
	print(df.head(30))

def get_query_dataframe(QUERY=0):
	#print(f"\nGiven log file index: {QUERY}")
	dataframe_files_dir = natsorted( glob.glob( os.path.join(dfs_path, "*.dump") ) )
	#print(dataframe_files_dir)

	all_files = [dfile[dfile.rfind("/")+1:dfile.rfind(".dump")] for dfile in dataframe_files_dir]
	#print(all_files)

	query_dataframe_file = all_files[QUERY] 
	print(f"\nQ: {QUERY}\t{query_dataframe_file}")

	return query_dataframe_file

if __name__ == '__main__':
	os.system('clear')
	make_folder(folder_name=rpath)
	main()