
from utils import *

parser = argparse.ArgumentParser(	
	description='Pre processing Document for each log files of National Library of Finland (NLF) dataset', 
	prog='Preprocess Logs', 
	epilog='Developed by Farid Alijani',
)
parser.add_argument(
	'-idf', 
	'--inputDF', 
	required=True,
	type=str,
	help="Input DataFrame",
)
parser.add_argument(
	'-odir', 
	'--outDIR', 
	type=str, 
	required=True, 
	help='output directory to save files',
)

args = parser.parse_args()

# how to run (local Ubuntu 22.04.4 LTS):
# python preprocess_documents.py --inputDF ~/datasets/Nationalbiblioteket/datasets/nikeX.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR ~/datasets/Nationalbiblioteket/trash/dataframes_XXX

# how to run (Puhti):
# python preprocess_documents.py --inputDF /scratch/project_2004072/Nationalbiblioteket/datasets/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR /scratch/project_2004072/Nationalbiblioteket/dataframes_XXX

# how to run (Pouta):
# python preprocess_documents.py --inputDF /media/volume/Nationalbiblioteket/datasets/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR /media/volume/Nationalbiblioteket/dataframes_XXX
# nohup python -u preprocess_documents.py --inputDF /media/volume/Nationalbiblioteket/datasets/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR /media/volume/Nationalbiblioteket/dataframes_XXX > nikeY_stanza.out &

fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021

def main():
	print(f"Running {__file__} using {nb.get_num_threads()} CPU core(s) (GPU not required!)")
	ORIGINAL_INP_DF = load_pickle(fpath=args.inputDF)
	print(f"-"*100)
	print(f"ORIGINAL_INPUT {type(ORIGINAL_INP_DF)} {ORIGINAL_INP_DF.shape}")
	print( ORIGINAL_INP_DF.info(memory_usage="deep", verbose=True) )

	if ORIGINAL_INP_DF.shape[0] == 0:
		print(f"Empty DF: {ORIGINAL_INP_DF.shape} => Exit...")
		return
	
	os.makedirs(args.outDIR, exist_ok=True)

	preprocessed_docs_fpath = os.path.join(args.outDIR, f"{fprefix}_preprocessed_listed_docs.gz")
	preprocessed_df_fpath = os.path.join(args.outDIR, f"{fprefix}_preprocessed_df.gz")
	# with HiddenPrints(): # with no prints
	# 	_, _ = get_preprocessed_doc(
	# 		dframe=ORIGINAL_INP_DF, 
	# 		preprocessed_docs_fpath=preprocessed_docs_fpath,
	# 		preprocessed_df_fpath=preprocessed_df_fpath,
	# 	)
	_, _ = get_preprocessed_doc(
		dframe=ORIGINAL_INP_DF, 
		preprocessed_docs_fpath=preprocessed_docs_fpath,
		preprocessed_df_fpath=preprocessed_df_fpath,
	)

if __name__ == '__main__':
	print(
		f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
		.center(160, " ")
	)
	START_EXECUTION_TIME = time.time()
	main()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)