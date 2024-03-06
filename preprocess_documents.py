
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
parser.add_argument(
	'--lmMethod', 
	default="stanza", 
	type=str,
)

args = parser.parse_args()

# how to run (local Ubuntu 22.04.4 LTS):
# python preprocess_documents.py --inputDF ~/datasets/Nationalbiblioteket/datasets/nikeX.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR ~/datasets/Nationalbiblioteket/trash/dataframes_XXX

# how to run (Puhti):
# python preprocess_documents.py --inputDF /scratch/project_2004072/Nationalbiblioteket/datasets/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR /scratch/project_2004072/Nationalbiblioteket/dataframes_XXX

fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021

def main():
	print(f"Running {__file__} with {args.lmMethod.upper()} lemmatizer & {nb.get_num_threads()} CPU core(s) (GPU not required!)")
	df_inp = load_pickle(fpath=args.inputDF)
	print(f"-"*100)
	print(f"df_inp: {df_inp.shape} | {type(df_inp)}")
	print( df_inp.info(memory_usage="deep", verbose=True) )
	print(f"-"*100)

	if df_inp.shape[0] == 0:
		print(f"Empty DF: {df_inp.shape} => Exit...")
		return
	
	os.makedirs(args.outDIR, exist_ok=True)
	preprocessed_docs_fpath = os.path.join(args.outDIR, f"{fprefix}_lemmaMethod_{args.lmMethod}_preprocessed_docs.gz")
	_ = get_preprocessed_document(dframe=df_inp, preprocessed_docs_fpath=preprocessed_docs_fpath)

if __name__ == '__main__':
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(140, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(140, " "))