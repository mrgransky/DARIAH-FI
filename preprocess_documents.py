
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
# python preprocess_documents.py --inputDF ~/datasets/Nationalbiblioteket/NLF_DATASET/nikeX.docworks.lib.helsinki.fi_access_log.07_02_2021.log.gz --outDIR ~/datasets/Nationalbiblioteket/trash/dataframes_XXX

# how to run (Puhti):
# python preprocess_documents.py --inputDF /scratch/project_2004072/Nationalbiblioteket/datasets/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR /scratch/project_2004072/Nationalbiblioteket/dataframes_XXX

# how to run (Pouta):
# python preprocess_documents.py --inputDF /media/volume/Nationalbiblioteket/datasets/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR /media/volume/Nationalbiblioteket/dataframes_XXX
# nohup python -u preprocess_documents.py --inputDF /media/volume/Nationalbiblioteket/datasets/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --outDIR /media/volume/Nationalbiblioteket/dataframes_XXX > nikeY_stanza.out &

fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021

def get_preprocessed_doc(dframe, preprocessed_docs_fpath: str="/path/2/prerocessed_list", preprocessed_df_fpath:str="/path/2/prerocessed_df"):
	print(f"Preprocessing ORIGINAL INPUT {type(dframe)} {dframe.shape}".center(140, "-"))
	print(dframe.info(verbose=True, memory_usage="deep"))
	print("<>"*60)
	try:
		preprocessed_df = load_pickle(fpath=preprocessed_df_fpath)
		preprocessed_docs = load_pickle(fpath=preprocessed_docs_fpath)
	except Exception as e:
		print(f"<!> preprocessed file NOT found\n{e}")
		preprocessed_df = dframe.copy()
		
		print(f"{f'Extracting Raw collection query phrase(s)':<80}", end="")
		st_t = time.time()
		preprocessed_df["collection_sq_phrase"] = preprocessed_df["collection_query_phrase"].map(get_raw_sqp, na_action="ignore")
		print(f"Elapsed_t: {time.time()-st_t:.1f} s")

		print(f"{f'Extracting Cleaned collection query phrase(s)':<80}", end="")
		st_t = time.time()
		preprocessed_df["cleaned_collection_sq_phrase"] = preprocessed_df["collection_query_phrase"].map(lambda lst: get_raw_sqp(lst, cleaned_docs=True), na_action="ignore")
		print(f"Elapsed_t: {time.time()-st_t:.1f} s")

		print(f"{f'Extracting Raw clipping query phrases':<80}", end="")
		st_t = time.time()
		preprocessed_df["clipping_sq_phrase"] = preprocessed_df["clipping_query_phrase"].map(get_raw_sqp, na_action="ignore")
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Cleaned clipping query phrases':<80}", end="")
		st_t = time.time()
		preprocessed_df["cleaned_clipping_sq_phrase"] = preprocessed_df["clipping_query_phrase"].map(lambda lst: get_raw_sqp(lst, cleaned_docs=True), na_action="ignore")
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Raw search query phrase(s)':<80}", end="")
		st_t = time.time()
		preprocessed_df["sq_phrase"] = preprocessed_df["search_query_phrase"].map(get_raw_sqp, na_action="ignore")
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Cleaned search query phrase(s)':<80}", end="")
		st_t = time.time()
		preprocessed_df["cleaned_sq_phrase"] = preprocessed_df["search_query_phrase"].map(lambda lst: get_raw_sqp(lst, cleaned_docs=True), na_action="ignore")
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Raw snippets':<80}", end="")
		st_t = time.time()
		preprocessed_df['search_results_snippets'] = preprocessed_df["search_results"].map(get_raw_sn, na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Cleaned snippets':<80}", end="")
		st_t = time.time()
		preprocessed_df['cleaned_search_results_snippets'] = preprocessed_df["search_results"].map(lambda res: get_raw_sn(res, cleaned_docs=True), na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Raw snippets < HWs >':<80}", end="")
		st_t = time.time()
		preprocessed_df['search_results_hw_snippets'] = preprocessed_df["search_results"].map(get_raw_snHWs, na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Cleaned snippets < HWs >':<80}", end="")
		st_t = time.time()
		preprocessed_df['cleaned_search_results_hw_snippets'] = preprocessed_df["search_results"].map(lambda res: get_raw_snHWs(res, cleaned_docs=True), na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Raw newspaper content < HWs >':<80}", end="")
		st_t = time.time()
		preprocessed_df['nwp_content_ocr_hw'] = preprocessed_df["nwp_content_results"].map(get_raw_cntHWs, na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Cleaned newspaper content < HWs >':<80}", end="")
		st_t = time.time()
		preprocessed_df['cleaned_nwp_content_ocr_hw'] = preprocessed_df["nwp_content_results"].map(lambda res: get_raw_cntHWs(res, cleaned_docs=True), na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Raw newspaper content < PTs >':<80}", end="")
		st_t = time.time()
		preprocessed_df['nwp_content_pt'] = preprocessed_df["nwp_content_results"].map(get_raw_cntPTs, na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Cleaned newspaper content < PTs >':<80}", end="")
		st_t = time.time()
		preprocessed_df['cleaned_nwp_content_pt'] = preprocessed_df["nwp_content_results"].map(lambda res: get_raw_cntPTs(res, cleaned_docs=True), na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Raw newspaper content':<80}", end="")
		st_t = time.time()
		preprocessed_df['nwp_content_ocr'] = preprocessed_df["nwp_content_results"].map(get_raw_cnt, na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.3f} s")

		print(f"{f'Extracting Cleaned newspaper content':<80}", end="")
		st_t = time.time()
		preprocessed_df['cleaned_nwp_content_ocr'] = preprocessed_df["nwp_content_results"].map(lambda res: get_raw_cnt(res, cleaned_docs=True), na_action='ignore')
		print(f"Elapsed_t: {time.time()-st_t:.1f} s")
		
		users_list = list()
		raw_texts_list = list()

		for n, g in preprocessed_df.groupby("user_ip"):
			users_list.append(n)
			lque = [ph for ph in g[g["sq_phrase"].notnull()]["sq_phrase"].values.tolist() if len(ph) > 0 ] # ["global warming", "econimic crisis", "", ]
			lcol = [ph for ph in g[g["collection_sq_phrase"].notnull()]["collection_sq_phrase"].values.tolist() if len(ph) > 0] # ["independence day", "suomen pankki", "helsingin pörssi", ...]
			lclp = [ph for ph in g[g["clipping_sq_phrase"].notnull()]["clipping_sq_phrase"].values.tolist() if len(ph) > 0] # ["", "", "", ...]

			lsnp = [sent for el in g[g["search_results_snippets"].notnull()]["search_results_snippets"].values.tolist() if el for sent in el if sent] # ["", "", "", ...]
			lsnpHW = [sent for el in g[g["search_results_hw_snippets"].notnull()]["search_results_hw_snippets"].values.tolist() if el for sent in el if sent] # ["", "", "", ...]
			# print(f"snHW: {lsnpHW}")

			lcnt = [sent for sent in g[g["nwp_content_ocr"].notnull()]["nwp_content_ocr"].values.tolist() if sent ] # ["", "", "", ...]
			lcntHW = [word for elm in g[g["nwp_content_ocr_hw"].notnull()]["nwp_content_ocr_hw"].values.tolist() if elm for word in elm if word ] # ["", "", "", ...]
			# print(lcntHW)
			
			ltot = lque + lcol + lclp + lsnp + lcnt + lcntHW + lsnpHW
			# ltot = lque + lcol + lclp + lsnp + lcnt
			raw_texts_list.append( ltot )

		print(
			len(users_list), 
			len(raw_texts_list), 
			type(raw_texts_list), 
			any(elem is None for elem in raw_texts_list),
		)
		print(f"Creating raw_docs_list(!#>?&) [..., ['', '', ...], [''], ['', '', '', ...], ...]", end=" ")
		t0 = time.time()

		raw_docs_list = [
			subitem 
			for itm in raw_texts_list 
			if itm 
			for subitem in itm 
			if (
				subitem
				and len(subitem) > 1
				and re.search(r'[a-zA-Z|ÄäÖöÅåüÜúùßẞàñéèíóò]', subitem)
				and re.search(r"\S", subitem)
				and re.search(r"\D", subitem)
				# and max([len(el) for el in subitem.split()]) > 2 # longest word within the subitem is at least 3 characters 
				and max([len(el) for el in subitem.split()]) >= 4 # longest word within the subitem is at least 4 characters
				and re.search(r"\b(?=\D)\w{3,}\b", subitem)
			)
		]
		print(f"Elapsed_t: {time.time()-t0:.1f} s len: {len(raw_docs_list)} {type(raw_docs_list)} any None? {any(elem is None for elem in raw_docs_list)}")
		raw_docs_list = list(set(raw_docs_list))
		print(f"Cleaning {len(raw_docs_list)} unique Raw Docs [Query Search + Collection + Clipping + Snippets + Content OCR]...")

		pst = time.time()

		# with HiddenPrints(): # with no prints
		# 	preprocessed_docs = [cdocs for _, vsnt in enumerate(raw_docs_list) if ((cdocs:=clean_(docs=vsnt)) and len(cdocs)>1) ]		

		preprocessed_docs = [cdocs for _, vsnt in enumerate(raw_docs_list) if ((cdocs:=clean_(docs=vsnt)) and len(cdocs)>1) ]

		print(f"Corpus of {len(preprocessed_docs)} raw docs [d1, d2, d3, ..., dN] created in {time.time()-pst:.1f} s")
		save_pickle(pkl=preprocessed_docs, fname=preprocessed_docs_fpath)
		save_pickle(pkl=preprocessed_df, fname=preprocessed_df_fpath)

		# ###########################################################################################################################
		# print(f">> Saving cleaned_sq_phrase into excel file... ")
		# df_filtered = preprocessed_df[preprocessed_df['cleaned_sq_phrase'].notnull()]  # Filter for non-null values
		# cleaned_sq_phrase = df_filtered['cleaned_sq_phrase']  # Select the cleaned_sq_phrase column
		# cleaned_sq_phrase.to_excel(f'{preprocessed_df_fpath}_cleaned_sq_phrase.xlsx', index=False)
		# ###########################################################################################################################

	print(f"Preprocessed {type(preprocessed_df)} containing Raw & Cleaned Documents: {preprocessed_df.shape}".center(140, "-"))
	print(preprocessed_df.info(verbose=True, memory_usage="deep"))
	print(f"-"*140)

	return preprocessed_df, preprocessed_docs

@measure_execution_time
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
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))