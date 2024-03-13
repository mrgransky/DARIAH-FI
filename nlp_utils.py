from tokenizer_utils import *
from utils import *

DEVICE: str="cpu"

lemmatizer_methods = {
	"nltk": nltk_lemmatizer,
	"spacy": spacy_tokenizer,
	"trankit": trankit_lemmatizer,
	"stanza": lambda docs: stanza_lemmatizer(docs, device=DEVICE),  # Pass args to stanza_lemmatizer
}

def get_agg_tk_apr(lst: List[str], wg: float, vb: Dict[str, int]):
	# print(len(vb), len(lst), wg, lst)
	result_vb: Dict[str, float]={}
	for _, vtk in enumerate(lst): # [tk1, tk2, …]
		if vb.get(vtk) is None: #["I", "go", "to", "school"] XXXXXXXXXXXXXXXXXXXXx
			# print(f"{vtk} not found! => continue...")
			continue # pass is wrong! # return is wrong!
		if result_vb.get(vtk) is not None: # check if this token is available in BoWs
			prev = result_vb.get(vtk)
			curr = prev + wg
			result_vb[vtk] = curr
		else:
			# print(f"initialize with wg={wg} for < {vtk} >")
			result_vb[vtk]=wg
	# print(f"result_vb {type(result_vb)} {len(result_vb)}")
	# print(json.dumps(result_vb, indent=2, ensure_ascii=False))
	return result_vb

def get_total_user_token_interest(df: pd.DataFrame):
	# print(f"Total USR-TOK interest of {df.user_ip} nNaNs({df.isnull().values.any()}): {df.isna().sum().sum()}")
	df = df.dropna()
	dict_usrInt_qu_tk = dict(Counter(df.usrInt_qu_tk)) if "usrInt_qu_tk" in df else dict()
	dict_usrInt_sn_hw_tk = dict(Counter(df.usrInt_sn_hw_tk)) if "usrInt_sn_hw_tk" in df else dict()
	dict_usrInt_sn_tk = dict(Counter(df.usrInt_sn_tk)) if "usrInt_sn_tk" in df else dict()
	dict_usrInt_cnt_hw_tk = dict(Counter(df.usrInt_cnt_hw_tk)) if "usrInt_cnt_hw_tk" in df else dict()
	dict_usrInt_cnt_pt_tk = dict(Counter(df.usrInt_cnt_pt_tk)) if "usrInt_cnt_pt_tk" in df else dict()
	dict_usrInt_cnt_tk = dict(Counter(df.usrInt_cnt_tk)) if "usrInt_cnt_tk" in df else dict()
	r = dict(
		Counter(dict_usrInt_qu_tk)
		+Counter(dict_usrInt_sn_hw_tk)
		+Counter(dict_usrInt_sn_tk)
		+Counter(dict_usrInt_cnt_hw_tk)
		+Counter(dict_usrInt_cnt_pt_tk)
		+Counter(dict_usrInt_cnt_tk)
	)
	result=dict( sorted( r.items() ) ) # sort by keys: ascending! A, B, .., Ö
	# print(f"Total VOCAB {df.user_ip} {type(result)} {len(result)}")
	# print(json.dumps(result, indent=2, ensure_ascii=False))
	return result

def get_lemmatized_sqp(qu_list, lm: str="stanza"):
	# qu_list = ['some word in this format with always length 1']
	# print(len(qu_list), qu_list)
	assert len(qu_list) == 1, f"query list length MUST be len(qu_list)==1, Now: {len(qu_list)}!!"
	return lemmatizer_methods.get(lm)( clean_(docs=qu_list[0]) )

def get_lemmatized_qu_(phrase: str="this is sample query", lm: str="stanza"):
	# qu_list = ['some word in this format with always length 1']
	# print(type(phrase), len(phrase), phrase)
	return lemmatizer_methods.get(lm)( docs=phrase )

def get_lemmatized_snHWs(results, lm: str="stanza"):
	# return [ tklm for el in results if ( el and len(el)>0 and ( lemmas:=lemmatizer_methods.get(lm)( clean_(docs=el) ) ) ) for tklm in lemmas if tklm ]
	return [ tklm for el in results if ( el and len(el)>0 and ( lemmas:=lemmatizer_methods.get(lm)( docs=el ) ) ) for tklm in lemmas if tklm ]

def get_lemmatized_cntHWs(results, lm: str="stanza"):
	# return [ tklm for el in results if ( el and len(el)>0 and ( lemmas:=lemmatizer_methods.get(lm)( clean_(docs=el) ) ) ) for tklm in lemmas if tklm ]
	return [ tklm for el in results if ( el and len(el)>0 and ( lemmas:=lemmatizer_methods.get(lm)( docs=el ) ) ) for tklm in lemmas if tklm ]

def get_lemmatized_cntPTs(results, lm: str="stanza"):
	# return [tklm for el in results if ( el and len(el)>0 and ( lemmas:=lemmatizer_methods.get(lm)( clean_(docs=el) ) ) ) for tklm in lemmas if tklm ]
	return [tklm for el in results if ( el and len(el)>0 and ( lemmas:=lemmatizer_methods.get(lm)( docs=el ) ) ) for tklm in lemmas if tklm ]

def get_lemmatized_sn(results, lm: str="stanza"):
	return [ tklm for el in results if ( el and len(el)>0 and (lemmas:=lemmatizer_methods.get(lm)( el ) ) ) for tklm in lemmas if tklm ]
	## XXXX not required to clean since results are already cleaned and saved in df XXXX
	# return [ tklm for el in results if ( el and len(el)>0 and (lemmas:=lemmatizer_methods.get(lm)( clean_(docs=el) ) ) ) for tklm in lemmas if tklm ]
	
def get_lemmatized_cnt(sentences: str="This is a sample text!", lm: str="stanza"):
	return lemmatizer_methods.get(lm)( docs=sentences )
	## XXXX not required to clean since sentences are already cleaned and saved in df XXXX
	# return lemmatizer_methods.get(lm)(clean_(docs=sentences))

def get_BoWs(preprocessed_docs, saveDIR: str="DIR", fprefix: str="fname_prefix", lm: str="stanza", MIN_DF: int=10, MAX_DF: float=0.8, MAX_FEATURES: int=None, device_: str="cpu"):
	print(f"{f'Bag-of-Words {userName} device: {device_} | Corpus of {len(preprocessed_docs)} preprocessed DOCs'.center(150, '-')}")
	global DEVICE
	DEVICE = device_
	print(f"#"*100)
	print(DEVICE)
	print(f"#"*100)

	tfidf_vec_fpath = os.path.join(saveDIR, f"{fprefix}_lemmaMethod_{lm}_tfidf_vec.gz")
	tfidf_rf_matrix_fpath = os.path.join(saveDIR, f"{fprefix}_lemmaMethod_{lm}_tfidf_matrix.gz")

	try:
		tfidf_matrix = load_pickle(fpath=tfidf_rf_matrix_fpath)
		tfidf_vec = load_pickle(fpath=tfidf_vec_fpath)
	except Exception as e:
		print(f"<!> TFIDF does not exist\n{e}")
		################################################################################################################################################################
		# max_df is used for removing terms that appear too frequently, also known as "corpus-specific stop words". For example:

		#     max_df = 0.50: ignore terms that appear in more than 50% of the documents

		# The default max_df is 1.0, which means "ignore terms that appear in more than 100% of the documents". Thus, the default setting does not ignore any terms.

		# min_df is used for removing terms that appear too infrequently. For example:

		#     min_df = 5: ignore terms that appear in less than 5 documents

		# The default min_df is 1, which means "ignore terms that appear in less than 1 document". Thus, the default setting does not ignore any terms.
		################################################################################################################################################################
		# Initialize TFIDF # not time consuming...
		print(
			f"TFIDF min_df[int]: {MIN_DF} "
			f"max_df[float]: {MAX_DF}] "
			f"max_feat: {MAX_FEATURES} "
			f"{len(preprocessed_docs)} raw corpus [d1, d2, d3, ..., dN]"
			.center(160, " ")
		)
		st_t = time.time()
		tfidf_vec=TfidfVectorizer(
			tokenizer=lemmatizer_methods.get(lm),
			lowercase=True,
			analyzer="word",
			dtype=np.float32,
			use_idf=True, # def: True => Enable inverse-document-frequency reweighting. If False, idf(t) = 1.
			# max_features=MAX_FEATURES, # retreive all features, DEFAULT: NONE!
			# max_df=MAX_DF, # ignore terms appear in more than P% of documents 1.0 does not ignore any terms # removing terms appearing too frequently
			# min_df=MIN_DF, # cut-off: ignore terms that have doc_freq strictly lower than the given threshold # removing terms appearing too infrequently
			token_pattern=None,
		)
		# Fit TFIDF # TIME CONSUMING:
		tfidf_matrix = tfidf_vec.fit_transform(raw_documents=preprocessed_docs)
		print(f"Elapsed_t: {time.time()-st_t:.2f} s".center(80, " "))
		save_pickle(pkl=tfidf_vec, fname=tfidf_vec_fpath)
		save_pickle(pkl=tfidf_matrix, fname=tfidf_rf_matrix_fpath)
	
	print(f">> Creating [sorted] BoWs: (token: idx) TFID: {type(tfidf_matrix)} {tfidf_matrix.dtype} {tfidf_matrix.shape}")
	
	# BOWs = dict( sorted( tfidf_vec.vocabulary_.items(), key=lambda x:x[1], reverse=False ) ) # ascending
	BOWs = {k: int(v) for k, v in sorted(tfidf_vec.vocabulary_.items(), key=lambda item: item[1])} # ascending

	save_vocab(	
		vb=BOWs,
		fname=os.path.join(saveDIR, f"{fprefix}_lemmaMethod_{lm}_{len(tfidf_vec.vocabulary_)}_vocabs.json"),
	)

	feat_names = tfidf_vec.get_feature_names_out()

	# example:
	# vb = {"example": 0, "is": 1, "simple": 2, "this": 3}
	#	   		example   is         simple     this	
	# 0  		0.377964  0.377964   0.377964   0.377964

	print(f"Features: {type(feat_names)} {feat_names.dtype} {feat_names.shape}  BoWs: {len(BOWs)} {type(BOWs)} TFIDF: {tfidf_matrix.shape}")
	assert len(BOWs) == tfidf_matrix.shape[1], f"size of vocabs: {len(BoWs)} != tfidf_matrix: {tfidf_matrix.shape[1]}"
	print(f"{f'Bag-of-Words [ Complete: {userName} ]'.center(110, '-')}")
	return BOWs