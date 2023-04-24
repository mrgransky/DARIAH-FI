from utils import *
from tokenizer_utils import *

parser = argparse.ArgumentParser(	description='User-Item Recommendation system developed based on National Library of Finland (NLF) dataset', 
																	prog='RecSys USER-TOKEN', 
																	epilog='Developed by Farid Alijani',
																)

parser.add_argument('--inputDF', default=os.path.join(dfs_path, "nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump"), type=str) # smallest
parser.add_argument('--qphrase', default="Juha Sipilä", type=str)
parser.add_argument('--lmMethod', default="stanza", type=str)
parser.add_argument('--normSP', default=False, type=bool)
parser.add_argument('--topTKs', default=5, type=int)
args = parser.parse_args()
# how to run:
# python RecSys_usr_token.py --inputDF ~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

lemmatizer_methods = {"nltk": nltk_lemmatizer,
											"spacy": spacy_tokenizer,
											"trankit": trankit_lemmatizer,
											"stanza": stanza_lemmatizer,
											}

RES_DIR = make_result_dir(infile=args.inputDF)

MODULE=60

# list of all weights:
weightQueryAppearance = 1.0 					# suggested by Jakko: 1.0
weightSnippetHWAppearance = 0.25			# suggested by Jakko: 0.2
weightSnippetAppearance = 0.2 				# suggested by Jakko: 0.2
weightContentHWAppearance = 0.055			# suggested by Jakko: 0.05
weightContentParsedAppearance = 0.005	# Did not consider initiially!
weightContentAppearance = 0.05 				# suggested by Jakko: 0.05

w_list = [weightQueryAppearance, 
					weightSnippetHWAppearance,
					weightSnippetAppearance,
					weightContentHWAppearance,
					weightContentParsedAppearance,
					weightContentAppearance,
				]

def get_qu_phrase_raw_text(phrase_list):
	assert len(phrase_list) == 1, f"Wrong length for {phrase_list}"
	phrase = phrase_list[0]
	return phrase

def get_snippet_raw_text(search_results_list):
	#snippets_list = [sn.get("textHighlights").get("text") for sn in search_results_list if sn.get("textHighlights").get("text") ] # [["sentA"], ["sentB"], ["sentC"]]
	snippets_list = [sent for sn in search_results_list if sn.get("textHighlights").get("text") for sent in sn.get("textHighlights").get("text")] # ["sentA", "sentB", "sentC"]
	return ' '.join(snippets_list)

def get_complete_BoWs(dframe,):
	print(f"{f'Bag-of-Words [ Complete: {userName} ]'.center(110, '-')}")
	print(f">> Extracting texts from query phrases...")
	st_t = time.time()
	dframe["query_phrase_raw_text"] = dframe["search_query_phrase"].map(get_qu_phrase_raw_text, na_action="ignore")
	print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

	print(f">> Extracting texts from newspaper content...")
	st_t = time.time()
	dframe['ocr_raw_text'] = dframe["nwp_content_results"].map(get_nwp_content_raw_text, na_action='ignore')
	print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
	
	print(f">> Extracting texts from snippets...")
	st_t = time.time()
	dframe['snippet_raw_text'] = dframe["search_results"].map(get_snippet_raw_text, na_action='ignore')
	print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
	
	#print(dframe.info())
	#print(dframe[["user_ip", "query_phrase_raw_text", "snippet_raw_text", "ocr_raw_text"]].tail(60))
	#print(f"<>"*120)
	#return

	users_list = list()
	raw_texts_list = list()
	
	for n, g in dframe.groupby("user_ip"):
		users_list.append(n)
		lq = [ phrases for phrases in g[g["query_phrase_raw_text"].notnull()]["query_phrase_raw_text"].values.tolist() if len(phrases) > 0 ]
		ls = [ sentences for sentences in g[g["snippet_raw_text"].notnull()]["snippet_raw_text"].values.tolist() if len(sentences) > 0 ]
		lc = [ sentences for sentences in g[g["ocr_raw_text"].notnull()]["ocr_raw_text"].values.tolist() if len(sentences) > 0 ]
		ltot = lq + ls + lc
		raw_texts_list.append( ltot )

	print(len(users_list), len(raw_texts_list), )
	"""
	raw_docs_list = list()
	for itm in raw_texts_list:
		#print(f">>>> item: (none: {itm is None})\tlen: {len(itm)}: {itm}")
		if ( itm is not None and len(itm) > 0 ):
			for subitem in itm:
				#print(f"<<!>> subitem: (none: {subitem is None})\tlen: {len(subitem)}")
				#if ( subitem.isalpha() or subitem.isdigit() ):
				if re.search(r"\S", subitem):
					#print(f"<> {subitem}")
					raw_docs_list.append(subitem)
	"""
	raw_docs_list = [subitem for itm in raw_texts_list if ( itm is not None and len(itm) > 0 ) for subitem in itm if re.search(r"\S", subitem) ]

	print(len(raw_docs_list), type(raw_docs_list), any(elem is None for elem in raw_docs_list))

	fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	tfidf_vec_fpath = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_tfidf_vectorizer_large.lz4")
	tfidf_rf_matrix_fpath = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_tfidf_matrix_RF_large.lz4")

	if not os.path.exists(tfidf_rf_matrix_fpath):
		print(f"Training TFIDF vector for {len(raw_docs_list)} raw words/phrases/sentences, might take a while...".center(150, " "))
		st_t = time.time()

		# Fit TFIDF # not time consuming...
		tfidf_vec = TfidfVectorizer(#min_df=5,
															#ngram_range=(1, 2),
															tokenizer=lemmatizer_methods.get(args.lmMethod),
															#stop_words=UNIQUE_STOPWORDS,
															)

		tfidf_matrix_rf = tfidf_vec.fit_transform(raw_documents=raw_docs_list)
		#tfidf_matrix_rf = np.random.choice(10_000, 10_000)

		save_pickle(pkl=tfidf_vec, fname=tfidf_vec_fpath)
		save_pickle(pkl=tfidf_matrix_rf, fname=tfidf_rf_matrix_fpath)
		save_vocab(	vb=dict( sorted( tfidf_vec.vocabulary_.items(), key=lambda x:x[1], reverse=False ) ), 
								fname=os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_{len(tfidf_vec.vocabulary_)}_vocabs.json"),
							)

		print(f"\t\tElapsed_t: {time.time()-st_t:.2f} s")
	else:
		tfidf_vec = load_pickle(fpath=tfidf_vec_fpath)
		tfidf_matrix_rf = load_pickle(fpath=tfidf_rf_matrix_fpath)
	#return

	feat_names = tfidf_vec.get_feature_names_out()
	BOWs = dict( sorted( tfidf_vec.vocabulary_.items(), key=lambda x:x[1], reverse=False ) ) # ascending
	# example:
	# vb = {"example": 0, "is": 1, "simple": 2, "this": 3}
	#	   		example   is         simple     this	
	# 0  		0.377964  0.377964   0.377964   0.377964

	print(f"Features: {feat_names.shape} | {type(feat_names)} | BoWs: {len(BOWs)} | {type(BOWs)}")
	print(f"TFIDF REF matrix: {tfidf_matrix_rf.shape}")
	assert len(BOWs) == tfidf_matrix_rf.shape[1] # to ensure size of vocabs are not different in saved files
	print(f"{f'Bag-of-Words [ Complete: {userName} ]'.center(110, '-')}")
	return BOWs

def get_bag_of_words(dframe,):
	print(f"{f'Bag-of-Words [{userName}]'.center(110, '-')}")
	print(f">> Extracting texts from query phrases...")
	st_t = time.time()
	dframe["query_phrase_raw_text"] = dframe["search_query_phrase"].map(get_qu_phrase_raw_text, na_action="ignore")
	print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

	users_list = list()
	raw_texts_list = list()
	
	for n, g in dframe.groupby("user_ip"):
		users_list.append(n)
		lq = [phrases for phrases in g[g["query_phrase_raw_text"].notnull()]["query_phrase_raw_text"].values.tolist() if len(phrases) > 0]
		ltot = lq
		raw_texts_list.append( ltot )

	print(len(users_list), len(raw_texts_list),)

	raw_docs_list = [subitem for itm in raw_texts_list if ( itm is not None and len(itm) > 0 ) for subitem in itm if re.search(r"\S", subitem) ]
	print(len(raw_docs_list), type(raw_docs_list), any(elem is None for elem in raw_docs_list))

	fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	tfidf_vec_fpath = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_tfidf_vectorizer.lz4")
	tfidf_rf_matrix_fpath = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_tfidf_matrix_RF.lz4")

	if not os.path.exists(tfidf_vec_fpath):
		print(f"Training TFIDF vector for {len(raw_docs_list)} raw words/phrases, might take a while...".center(110, " "))
		st_t = time.time()

		# Fit TFIDF # not time consuming...
		tfidf_vec = TfidfVectorizer(#min_df=5,
															#ngram_range=(1, 2),
															tokenizer=lemmatizer_methods.get(args.lmMethod),
															#stop_words=UNIQUE_STOPWORDS,
															)

		tfidf_matrix_rf = tfidf_vec.fit_transform(raw_documents=raw_docs_list)
		#tfidf_matrix_rf = np.random.choice(10_000, 10_000)

		save_pickle(pkl=tfidf_vec, fname=tfidf_vec_fpath)
		save_pickle(pkl=tfidf_matrix_rf, fname=tfidf_rf_matrix_fpath)
		save_vocab(	vb=dict( sorted( tfidf_vec.vocabulary_.items(), key=lambda x:x[1], reverse=False ) ), 
								fname=os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_{len(tfidf_vec.vocabulary_)}_vocabs.json"),
							)
		print(f"\t\tElapsed_t: {time.time()-st_t:.2f} s")
	else:
		tfidf_vec = load_pickle(fpath=tfidf_vec_fpath)
		tfidf_matrix_rf = load_pickle(fpath=tfidf_rf_matrix_fpath)
	#return

	feat_names = tfidf_vec.get_feature_names_out()
	#print(f"1st 100 features:\n{feat_names[:60]}\n")
	
	# dictionary mapping from words to their indices in vocabulary:
	BOWs = dict( sorted( tfidf_vec.vocabulary_.items(), key=lambda x:x[1], reverse=False ) ) # ascending

	# example:
	# vb = {"example": 0, "is": 1, "simple": 2, "this": 3}
	#	   		example   is         simple     this	
	# 0  		0.377964  0.377964   0.377964   0.377964

	print(f"Features: {feat_names.shape} | {type(feat_names)} | BOWs: {len(BOWs)} | {type(BOWs)}")
	print(f"TFIDF REF matrix: {tfidf_matrix_rf.shape}")
	assert len(BOWs) == tfidf_matrix_rf.shape[1] # to ensure size of vocabs are not different in saved files
	print(f"{f'Bag-of-Words [{userName}]'.center(110, '-')}")
	return BOWs

def sum_tk_apperance_vb(dframe, qcol, wg, vb):
	updated_vb = dict.fromkeys(vb.keys(), 0.0)
	for tk in dframe[qcol]:
		if updated_vb.get(tk) is not None:
			updated_vb[tk] = updated_vb.get(tk) + wg
			#print(tk, wg, updated_vb[tk])
	#print(f"{dframe.user_ip}".center(50, '-'))
	return updated_vb

def sum_all_tokens_appearance_in_vb(dframe, weights_list, vb):
	w_qu, w_hw_sn, w_sn, w_hw_cnt, w_pt_cnt, w_cnt = weights_list
	updated_vocab = dict.fromkeys(vb.keys(), 0.0)

	for q_tk in dframe.qu_tokens:
		if updated_vocab.get(q_tk) is not None:
			updated_vocab[q_tk] = updated_vocab.get(q_tk) + w_qu

	for sn_hw_tk in dframe.snippets_hw_tokens:
		if updated_vocab.get(sn_hw_tk) is not None:
			updated_vocab[sn_hw_tk] = updated_vocab.get(sn_hw_tk) + w_hw_sn
		
	for sn_tk in dframe.snippets_tokens:
		if updated_vocab.get(sn_tk) is not None:
			updated_vocab[sn_tk] = updated_vocab.get(sn_tk) + w_sn
		
	for c_hw_tk in dframe.nwp_content_hw_tokens:
		if updated_vocab.get(c_hw_tk) is not None:
			updated_vocab[c_hw_tk] = updated_vocab.get(c_hw_tk) + w_hw_cnt
		
	for c_pt_tk in dframe.nwp_content_pt_tokens:
		if updated_vocab.get(c_pt_tk) is not None:
			updated_vocab[c_pt_tk] = updated_vocab.get(c_pt_tk) + w_pt_cnt
		
	for c_tk in dframe.nwp_content_tokens:
		if updated_vocab.get(c_tk) is not None:
			updated_vocab[c_tk] = updated_vocab.get(c_tk) + w_cnt

	return updated_vocab

def get_search_results_snippet_text(search_results_list):
	#snippets_list = [sn.get("textHighlights").get("text") for sn in search_results_list if sn.get("textHighlights").get("text") ] # [["sentA"], ["sentB"], ["sentC"]]
	snippets_list = [sent for sn in search_results_list if sn.get("textHighlights").get("text") for sent in sn.get("textHighlights").get("text")] # ["sentA", "sentB", "sentC"]
	return snippets_list

def get_search_results_hw_snippets(search_results_list):
	#hw_snippets = [sn.get("terms") for sn in search_results_list if ( sn.get("terms") and len(sn.get("terms")) > 0 )] # [["A"], ["B"], ["C"]]
	hw_snippets = [w for sn in search_results_list if ( sn.get("terms") and len(sn.get("terms")) > 0 ) for w in sn.get("terms")] # ["A", "B", "C"]

	return hw_snippets

def get_nwp_content_raw_text(cnt_dict):
	return cnt_dict.get("text")

def get_nwp_content_pt(cnt_dict):
	return cnt_dict.get("parsed_term")

def get_nwp_content_hw(cnt_dict):
	return cnt_dict.get("highlighted_term")

def tokenize_hw_snippets(results_list):
	return [tklm for el in results_list for tklm in lemmatizer_methods.get(args.lmMethod)(el)]
	
def tokenize_snippets(results_list):
	return [tklm for el in results_list for tklm in lemmatizer_methods.get(args.lmMethod)(el)]

def tokenize_nwp_content(sentences):
	return lemmatizer_methods.get(args.lmMethod)(sentences)

def tokenize_hw_nwp_content(results_list):
	return [tklm for el in results_list for tklm in lemmatizer_methods.get(args.lmMethod)(el)]

def tokenize_pt_nwp_content(results_list):
	return [tklm for el in results_list for tklm in lemmatizer_methods.get(args.lmMethod)(el)]

def tokenize_query_phrase(qu_list):
	# qu_list = ['some word in this format with always length 1']
	#print(len(qu_list), qu_list)
	assert len(qu_list) == 1, f"query list length MUST be 1, it is now {len(qu_list)}!!"
	return lemmatizer_methods.get(args.lmMethod)(qu_list[0])

def get_usr_tk_df(dframe, bow):
	fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	df_preprocessed_fname = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_df_preprocessed.lz4")
	print(f"\n>> Getting {df_preprocessed_fname} ...")
	
	try:
		df_preprocessed = load_pickle(fpath=df_preprocessed_fname)
		print(f"\tLoaded from {df_preprocessed_fname} successfully!")
	except:
		print(f"Updating Original DF: {dframe.shape} with Tokenized texts".center(110, "-"))
		df_preprocessed = dframe.copy()
		print(f">> Analyzing query phrases [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed["search_query_phrase_tklm"] = df_preprocessed["search_query_phrase"].map(tokenize_query_phrase , na_action="ignore")
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

		print(f">> Analyzing highlighted words in snippets [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['search_results_hw_snippets'] = df_preprocessed["search_results"].map(get_search_results_hw_snippets, na_action='ignore')
		df_preprocessed['search_results_hw_snippets_tklm'] = df_preprocessed["search_results_hw_snippets"].map(tokenize_hw_snippets, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

		print(f">> Analyzing highlighted words in newspaper content [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['nwp_content_ocr_text_hw'] = df_preprocessed["nwp_content_results"].map(get_nwp_content_hw, na_action='ignore')
		df_preprocessed['nwp_content_ocr_text_hw_tklm'] = df_preprocessed["nwp_content_ocr_text_hw"].map(tokenize_hw_nwp_content, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

		print(f">> Analyzing parsed terms in newspaper content [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['nwp_content_ocr_text_pt'] = df_preprocessed["nwp_content_results"].map(get_nwp_content_pt, na_action='ignore')
		df_preprocessed['nwp_content_ocr_text_pt_tklm'] = df_preprocessed["nwp_content_ocr_text_pt"].map(tokenize_pt_nwp_content, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		
		#####################################################
		# comment for speedup:
		print(f">> Analyzing newspaper content [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['nwp_content_ocr_text'] = df_preprocessed["nwp_content_results"].map(get_nwp_content_raw_text, na_action='ignore')
		df_preprocessed['nwp_content_ocr_text_tklm'] = df_preprocessed["nwp_content_ocr_text"].map(tokenize_nwp_content, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		
		print(f">> Analyzing snippets [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['search_results_snippets'] = df_preprocessed["search_results"].map(get_search_results_snippet_text, na_action='ignore')
		df_preprocessed['search_results_snippets_tklm'] = df_preprocessed["search_results_snippets"].map(tokenize_snippets, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		#####################################################
		
		save_pickle(pkl=df_preprocessed, fname=df_preprocessed_fname)
	
	print(f"Original_DF: {dframe.shape} => DF_preprocessed: {df_preprocessed.shape}".center(110, "-"))

	#print(df_preprocessed.info())
	#print(df_preprocessed.tail(60))
	
	print(f"USER-TOKENS DF".center(100, "-"))
	users_list = list()
	search_query_phrase_tokens_list = list()
	search_results_hw_snippets_tokens_list = list()
	search_results_snippets_tokens_list = list()

	nwp_content_pt_tokens_list = list()
	nwp_content_hw_tokens_list = list()
	nwp_content_tokens_list = list()
	
	for n, g in df_preprocessed.groupby("user_ip"):
		users_list.append(n)

		search_query_phrase_tokens_list.append( [tk for tokens in g[g["search_query_phrase_tklm"].notnull()]["search_query_phrase_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		search_results_hw_snippets_tokens_list.append( [tk for tokens in g[g["search_results_hw_snippets_tklm"].notnull()]["search_results_hw_snippets_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		nwp_content_hw_tokens_list.append( [tk for tokens in g[g["nwp_content_ocr_text_hw_tklm"].notnull()]["nwp_content_ocr_text_hw_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		nwp_content_pt_tokens_list.append( [tk for tokens in g[g["nwp_content_ocr_text_pt_tklm"].notnull()]["nwp_content_ocr_text_pt_tklm"].values.tolist() if tokens for tk in tokens if tk] )

		# comment for speedup:
		search_results_snippets_tokens_list.append( [tk for tokens in g[g["search_results_snippets_tklm"].notnull()]["search_results_snippets_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		nwp_content_tokens_list.append([tk for tokens in g[g["nwp_content_ocr_text_tklm"].notnull()]["nwp_content_ocr_text_tklm"].values.tolist() if tokens for tk in tokens if tk] )

	# uncomment for speedup:
	#nwp_content_tokens_list = [f"nwp_content_{i}" for i in range(len(users_list))]
	#search_results_snippets_tokens_list = [f"snippet_{i}" for i in range(len(users_list))]

	print(len(users_list), 
				len(search_query_phrase_tokens_list),
				len(search_results_hw_snippets_tokens_list),
				len(search_results_snippets_tokens_list),
				len(nwp_content_tokens_list),
				len(nwp_content_pt_tokens_list),
				len(nwp_content_hw_tokens_list),
				)
	#return

	df_user_token = pd.DataFrame(list(zip(users_list, 
																				search_query_phrase_tokens_list, 
																				search_results_hw_snippets_tokens_list, 
																				search_results_snippets_tokens_list, 
																				nwp_content_tokens_list, 
																				nwp_content_pt_tokens_list, 
																				nwp_content_hw_tokens_list,
																			)
																	),
																columns =['user_ip', 
																					'qu_tokens', 
																					'snippets_hw_tokens', 
																					'snippets_tokens', 
																					'nwp_content_tokens', 
																					'nwp_content_hw_tokens', 
																					'nwp_content_pt_tokens',
																				]
															)
	#print(df_user_token[["user_ip", "qu_tokens", "snippets_tokens", "snippets_hw_tokens", "nwp_content_tokens", "nwp_content_hw_tokens", "nwp_content_pt_tokens"]].head(50))
	#print("#"*100)
	#return
	
	print(f">> Creating Implicit Feedback for user interests...")
	df_user_token["user_token_interest"] = df_user_token.apply( lambda x_df: sum_all_tokens_appearance_in_vb(x_df, w_list, bow), axis=1, )
	
	df_user_token["usrInt_qu_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="qu_tokens", wg=weightQueryAppearance, vb=bow), axis=1)
	df_user_token["usrInt_sn_hw_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="snippets_hw_tokens", wg=weightSnippetHWAppearance, vb=bow), axis=1)
	df_user_token["usrInt_sn_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="snippets_tokens", wg=weightSnippetAppearance, vb=bow), axis=1)
	df_user_token["usrInt_cnt_hw_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="nwp_content_hw_tokens", wg=weightContentHWAppearance, vb=bow), axis=1)
	df_user_token["usrInt_cnt_pt_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="nwp_content_pt_tokens", wg=weightContentParsedAppearance, vb=bow), axis=1)
	df_user_token["usrInt_cnt_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="nwp_content_tokens", wg=weightContentAppearance, vb=bow), axis=1)
	
	#print(type( df_user_token["user_token_interest"].values.tolist()[0] ), type( df_user_token["usrInt_qu_tk"].values.tolist()[0] ))
	#print( len(df_user_token["user_token_interest"].values.tolist()), df_user_token["user_token_interest"].values.tolist() )
	#print( len(df_user_token["usrInt_qu_tk"].values.tolist()), df_user_token["usrInt_qu_tk"].values.tolist() )
	print(df_user_token.shape, list(df_user_token.columns))
	
	#print(df_user_token.info())
	#print("#"*100)
	
	#print(df_user_token[["user_ip", "usrInt_qu_tk"]].head())
	#print("#"*100)
	#print(df_user_token[["user_ip", "user_token_interest"]].head())
	
	df_user_token_fname = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_user_tokens_df_{len(bow)}_BoWs.lz4")
	save_pickle(pkl=df_user_token, fname=df_user_token_fname)

	#print(f"USER-TOKENS DF".center(100, "-"))
	return df_user_token

def get_sparse_matrix(df):
	print(f"Getting Sparse Matrix from DF: {df.shape}".center(110, '-'))
	print(list(df.columns))
	#print(df.dtypes)

	#print( df['user_token_interest'].apply(pd.Series).head(15) )
	#print(">"*100)
	df_new = pd.concat( [df["user_ip"], df['user_token_interest'].apply(pd.Series)], axis=1).set_index("user_ip")

	#print(df_new.head(15))
	#print("<"*100)

	##########################Sparse Matrix info##########################
	sparse_matrix = csr_matrix(df_new.values, dtype=np.float32) # (n_usr x n_vb)
	print("#"*110)
	print(f"{type(sparse_matrix)} {sparse_matrix.shape} : |tot_elem|: {sparse_matrix.shape[0]*sparse_matrix.shape[1]}")
	print(f"<> Non-zeros vals: {sparse_matrix.data}")# Viewing stored data (not the zero items)
	#print(sparse_matrix.toarray()[:25, :18])
	print(f"<> |Non-zero vals|: {sparse_matrix.count_nonzero()}") # Counting nonzeros
	print("#"*110)
	##########################Sparse Matrix info##########################
	fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	sp_mat_user_token_fname = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_user_tokens_sparse_matrix_{sparse_matrix.shape[1]}_BoWs.lz4")

	save_pickle(pkl=sparse_matrix, fname=sp_mat_user_token_fname)

	return sparse_matrix

def run_RecSys(df_inp, qu_phrase, topK=5, normalize_sp_mtrx=False, ):
	#print_df_detail(df=df_inp, fname=__file__)
	#return
	print(f">> Running {__file__} with {args.lmMethod.upper()} lemmatizer")
	"""
	if userName.endswith("xenial"):
		BoWs = get_bag_of_words(dframe=df_inp)
	else:
		BoWs = get_complete_BoWs(dframe=df_inp)
	#return
	"""
	BoWs = get_bag_of_words(dframe=df_inp)
	#BoWs = get_complete_BoWs(dframe=df_inp)
	
	try:
		df_usr_tk = load_pickle(fpath=os.path.join(dfs_path, f"{get_filename_prefix(dfname=args.inputDF)}_lemmaMethod_{args.lmMethod}_user_tokens_df_{len(BoWs)}_BoWs.lz4"))
	except:
		df_usr_tk = get_usr_tk_df(dframe=df_inp, bow=BoWs)
	
	#print(df_usr_tk.info())
	print(f"Users-Tokens DF {df_usr_tk.shape} {list(df_usr_tk.columns)}")
	"""
	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 800):
		print( df_usr_tk[["user_ip", "user_token_interest",]].tail(10) )

	#print(df_usr_tk.info())

	print("#"*150)

	print(json.dumps(df_usr_tk.loc[int(df_usr_tk.index[df_usr_tk['user_ip'] == "ip6840"].tolist()[0]), "usrInt_qu_tk"], indent=2, ensure_ascii=False))
	print("#"*150)

	with open("ip6840.json", "w") as fw:
		json.dump(df_usr_tk[df_usr_tk["user_ip"]=="ip6840"]["user_token_interest"].tolist(), fw, indent=4, ensure_ascii=False)

	with open("ip3665.json", "w") as fw:
		json.dump(df_usr_tk[df_usr_tk["user_ip"]=="ip3665"]["user_token_interest"].tolist(), fw, indent=4, ensure_ascii=False)

	with open("ip6020.json", "w") as fw:
		json.dump(df_usr_tk[df_usr_tk["user_ip"]=="ip6020"]["user_token_interest"].tolist(), fw, indent=4, ensure_ascii=False)
	
	with open("ip6840_qu.json", "w") as fw:
		json.dump(df_usr_tk.loc[int(df_usr_tk.index[df_usr_tk['user_ip'] == "ip6840"].tolist()[0]), "usrInt_qu_tk"], fw, indent=4, ensure_ascii=False)

	with open("ip6840_sn.json", "w") as fw:
		json.dump(df_usr_tk.loc[int(df_usr_tk.index[df_usr_tk['user_ip'] == "ip6840"].tolist()[0]), "usrInt_sn_tk"], fw, indent=4, ensure_ascii=False)

	with open("ip6840_cnt.json", "w") as fw:
		json.dump(df_usr_tk.loc[int(df_usr_tk.index[df_usr_tk['user_ip'] == "ip6840"].tolist()[0]), "usrInt_cnt_tk"], fw, indent=4, ensure_ascii=False)

	with open("ip6840_sn_hw.json", "w") as fw:
		json.dump(df_usr_tk.loc[int(df_usr_tk.index[df_usr_tk['user_ip'] == "ip6840"].tolist()[0]), "usrInt_sn_hw_tk"], fw, indent=4, ensure_ascii=False)

	with open("ip6840_cnt_hw.json", "w") as fw:
		json.dump(df_usr_tk.loc[int(df_usr_tk.index[df_usr_tk['user_ip'] == "ip6840"].tolist()[0]), "usrInt_cnt_hw_tk"], fw, indent=4, ensure_ascii=False)

	with open("ip6840_cnt_pt.json", "w") as fw:
		json.dump(df_usr_tk.loc[int(df_usr_tk.index[df_usr_tk['user_ip'] == "ip6840"].tolist()[0]), "usrInt_cnt_pt_tk"], fw, indent=4, ensure_ascii=False)

	#return
	"""

	try:
		sp_mat_rf = load_pickle(fpath=os.path.join(dfs_path, f"{get_filename_prefix(dfname=args.inputDF)}_lemmaMethod_{args.lmMethod}_user_tokens_sparse_matrix_{len(BoWs)}_BoWs.lz4"))
	except:
		sp_mat_rf = get_sparse_matrix(df_usr_tk)

	print(f"Sparse Matrix (Users-Tokens) | {type(sp_mat_rf)} {sp_mat_rf.shape}")
	print(type(sp_mat_rf), sp_mat_rf.shape, sp_mat_rf.toarray().nbytes, sp_mat_rf.min(), sp_mat_rf.max())

	if normalize_sp_mtrx:
		sp_mat_rf = normalize(sp_mat_rf, norm="l2", axis=0) # l2 normalize by column -> items
		
	#get_user_n_maxVal_byTK(sp_mat_rf, df_usr_tk, BoWs, )
	#return
	plot_heatmap_sparse(sp_mat_rf, df_usr_tk, BoWs, norm_sp=normalize_sp_mtrx)
	
	#print("#"*150)
	print(f"Input Raw Query Phrase: {qu_phrase}".center(100,' '))

	query_phrase_tk = tokenize_query_phrase(qu_list=[qu_phrase])
	print(f"\nTokenized & Lemmatized '{qu_phrase}' contains {len(query_phrase_tk)} element(s)\t{query_phrase_tk}")

	query_vector = np.zeros(len(BoWs))
	for qutk in query_phrase_tk:
		#print(qutk, BoWs.get(qutk))
		if BoWs.get(qutk):
			query_vector[BoWs.get(qutk)] += 1.0

	print(f">> queryVec in vocab\tAllzero: {np.all(query_vector==0.0)}\t"
				f"( |NonZeros|: {np.count_nonzero(query_vector)} @ idx(s): {np.nonzero(query_vector)[0]} )")

	if np.all( query_vector==0.0 ):
		print(f"Sorry, We couldn't find tokenized words similar to {Fore.RED+Back.WHITE}{qu_phrase}{Style.RESET_ALL} in our BoWs! Search other phrases!")
		return

	cos_sim = get_cosine_similarity(query_vector, sp_mat_rf.toarray(), qu_phrase, query_phrase_tk, df_usr_tk, norm_sp=normalize_sp_mtrx) # qu_ (nItems,) => (1, nItems) -> cos: (1, nUsers)
	"""
	print(f"Cosine Similarity (1 x nUsers): {cos_sim.shape} {type(cos_sim)}\t" 
				f"Allzero: {np.all(cos_sim.flatten()==0.0)}\t"
				f"(min, max, sum): ({cos_sim.min()}, {cos_sim.max():.2f}, {cos_sim.sum():.2f})"
			)
	"""
	if np.all(cos_sim.flatten()==0.0):
		print(f"Sorry, We couldn't find similar results to >> {Fore.RED+Back.WHITE}{qu_phrase}{Style.RESET_ALL} << in our database! Search again!")
		return

	plot_tokens_by_max(cos_sim, sp_mtrx=sp_mat_rf, users_tokens_df=df_usr_tk, bow=BoWs, norm_sp=normalize_sp_mtrx)
	#return
	nUsers, nItems = sp_mat_rf.shape
	print(f"Getting avgRecSysVec (1 x nItems) | Users: {nUsers} vs. Items: {nItems}".center(110, "-"))
	#print("#"*120)
	#cos = np.random.rand(nUsers).reshape(1, -1)
	#usr_itm = np.random.randint(100, size=(nUsers, nItems))
	avgrec = np.zeros((1, nItems))
	#print(f"> avgrec{avgrec.shape}:\n{avgrec}")
	#print()
	
	#print(f"> cos{cos.shape}:\n{cos}")
	#print()

	#print(f"> user-item{usr_itm.shape}:\n{usr_itm}")
	#print("#"*100)
	st_t = time.time()
	for iUser in range(nUsers):
		userInterest = sp_mat_rf.toarray()[iUser, :]
		"""
		print(f"user: {iUser} | {df_usr_tk.loc[iUser, 'user_ip']}".center(100, " "))
		print(f"<> userInterest: {userInterest.shape} " 
					f"(min, max_@(idx), sum): ({userInterest.min()}, {userInterest.max():.2f}_@(idx: {np.argmax(userInterest)}), {userInterest.sum():.1f}) "
					f"{userInterest} | Allzero: {np.all(userInterest==0.0)}"
				)
		
		print(f"avgrec (previous): {avgrec.shape} "
					f"(min, max_@(idx), sum): ({avgrec.min()}, {avgrec.max():.2f}_@(idx: {np.argmax(avgrec)}), {avgrec.sum():.1f}) "
					f"{avgrec} | Allzero: {np.all(avgrec==0.0)}"
				)
		"""
		#userInterest = userInterest / np.linalg.norm(userInterest)
		userInterest = np.where(np.linalg.norm(userInterest) != 0, userInterest/np.linalg.norm(userInterest), 0.0)
		"""
		print(f"<> userInterest(norm): {userInterest.shape} " 
					f"(min, max_@(idx), sum): ({userInterest.min()}, {userInterest.max():.2f}_@(idx: {np.argmax(userInterest)}), {userInterest.sum():.1f}) "
					f"{userInterest} | Allzero: {np.all(userInterest==0.0)}"
				)

		print(f"cos[{iUser}]: {cos_sim[0, iUser]}")
		"""
		avgrec = avgrec + (cos_sim[0, iUser] * userInterest)
		"""
		print(f"avgrec (current): {avgrec.shape} "
					f"(min, max_@(idx), sum): ({avgrec.min()}, {avgrec.max():.2f}_@(idx: {np.argmax(avgrec)}), {avgrec.sum():.1f}) "
					f"{avgrec} | Allzero: {np.all(avgrec==0.0)}"
				)
		print("-"*110)
		"""
	avgrec = avgrec / np.sum(cos_sim)
	
	print(f"avgRecSys: {avgrec.shape} {type(avgrec)}\t" 
				f"Allzero: {np.all(avgrec.flatten() == 0.0)}\t"
				f"(min, max, sum): ({avgrec.min()}, {avgrec.max():.2f}, {avgrec.sum():.2f})")
	
	all_recommended_tks = [k for idx in avgrec.flatten().argsort()[-50:] for k, v in BoWs.items() if (idx not in np.nonzero(query_vector)[0] and v==idx)]
	print(f"ALL (15): {len(all_recommended_tks)} : {all_recommended_tks[-15:]}")
	
	topK_recommended_tokens = all_recommended_tks[-(topK+0):]
	print(f"top-{topK} recommended Tokens: {len(topK_recommended_tokens)} : {topK_recommended_tokens}")
	topK_recommended_tks_weighted_user_interest = [ avgrec.flatten()[BoWs.get(vTKs)] for iTKs, vTKs in enumerate(topK_recommended_tokens)]
	print(f"top-{topK} recommended Tokens weighted user interests: {len(topK_recommended_tks_weighted_user_interest)} : {topK_recommended_tks_weighted_user_interest}")
	
	print(f"Elapsed_t: {time.time()-st_t:.2f} s".center(80, " "))
	print(f"-"*100)
	#return

	print(f"Getting users of {len(topK_recommended_tokens)} tokens of top-{topK} RecSys".center(100, "-"))
	for ix, tkv in enumerate(topK_recommended_tokens):
		users_names, users_values_total, users_values_separated = get_users_byTK(sp_mat_rf, df_usr_tk, BoWs, token=tkv)
		
		plot_users_by(token=tkv, usrs_name=users_names, usrs_value_all=users_values_total, usrs_value_separated=users_values_separated, topUSRs=20, norm_sp=normalize_sp_mtrx )
		plot_usersInterest_by(token=tkv, sp_mtrx=sp_mat_rf, users_tokens_df=df_usr_tk, bow=BoWs, norm_sp=normalize_sp_mtrx)
	
	print(f"DONE".center(100, "-"))
	print()
	print(f"Implicit Feedback Recommendation: {f'Unique Users: {nUsers} vs. Tokenzied word Items: {nItems}'}".center(150,'-'))
	print(f"Since you searched for query phrase(s)\t{Fore.BLUE+Back.YELLOW}{args.qphrase}{Style.RESET_ALL}"
				f"\tTokenized + Lemmatized: {query_phrase_tk}\n"
				f"you might also be interested in Phrases:\n{Fore.GREEN}{topK_recommended_tokens[::-1]}{Style.RESET_ALL}")
	print()
	print(f"{f'Top-{topK+0} Tokens':<20}{f'Weighted userInterest {avgrec.shape} (min, max, sum): ({avgrec.min()}, {avgrec.max():.2f}, {avgrec.sum():.2f})':<80}")
	#for tk, weighted_usrInterest in zip(topK_recommended_tokens[::-1], topk_matches_avgRecSys[::-1]):
	for tk, weighted_usrInterest in zip(topK_recommended_tokens[::-1], topK_recommended_tks_weighted_user_interest[::-1]):
		print(f"{tk:<20}{weighted_usrInterest:^{60}.{3}f}")
	print()
	print(f"Implicit Feedback Recommendation: {f'Unique Users: {nUsers} vs. Tokenzied word Items: {nItems}'}".center(150,'-'))

	plot_tokens_distribution(sp_mat_rf, df_usr_tk, query_vector, avgrec, BoWs, norm_sp=normalize_sp_mtrx, topK=topK)

def get_cosine_similarity(QU, RF, query_phrase, query_token, users_tokens_df, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original"

	print(f"Getting Cosine Similarity: QUERY_VEC: {QU.reshape(1, -1).shape} vs. REFERENCE_SPARSE_MATRIX: {RF.shape}".center(110, " ")) # QU: (nItems, ) => (1, nItems) | RF: (nUsers, nItems) 
	st_t = time.time()
	cos_sim = cosine_similarity(QU.reshape(1, -1), RF) # qu_ (nItems,) => (1, nItems) -> cos: (1, nUsers)

	print(f"<> Plotting Cosine Similarity {cos_sim.shape} | Raw Query Phrase: {query_phrase} | Query Token(s) : {query_token}")	
	
	alphas = np.ones_like(cos_sim.flatten())
	scales = 100*np.ones_like(cos_sim.flatten())
	for i, v in np.ndenumerate(cos_sim.flatten()):
		if v==0:
			alphas[i] = 0.05
			scales[i] = 5

	f, ax = plt.subplots()
	ax.scatter(	x=np.arange(len(cos_sim.flatten())), 
							y=cos_sim.flatten(), 
							facecolor="g", 
							s=scales, 
							edgecolors='w',
							alpha=alphas,
							marker=".",
						)
	
	N=5
	if np.count_nonzero(cos_sim.flatten()) < N:
		N = np.count_nonzero(cos_sim.flatten())
	
	topN_max_cosine_user_idx = np.argsort(cos_sim.flatten())[-N:]
	topN_max_cosine = cos_sim.flatten()[topN_max_cosine_user_idx]
	topN_max_cosine_user_ip = users_tokens_df.loc[topN_max_cosine_user_idx, 'user_ip'].values.tolist()
	ax.scatter(x=topN_max_cosine_user_idx, y=topN_max_cosine, facecolor='none', marker="o", edgecolors="r", s=100)
	#ax.set_xlabel('Users', fontsize=10)
	ax.set_ylabel('Cosine Similarity', fontsize=10.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
	plt.xticks(	[i for i in range(len(users_tokens_df["user_ip"])) if i%MODULE==0], 
							[f"{users_tokens_df.loc[i, 'user_ip']}" for i in range(len(users_tokens_df["user_ip"])) if i%MODULE==0],
							rotation=90,
							fontsize=10.0,
							)

	#ax.grid(linestyle="dashed", linewidth=1.5, alpha=0.5)
	ax.grid(which = "major", linewidth = 1)
	ax.grid(which = "minor", linewidth = 0.2)
	ax.minorticks_on()
	ax.set_axisbelow(True)
	ax.margins(1e-3, 3e-2)
	ax.spines[['top', 'right']].set_visible(False)

	plt.text(	x=0.5, 
						y=0.94, 
						s=f"Raw Input Query Phrase: {query_phrase}", 
						fontsize=10.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.text(	x=0.5,
						y=0.91,
						s=f"Query (1 x nItems): {QU.reshape(1, -1).shape} & {sp_type} Sparse Matrix (nUsers x nItems): {RF.shape} : Cosine Similarity (1 x nUsers): {cos_sim.shape}",
						fontsize=9.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.text(	x=0.5,
						y=0.88,
						s=f"{N}-Max cosine(s): {topN_max_cosine_user_ip} : {topN_max_cosine}",
						fontsize=8.5,
						ha="center", 
						color="r",
						transform=f.transFigure,
					)
	
	plt.subplots_adjust(top=0.86, wspace=0.1)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_cos_sim_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)
	print(f"Elapsed_t: {time.time()-st_t:.2f} s".center(100, " "))

	return cos_sim

def plot_tokens_by_max(cos_sim, sp_mtrx, users_tokens_df, bow, norm_sp=False):
	"""
	# retrieve user with max cosine:
	max_cosine = cos_sim.max()
	max_cosine_user_idx = np.argmax(cos_sim)
	max_cosine_user_ip = users_tokens_df.loc[np.argmax(cos_sim), 'user_ip']
	print(f"Sparse Matrix (normalized: {norm_sp}) (nUsers x nItems): {sp_mtrx.shape}\n"
				f"Cosine Similarity (1 x nUsers): {cos_sim.shape}\n"
				f"max(cosine) = {max_cosine:.3f} @ (userIdx: {max_cosine_user_idx} userIP: {max_cosine_user_ip})\n",
			)
	"""
	N=5
	if np.count_nonzero(cos_sim.flatten()) < N:
		N = np.count_nonzero(cos_sim.flatten())

	print(f"Getting tokens of {N} users with max cosine similarity".center(100, "-"))
	topN_max_cosine_user_idx = np.argsort(cos_sim.flatten())[-N:]
	topN_max_cosine = cos_sim.flatten()[topN_max_cosine_user_idx]
	topN_max_cosine_user_ip = users_tokens_df.loc[topN_max_cosine_user_idx, 'user_ip'].values.tolist()

	for _, usr in enumerate(topN_max_cosine_user_ip):
		tokens_names, tokens_values_total, tokens_values_separated = get_tokens_byUSR(sp_mtrx, users_tokens_df, bow, user=usr)
		plot_tokens_by(	userIP=usr, 
										tks_name=tokens_names, 
										tks_value_all=tokens_values_total, 
										tks_value_separated=tokens_values_separated, 
										topTKs=15,
										norm_sp=norm_sp,
									)

	print(f"DONE".center(100, "-"))

def plot_tokens_by(userIP, tks_name, tks_value_all, tks_value_separated, topTKs=50, norm_sp=False):
	sp_type = "Normalized" if norm_sp else "Original"
	nTokens_orig = len(tks_name)

	if len(tks_name) > topTKs:
		tks_name = tks_name[:topTKs]
		tks_value_all = tks_value_all[:topTKs]
		tks_value_separated = [elem[:topTKs] for elem in tks_value_separated]

	nTokens = len(tks_name)
	print(f"<> Plotting top-{nTokens} out of |ALL_TKs = {nTokens_orig}| tokens by user: {userIP}")

	f, ax = plt.subplots()
	ax.barh(tks_name, 
					tks_value_all, 
					#color=clrs,
					color="#0000ff",
					height=0.35,
				)
	ax.tick_params(axis='x', labelrotation=0, labelsize=7.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
	ax.set_xlabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)
	ax.invert_yaxis()  # labels read top-to-botto
	ax.set_title(f'Top-{nTokens} Tokens / |ALL_TKs = {nTokens_orig}| by User: {userIP}', fontsize=11)
	ax.margins(1e-2, 5e-3)
	ax.spines[['top', 'right']].set_visible(False)
	for container in ax.containers:
		ax.bar_label(	container, 
									#rotation=45, # no rotation for barh 
									fontsize=7.0,
									padding=1.5,
									fmt='%.3f', #if norm_sp else '%.2f',
									label_type='edge',
								)
	ax.set_xlim(right=ax.get_xlim()[1]+0.5, auto=True)
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_usr_{userIP}_topTKs{nTokens}_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

	f, ax = plt.subplots()
	lft = np.zeros(len(tks_name))
	qcol_list = ["Search PHRs", "Snippet HWs", "Snippet Appr", "Content HWs", "Content PRTs", "Content Appr",]
	hbars = list()
	for i, v in enumerate( tks_value_separated ):
		print(i, tks_name, v)
		hbar = ax.barh(tks_name, v, color=clrs[i], height=0.4, left=lft, edgecolor='w', lw=0.4, label=f"{qcol_list[i]:<20}w: {w_list[i]:<{10}.{3}f}{[f'{val:.3f}' for val in v]}")
		lft += v
		hbars.append(hbar)
		#print(hbar.datavalues)
	
	ax.tick_params(axis='x', labelrotation=0, labelsize=7.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
	
	ax.set_xlabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)
	ax.invert_yaxis()  # labels read top-to-botto
	ax.set_title(f'Top-{nTokens} Tokens / |ALL_TKs = {nTokens_orig}| by User: {userIP}', fontsize=11)
	ax.margins(1e-2, 5e-3)
	ax.spines[['top', 'right']].set_visible(False)
	ax.legend(loc="best", fontsize=8.0)
	
	for bar in hbars:
		filtered_lbls = [f"{v:.1f}" if v>=0.8 else "" for v in bar.datavalues]
		ax.bar_label(bar, labels=filtered_lbls, label_type='center', rotation=0.0, fontsize=6.0)
	
	ax.set_xlim(right=ax.get_xlim()[1]+0.5, auto=True)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_usr_{userIP}_topTKs{nTokens}_seperated_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_users_by(token, usrs_name, usrs_value_all, usrs_value_separated, topUSRs=50, norm_sp=False):
	sp_type = "Normalized" if norm_sp else "Original"
	
	nUsers_orig = len(usrs_name)

	if len(usrs_name) > topUSRs:
		usrs_name = usrs_name[:topUSRs]
		usrs_value_all = usrs_value_all[:topUSRs]
		usrs_value_separated = [elem[:topUSRs] for elem in usrs_value_separated]

	nUsers = len(usrs_name)

	print(f"Plotting top-{nUsers} users out of |ALL_USRs = {nUsers_orig} | token: {token}".center(110, "-"))

	f, ax = plt.subplots()
	ax.bar(usrs_name, usrs_value_all, color="#a6111122", width=0.4)
	#ax.set_xlabel('Tokens', rotation=90)
	ax.tick_params(axis='x', labelrotation=90, labelsize=8.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=8.0)
	ax.set_ylabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)
	ax.set_title(f'Top-{nUsers} User(s) / |ALL_USRs = {nUsers_orig}| for token: {token}', fontsize=11)
	ax.margins(1e-2, 3e-2)
	ax.spines[['top', 'right']].set_visible(False)
	for container in ax.containers:
		ax.bar_label(	container, 
									rotation=0, 
									fontsize=6.5,
									padding=1.5,
									fmt='%.3f',# if norm_sp else '%.2f', 
									label_type='edge',
									)
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_topUSRs{nUsers}_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)
	print(f"Separated Values".center(100, " "))
	f, ax = plt.subplots()
	btn = np.zeros(len(usrs_name))
	qcol_list = ["Search PHRs", "Snippet HWs", "Snippet Appr", "Content HWs", "Content PRTs", "Content Appr",]
	vbars = list()
	for i, v in enumerate( usrs_value_separated ):
		print(i, usrs_name, v)
		vbar = ax.bar(usrs_name, v, color=clrs[i], width=0.4, bottom=btn, edgecolor='w', lw=0.4, label=f"{qcol_list[i]:<20}w: {w_list[i]:<{10}.{3}f}{[f'{val:.3f}' for val in v]}")
		btn += v
		vbars.append(vbar)
	
	ax.tick_params(axis='x', labelrotation=90, labelsize=8.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=8.0)
	ax.set_ylabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)	
	ax.set_title(f'Top-{nUsers} User(s) / |ALL_USRs = {nUsers_orig}| for token: {token}', fontsize=11)
	ax.margins(1e-2, 5e-3)
	ax.spines[['top', 'right']].set_visible(False)
	ax.legend(loc="best", fontsize=8.0)

	for b in vbars:
		filtered_lbls = [f"{v:.1f}" if v>=1.0 else "" for v in b.datavalues]
		ax.bar_label(b, labels=filtered_lbls, label_type='center', rotation=0.0, fontsize=8.0)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_topUSRs{nUsers}_separated_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_usersInterest_by(token, sp_mtrx, users_tokens_df, bow, norm_sp=False):
	matrix = sp_mtrx.toarray()
	sp_type = "Normalized" if matrix.max() == 1.0 else "Original" 
	tkIdx = bow.get(token)
	usersInt = matrix[:, tkIdx]
	alphas = np.ones_like(usersInt)
	scales = 100*np.ones_like(usersInt)
	for i, v in np.ndenumerate(usersInt):
		if v==0:
			alphas[i] = 0.05
			scales[i] = 5

	f, ax = plt.subplots()
	ax.scatter(	x=np.arange(len(usersInt)), 
							y=usersInt, 
							facecolor="b", 
							s=scales,
							edgecolors='w',
							alpha=alphas,
							marker=".",
						)
	
	N=5
	if np.count_nonzero(usersInt) < N:
		N = np.count_nonzero(usersInt)

	topN_max_usrInt_user_idx = np.argsort(usersInt)[-N:]
	topN_max_user_interest = usersInt[topN_max_usrInt_user_idx]
	topN_max_usrInt_user_ip = users_tokens_df.loc[topN_max_usrInt_user_idx, 'user_ip'].values.tolist()
	ax.scatter(x=topN_max_usrInt_user_idx, y=topN_max_user_interest, facecolor='none', marker="o", edgecolors="r", s=100)

	#ax.set_xlabel('Users', fontsize=10)
	ax.set_ylabel('UserInterest [Implicit Feedback]', fontsize=10)
	"""
	ax.set_title(	f"{sp_type} Sparse Matrix (nUsers x nItems): {matrix.shape}\n"
								f"Users Interests by (token: {token} idx: {tkIdx})\n"
								f"max(UserInterest): {usersInt.max():.3f}@(userIdx: {np.argmax(usersInt)} userIP: {users_tokens_df.loc[np.argmax(usersInt), 'user_ip']})", 
								fontsize=10,
							)
	"""
	ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
	plt.xticks(	[i for i in range(len(users_tokens_df["user_ip"])) if i%MODULE==0], 
							[f"{users_tokens_df.loc[i, 'user_ip']}" for i in range(len(users_tokens_df["user_ip"])) if i%MODULE==0],
							rotation=90,
							fontsize=10.0,
							)

	#ax.grid(linestyle="dashed", linewidth=1.5, alpha=0.5)
	ax.grid(which = "major", linewidth = 1)
	ax.grid(which = "minor", linewidth = 0.2)
	ax.minorticks_on()
	ax.set_axisbelow(True)
	ax.margins(1e-3, 3e-2)
	ax.spines[['top', 'right']].set_visible(False)

	plt.text(	x=0.5, 
						y=0.94, 
						s=f"Raw Input Query Phrase: {args.qphrase}", 
						fontsize=10.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.text(	x=0.5,
						y=0.91,
						s=f"{sp_type} Sparse Matrix (nUsers x nItems): {matrix.shape} | Users Interests by (token: {token} @ idx: {tkIdx})",
						fontsize=9.0, 
						ha="center", 
						transform=f.transFigure,
					)

	plt.text(	x=0.5,
						y=0.88,
						s=f"{N}-Max UsersInterest(s): {topN_max_usrInt_user_ip} : {topN_max_user_interest}",
						fontsize=9.0, 
						ha="center", 
						color="r",
						transform=f.transFigure,
					)

	plt.subplots_adjust(top=0.86, wspace=0.1)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_usersInterest_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_heatmap_sparse(sp_mtrx, df_usr_tk, bow, norm_sp=False, ifb_log10=False):
	name_="sparse_matrix_user_vs_token"
	sp_type = "Normalized" if norm_sp else "Original"

	print(f"{f'{sp_type} Sparse Matrix {sp_mtrx.shape}'.center(100,'-')}")

	print(f"<> Non-zeros vals: {sp_mtrx.data}")# Viewing stored data (not the zero items)
	print(f"<> |Non-zero cells|: {sp_mtrx.count_nonzero()}") # Counting nonzeros
	mtrx = sp_mtrx.toarray() # to numpy array

	####################################################
	if ifb_log10:
		mtrx = np.log10(mtrx, out=np.zeros_like(mtrx), where=(mtrx!=0))
	####################################################

	max_pose = np.unravel_index(mtrx.argmax(), mtrx.shape)
	print(mtrx.max(), 
				max_pose,
				mtrx[max_pose],
				df_usr_tk['user_ip'].iloc[max_pose[0]],
				[k for k, v in bow.items() if v==max_pose[1]],

				#np.argmax(mtrx, axis=0).shape, 
				#np.argmax(mtrx, axis=0)[-10:], 
				#np.argmax(mtrx, axis=1).shape, 
				#np.argmax(mtrx, axis=1)[-10:],
			) # axis=0-> col, axis=1 -> row
	
	f, ax = plt.subplots()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	im = ax.imshow(mtrx, 
								#cmap="viridis",#"magma", # https://matplotlib.org/stable/tutorials/colors/colormaps.html
								#cmap="gist_yarg",
								cmap="gist_gray",
								)
	cbar = ax.figure.colorbar(im,
														ax=ax,
														orientation="vertical",
														cax=cax,
														#ticks=[0.0, 0.5, 1.0],
														)
	cbar.ax.tick_params(labelsize=7.5)
	cbar.set_label(	label="Implicit Feedback ( $Log_{{10}}$ )" if ifb_log10 else "Implicit Feedback",
									size=9.0, 
									#weight='bold',
									style='italic',
								)
	ax.set_xlabel(f"Token Indices", fontsize=10.0)
	ax.set_ylabel(f"{'user indeces'.capitalize()}\n"
								f"{df_usr_tk['user_ip'].iloc[-1]}$\longleftarrow${df_usr_tk['user_ip'].iloc[0]}", fontsize=10.0)

	#ax.set_yticks([])
	#ax.set_xticks([])
	#ax.xaxis.tick_top() # put ticks on the upper part
	ax.tick_params(axis='x', labelrotation=0, labelsize=8.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=8.0)

	plt.text(x=0.5, y=0.94, s=f"{sp_type} Sparse Matrix Heatmap (nUsers, nItems): {sp_mtrx.shape}", fontsize=10.0, ha="center", transform=f.transFigure)
	plt.text(	x=0.5, 
						y=0.88, 
						s=f"|non-zeros|: {sp_mtrx.count_nonzero()} / |tot_elem|: {sp_mtrx.shape[0]*sp_mtrx.shape[1]}\n"
							f"max: {mtrx.max():{2}.{1}f}@{max_pose}: (User: {df_usr_tk['user_ip'].iloc[max_pose[0]]}, Token: {[k for k, v in bow.items() if v==max_pose[1]]})", 
						fontsize=9.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.subplots_adjust(top=0.8, wspace=0.3)

	plt.savefig(os.path.join( RES_DIR, f"heatmap_{sp_type}_log10_{ifb_log10}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)
	print(f"Done".center(70, "-"))

def plot_tokens_distribution(sparseMat, users_tokens_df, queryVec, recSysVec, bow, norm_sp=False, topK=5):
	sp_type = "Normalized" if norm_sp else "Original"

	print(f"<> Plotting Tokens Distribution in {sp_type} Sparse Matrix (nUsers, nItems): {sparseMat.shape}")
	sparse_df = pd.DataFrame(sparseMat.toarray(), index=users_tokens_df["user_ip"])
	#sparse_df.columns = sparse_df.columns.map(str) # convert int col -> str col
	sparse_df = sparse_df.replace(0.0, np.nan) # 0.0 -> None: Matplotlib won't plot NaN values.
	#print(f">> queryVec: {queryVec.shape} | recSysVec: {recSysVec.shape}")

	if len(recSysVec.shape) > 1:
		recSysVec = recSysVec.flatten()	

	if len(queryVec.shape) > 1:
		queryVec = queryVec.flatten()
	
	qu_indices = np.nonzero(queryVec)[0]

	all_recommended_tks = [k for idx in recSysVec.flatten().argsort()[-50:] for k, v in bow.items() if (idx not in qu_indices and v==idx)]
	#print(f"ALL (15): {len(all_recommended_tks)} : {all_recommended_tks[-15:]}")
	
	topK_recommended_tokens = all_recommended_tks[-(topK+0):]
	#print(f"top-{topK} recommended Tokens: {len(topK_recommended_tokens)} : {topK_recommended_tokens}")
	topK_recommended_tks_weighted_user_interest = [ recSysVec.flatten()[bow.get(vTKs)] for iTKs, vTKs in enumerate(topK_recommended_tokens)]
	#print(f"top-{topK} recommended Tokens weighted user interests: {len(topK_recommended_tks_weighted_user_interest)} : {topK_recommended_tks_weighted_user_interest}")

	#recSysVec_indices = recSysVec.argsort()[-(topK+0):]
	recSysVec_indices = np.array([bow.get(vTKs) for iTKs, vTKs in enumerate(topK_recommended_tokens)])

	plt.rcParams["figure.subplot.right"] = 0.8
	quTksLegends = []
	f, ax = plt.subplots()	
	for ix, col in np.ndenumerate(qu_indices):
		sc1 = ax.scatter(	x=sparse_df.index, 
											y=sparse_df[col], 
											label=f"{[k for k, v in bow.items() if v==col]} | {col}",
											marker="H",
											s=260,
											facecolor="none", 
											edgecolors=clrs[::-1][int(ix[0])],
										)
		quTksLegends.append(sc1)

	recLegends = []
	for ix, col in np.ndenumerate(np.flip(recSysVec_indices)):
		sc2 = ax.scatter(	x=sparse_df.index, 
											y=sparse_df[col], 
											label=f"{[k for k, v in bow.items() if v==col]} | {col} | {recSysVec[col]:.3f}",
											marker=".",
											s=900*recSysVec[col],
											alpha=1/((2*int(ix[0]))+1),
											#cmap='magma',
											#c=clrs[int(ix[0])],
											edgecolors=clrs[int(ix[0])],
											facecolor="none",
										)
		recLegends.append(sc2)

	leg1 = plt.legend(handles=quTksLegends, loc=(1.03, 0.8), fontsize=9.0, title=f"Lemmatized Query Phrase(s)\nToken | vbIdx", fancybox=True, shadow=True,)
	plt.setp(leg1.get_title(), multialignment='center', fontsize=9.0)
	plt.gca().add_artist(leg1)
	leg2 = plt.legend(handles=recLegends, loc=(1.03, 0.0), fontsize=8.0, title=f"Top-{topK} Recommended Results\nToken | vbIdx | wightedUserInterest", fancybox=True, shadow=True,)
	plt.setp(leg2.get_title(), multialignment='center', fontsize=9.0)
	
	ax.spines[['top', 'right']].set_visible(False)
	ax.margins(1e-3, 5e-2)
	plt.xticks(	[i for i in range(len(users_tokens_df["user_ip"])) if i%MODULE==0], 
							[f"{users_tokens_df.loc[i, 'user_ip']}" for i in range(len(users_tokens_df["user_ip"])) if i%MODULE==0],
							rotation=90,
							fontsize=10.0,
							)

	plt.yticks(fontsize=10.0)
	plt.ylabel(f"Cell Value in {sp_type} Sparse Matrix", fontsize=10)
	#plt.xlabel(f"Users", fontsize=11)
	plt.title(f"Token(s) Distribution | {sp_type} Sparse Matrix (nUsers, nItems): {sparse_df.shape}\n"
						f"Raw Input Query Phrase: {args.qphrase}\n"
						f"Top-{topK}: {[k for idx in recSysVec_indices for k, v in bow.items() if v==idx][::-1]}", 
						fontsize=9.0,
					)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_{args.lmMethod}_top{topK}_recs_{sp_type}_SP.png" ), bbox_inches='tight')

	plt.clf()
	plt.close(f)
	print(">> Done!")

def main():
	df_raw = load_df(infile=args.inputDF)
	run_RecSys(df_inp=df_raw, qu_phrase=args.qphrase, normalize_sp_mtrx=args.normSP, topK=args.topTKs)
	#return

def practice(topK=5):
	nUsers = 5
	nItems = 11
	cos = np.random.rand(nUsers).reshape(1, -1)
	usr_itm = np.random.randint(100, size=(nUsers, nItems))
	avgrec = np.zeros((1,nItems))
	print(f"> avgrec{avgrec.shape}:\n{avgrec}")
	print()
	
	print(f"> cos{cos.shape}:\n{cos}")
	print()

	print(f"> user-item{usr_itm.shape}:\n{usr_itm}")
	print("#"*100)

	for iUser in range(nUsers):
		print(f"USER: {iUser}")
		userInterest = usr_itm[iUser, :]
		print(f"<> userInterest{userInterest.shape}:\n{userInterest}")
		userInterest = userInterest / np.linalg.norm(userInterest)
		print(f"<> userInterest_norm{userInterest.shape}:\n{userInterest}")
		print(f"cos[{iUser}]: {cos[0, iUser]}")
		print(f"<> avgrec (B4):{avgrec.shape}\n{avgrec}")
		avgrec = avgrec + cos[0, iUser] * userInterest
		print(f"<> avgrec (After):{avgrec.shape}\n{avgrec}")
		print()

	print(f"-"*100)
	avgrec = avgrec / np.sum(cos)
	print(f"avgrec:{avgrec.shape}\n{avgrec}")
	topk_matches_idx_avgRecSys = avgrec.flatten().argsort()
	topk_matches_avgRecSys = np.sort(avgrec.flatten())

	print(f"top-{topK} idx: {topk_matches_idx_avgRecSys}\ntop-{topK} res: {topk_matches_avgRecSys}")

if __name__ == '__main__':
	#os.system("clear")
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120, " "))
	main()
	#practice()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120, " "))