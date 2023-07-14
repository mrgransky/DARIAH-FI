from utils import *
from tokenizer_utils import *

lemmatizer_methods = {"nltk": nltk_lemmatizer,
											"spacy": spacy_tokenizer,
											"trankit": trankit_lemmatizer,
											"stanza": stanza_lemmatizer,
											}

def get_raw_sqp(phrase_list):
	assert len(phrase_list) == 1, f"<!> Wrong length for {phrase_list}, must be = 1!"
	phrase = phrase_list[0]
	return phrase

def get_lemmatized_sqp(qu_list, lm: str="stanza"):
	# qu_list = ['some word in this format with always length 1']
	#print(len(qu_list), qu_list)
	assert len(qu_list) == 1, f"query list length MUST be 1, it is now {len(qu_list)}!!"
	return lemmatizer_methods.get(lm)(qu_list[0])

def get_raw_snHWs(search_results_list):
	#hw_snippets = [sn.get("terms") for sn in search_results_list if ( sn.get("terms") and len(sn.get("terms")) > 0 )] # [["A"], ["B"], ["C"]]
	hw_snippets = [w for sn in search_results_list if ( sn.get("terms") and len(sn.get("terms")) > 0 ) for w in sn.get("terms")] # ["A", "B", "C"]
	return hw_snippets

def get_lemmatized_snHWs(results, lm: str="stanza"):
	return [tklm for el in results if ( el and (lemmas:=lemmatizer_methods.get(lm)(el)) ) for tklm in lemmas if tklm ]

def get_raw_cntHWs(cnt_dict):
	return cnt_dict.get("highlighted_term")

def get_lemmatized_cntHWs(results, lm: str="stanza"):
	# print(results)
	return [tklm for el in results if ( el and (lemmas:=lemmatizer_methods.get(lm)(el)) ) for tklm in lemmas if tklm ]
	# if results:
	# 	return [tklm for el in results if ( el and (lemmas:=lemmatizer_methods.get(lm)(el)) ) for tklm in lemmas if tklm ]

def get_raw_cntPTs(cnt_dict):
	return cnt_dict.get("parsed_term")

def get_lemmatized_cntPTs(results, lm: str="stanza"):
	# print(results)
	return [tklm for el in results if ( el and (lemmas:=lemmatizer_methods.get(lm)(el)) ) for tklm in lemmas if tklm ]
	# if results:
	# 	return [tklm for el in results if ( el and (lemmas:=lemmatizer_methods.get(lm)(el)) ) for tklm in lemmas if tklm ]

def get_raw_sn(results):
	#snippets_list = [sn.get("textHighlights").get("text") for sn in results if sn.get("textHighlights").get("text") ] # [["sentA"], ["sentB"], ["sentC"]]
	snippets_list = [sent for sn in results if sn.get("textHighlights").get("text") for sent in sn.get("textHighlights").get("text")] # ["sentA", "sentB", "sentC"]
	return snippets_list

def get_lemmatized_sn(results, lm: str="stanza"):
	return [tklm for el in results if ( el and (lemmas:=lemmatizer_methods.get(lm)(el)) ) for tklm in lemmas if tklm ]

def get_raw_snTEXTs(results):
	#snippets_list = [sn.get("textHighlights").get("text") for sn in results if sn.get("textHighlights").get("text") ] # [["sentA"], ["sentB"], ["sentC"]]
	snippets_list = [sent for sn in results if sn.get("textHighlights").get("text") for sent in sn.get("textHighlights").get("text")] # ["sentA", "sentB", "sentC"]
	return ' '.join(snippets_list)

def get_raw_cnt(cnt_dict):
	return cnt_dict.get("text")

def get_lemmatized_cnt(sentences: str, lm: str="stanza"):
	return lemmatizer_methods.get(lm)(sentences)

def get_cBoWs(dframe: pd.DataFrame, fprefix: str="df_concat", lm: str="stanza"):
	print(f"{f'Bag-of-Words [ Complete: {userName} ]'.center(110, '-')}")

	print(f"{f'Extracting texts search query phrases':<50}", end="")
	st_t = time.time()
	dframe["query_phrase_raw_text"] = dframe["search_query_phrase"].map(get_raw_sqp, na_action="ignore")
	print(f"Elapsed_t: {time.time()-st_t:.3f} s")
	
	print(f"{f'Extracting texts collection query phrases':<50}", end="")
	st_t = time.time()
	dframe["collection_query_phrase_raw_text"] = dframe["collection_query_phrase"].map(get_raw_sqp, na_action="ignore")
	print(f"Elapsed_t: {time.time()-st_t:.3f} s")

	print(f"{f'Extracting texts clipping query phrases':<50}", end="")
	st_t = time.time()
	dframe["clipping_query_phrase_raw_text"] = dframe["clipping_query_phrase"].map(get_raw_sqp, na_action="ignore")
	print(f"Elapsed_t: {time.time()-st_t:.3f} s")

	print(f"{f'Extracting texts newspaper content':<50}", end="")
	st_t = time.time()
	dframe['ocr_raw_text'] = dframe["nwp_content_results"].map(get_raw_cnt, na_action='ignore')
	print(f"Elapsed_t: {time.time()-st_t:.3f} s")
	
	print(f"{f'Extracting raw texts snippets':<50}", end="")
	st_t = time.time()
	dframe['snippet_raw_text'] = dframe["search_results"].map(get_raw_snTEXTs, na_action='ignore')
	print(f"Elapsed_t: {time.time()-st_t:.3f} s")

	# print(dframe.info())
	# print(dframe[["user_ip", "query_phrase_raw_text", "snippet_raw_text", "ocr_raw_text"]].tail(60))
	# print(f"<>"*50)
	# return

	users_list = list()
	raw_texts_list = list()
	
	for n, g in dframe.groupby("user_ip"):
		users_list.append(n)
		lq = [ phrases for phrases in g[g["query_phrase_raw_text"].notnull()]["query_phrase_raw_text"].values.tolist() if len(phrases) > 0 ] # ["global warming", "econimic crisis", "", ]
		lcol = [phrases for phrases in g[g["collection_query_phrase_raw_text"].notnull()]["collection_query_phrase_raw_text"].values.tolist() if len(phrases) > 0] # ["independence day", "suomen pankki", "helsingin pörssi", ...]
		lclp = [phrases for phrases in g[g["clipping_query_phrase_raw_text"].notnull()]["clipping_query_phrase_raw_text"].values.tolist() if len(phrases) > 0] # ["", "", "", ...]

		ls = [ sentences for sentences in g[g["snippet_raw_text"].notnull()]["snippet_raw_text"].values.tolist() if len(sentences) > 0 ] # ["", "", "", ...]
		lc = [ sentences for sentences in g[g["ocr_raw_text"].notnull()]["ocr_raw_text"].values.tolist() if len(sentences) > 0 ] # ["", "", "", ...]

		ltot = lq + lcol + lclp + ls + lc
		raw_texts_list.append( ltot )

	del dframe
	gc.collect()

	print(len(users_list), len(raw_texts_list), type(raw_texts_list), any(elem is None for elem in raw_texts_list))
	print(f">> creating raw_docs_list", end=" ")
	raw_docs_list = [subitem for itm in raw_texts_list if ( itm is not None and len(itm) > 0 ) for subitem in itm if ( re.search(r'[a-zA-Z|ÄäÖöÅåüÜúùßẞàñéèíóò]', subitem) and re.search(r"\S", subitem) and re.search(r"\D", subitem) and max([len(el) for el in subitem.split()])>2  and re.search(r"\b(?=\D)\w{3,}\b", subitem)) ]

	del raw_texts_list
	gc.collect()

	print(len(raw_docs_list), type(raw_docs_list), any(elem is None for elem in raw_docs_list))

	raw_docs_list = list(set(raw_docs_list))
	print(f"<<!>> unique phrases: {len(raw_docs_list)}")

	tfidf_vec_fpath = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{lm}_tfidf_vectorizer_large.gz")
	tfidf_rf_matrix_fpath = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{lm}_tfidf_matrix_RF_large.gz")
	
	try:
		tfidf_matrix_rf = load_pickle(fpath=tfidf_rf_matrix_fpath)
		tfidf_vec = load_pickle(fpath=tfidf_vec_fpath)
	# except Exception as e:
	# 	logging.exception(e)
	except:
		print(f"Training TFIDF vector for {len(raw_docs_list)} raw words/phrases/sentences, might take a while...".center(150, " "))
		st_t = time.time()
		# Initialize TFIDF # not time consuming...
		tfidf_vec = TfidfVectorizer(tokenizer=lemmatizer_methods.get(lm),)
		# Fit TFIDF # TIME CONSUMING:
		try:
			tfidf_matrix_rf = tfidf_vec.fit_transform(raw_documents=raw_docs_list)
		except Exception as e:
			print(f"<!> TfidfVectorizer Error: {e}")
			# logging.exception(e)
		#tfidf_matrix_rf = np.random.choice(10_000, 10_000)
		del raw_docs_list
		gc.collect()
		save_pickle(pkl=tfidf_vec, fname=tfidf_vec_fpath)
		save_pickle(pkl=tfidf_matrix_rf, fname=tfidf_rf_matrix_fpath)
		save_vocab(	vb=dict( sorted( tfidf_vec.vocabulary_.items(), key=lambda x:x[1], reverse=False ) ), 
								fname=os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{lm}_{len(tfidf_vec.vocabulary_)}_vocabs.json"),
							)
		print(f"Elapsed_t: {time.time()-st_t:.2f} s".center(80, " "))

	feat_names = tfidf_vec.get_feature_names_out()
	BOWs = dict( sorted( tfidf_vec.vocabulary_.items(), key=lambda x:x[1], reverse=False ) ) # ascending
	# example:
	# vb = {"example": 0, "is": 1, "simple": 2, "this": 3}
	#	   		example   is         simple     this	
	# 0  		0.377964  0.377964   0.377964   0.377964

	print(f"Features: {feat_names.shape} | {type(feat_names)} | BoWs: {len(BOWs)} | {type(BOWs)}")
	print(f"TFIDF REF matrix: {tfidf_matrix_rf.shape}")
	assert len(BOWs) == tfidf_matrix_rf.shape[1], f"size of vocabs: {len(BoWs)} != tfidf_matrix_rf: {tfidf_matrix_rf.shape[1]}"
	print(f"{f'Bag-of-Words [ Complete: {userName} ]'.center(110, '-')}")
	return BOWs

def get_BoWs(dframe: pd.DataFrame, fprefix: str="df_concat", lm: str="stanza"):
	print(f"{f'Bag-of-Words [{userName}]'.center(110, '-')}")

	print(f">> Extracting texts from query phrases...")
	st_t = time.time()
	dframe["query_phrase_raw_text"] = dframe["search_query_phrase"].map(get_raw_sqp, na_action="ignore")
	print(f"\tElapsed_t: {time.time()-st_t:.3f} s")

	print(f">> Extracting texts from collection query phrases...")
	st_t = time.time()
	dframe["collection_query_phrase_raw_text"] = dframe["collection_query_phrase"].map(get_raw_sqp, na_action="ignore")
	print(f"\tElapsed_t: {time.time()-st_t:.3f} s")

	print(f">> Extracting texts from clipping query phrases...")
	st_t = time.time()
	dframe["clipping_query_phrase_raw_text"] = dframe["clipping_query_phrase"].map(get_raw_sqp, na_action="ignore")
	print(f"\tElapsed_t: {time.time()-st_t:.3f} s")

	users_list = list()
	raw_texts_list = list()
	
	for n, g in dframe.groupby("user_ip"):
		users_list.append(n)
		lq = [phrases for phrases in g[g["query_phrase_raw_text"].notnull()]["query_phrase_raw_text"].values.tolist() if len(phrases) > 0]
		lcol = [phrases for phrases in g[g["collection_query_phrase_raw_text"].notnull()]["collection_query_phrase_raw_text"].values.tolist() if len(phrases) > 0]
		lclp = [phrases for phrases in g[g["clipping_query_phrase_raw_text"].notnull()]["clipping_query_phrase_raw_text"].values.tolist() if len(phrases) > 0]
		ltot = lq + lcol + lclp
		#print(n, ltot)
		raw_texts_list.append( ltot )

	print(len(users_list), len(raw_texts_list), type(raw_texts_list), any(elem is None for elem in raw_texts_list))

	raw_docs_list = [subitem for itm in raw_texts_list if ( itm is not None and len(itm) > 0 ) for subitem in itm if (re.search(r"\S", subitem) and not re.search(r"\d", subitem)) ]
	print(len(raw_docs_list), type(raw_docs_list), any(elem is None for elem in raw_docs_list))

	print(f"<<!>> unique phrases: {len(list(set(raw_docs_list)))}")
	raw_docs_list = list(set(raw_docs_list))
	"""
	with open("raw_list_words.json", "w") as fw:
		json.dump(Counter(raw_docs_list), fw, indent=4, ensure_ascii=False)
	"""
	tfidf_vec_fpath = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{lm}_tfidf_vectorizer.gz")
	tfidf_rf_matrix_fpath = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{lm}_tfidf_matrix_RF.gz")

	if not os.path.exists(tfidf_vec_fpath):
		print(f"Training TFIDF vector for {len(raw_docs_list)} raw words/phrases, might take a while...".center(110, " "))
		st_t = time.time()

		# Fit TFIDF # not time consuming...
		tfidf_vec = TfidfVectorizer(#min_df=5,
															#ngram_range=(1, 2),
															tokenizer=lemmatizer_methods.get(lm),
															#stop_words=UNIQUE_STOPWORDS,
															)

		tfidf_matrix_rf = tfidf_vec.fit_transform(raw_documents=raw_docs_list)
		#tfidf_matrix_rf = np.random.choice(10_000, 10_000)

		save_pickle(pkl=tfidf_vec, fname=tfidf_vec_fpath)
		save_pickle(pkl=tfidf_matrix_rf, fname=tfidf_rf_matrix_fpath)
		save_vocab(	vb=dict( sorted( tfidf_vec.vocabulary_.items(), key=lambda x:x[1], reverse=False ) ), 
								fname=os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{lm}_{len(tfidf_vec.vocabulary_)}_vocabs.json"),
							)
		print(f"Elapsed_t: {time.time()-st_t:.2f} s".center(100, " "))
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