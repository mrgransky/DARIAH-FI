from utils import *
from nlp_utils import *

parser = argparse.ArgumentParser(	description='User-Item Recommendation system developed based on National Library of Finland (NLF) dataset', 
																	prog='RecSys USER-TOKEN', 
																	epilog='Developed by Farid Alijani',
																)

parser.add_argument('--dsPath', default=dataset_path, type=str) # smallest
parser.add_argument('--qphrase', default="juha sipilä", type=str)
parser.add_argument('--lmMethod', default="nltk", type=str)
parser.add_argument('--normSP', default=False, type=bool)
parser.add_argument('--topTKs', default=5, type=int)
args = parser.parse_args()
# how to run:
# python RecSys_usr_token.py --inputDF ~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

fprefix = "df_concat"
#RES_DIR = make_result_dir(infile=fprefix)
make_folder(folder_name=dfs_path)
MODULE=60

# list of all weights:
weightQueryAppearance:float = 1.0				# suggested by Jakko: 1.0
weightSnippetHWAppearance:float = 0.4		# suggested by Jakko: 0.2
weightSnippetAppearance:float = 0.2			# suggested by Jakko: 0.2
weightContentHWAppearance:float = 0.1		# suggested by Jakko: 0.05
weightContentPTAppearance:float = 0.005	# Did not consider initiially!
weightContentAppearance:float = 0.05 		# suggested by Jakko: 0.05

w_list:List[float] = [weightQueryAppearance, 
											weightSnippetHWAppearance,
											weightSnippetAppearance,
											weightContentHWAppearance,
											weightContentPTAppearance,
											weightContentAppearance,
										]

def sum_tk_apperance_vb(dframe, qcol, wg, vb):
	updated_vb = dict.fromkeys(vb.keys(), 0.0)
	for tk in dframe[qcol]: # [tk1, tk2, …]
		if updated_vb.get(tk) is not None: # check if this token is available in BoWs
			updated_vb[tk] = updated_vb.get(tk) + wg
			#print(tk, wg, updated_vb[tk])
	#print(f"{dframe.user_ip}".center(50, '-'))
	return updated_vb

def sum_all_tokens_appearance_in_vb(dframe, weights: List[float], vb: Dict[str, int]):
	w_qu, w_hw_sn, w_sn, w_hw_cnt, w_pt_cnt, w_cnt = weights
	updated_vocab = dict.fromkeys(vb.keys(), 0.0)
	print(f"{dframe.user_ip}: "
				f"qu: {len(dframe.qu_tokens)}, "
				f"snHW: {len(dframe.snippets_hw_token)}, "
				f"sn: {len(dframe.snippets_token)}, "
				f"cntHW: {len(dframe.nwp_content_hw_token)}, "
				f"cntPT: {len(dframe.nwp_content_pt_token)}, "
				f"cnt: {len(dframe.nwp_content_lemma_all)}".center(130, " ")
			)
	for i, q_tk in enumerate(dframe.qu_tokens): # [qtk1, qtk2, qtk3, ...]
		#print(f"QU[{i}]: {q_tk:<25}w: {w_qu} | vb_exist? {updated_vocab.get(q_tk) is not None} ")
		if updated_vocab.get(q_tk) is not None:
			prev = updated_vocab.get(q_tk)
			curr = prev + w_qu
			print(f"QU[{i}]: {q_tk:<25}w: {w_qu}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[q_tk] = curr
	print('*'*60)
	for i, sn_hw_tk in enumerate(dframe.snippets_hw_token):
		#print(f"snHW[{i}]: {sn_hw_tk:<25}w: {w_hw_sn} | vb_exist? {updated_vocab.get(sn_hw_tk) is not None} ")
		if updated_vocab.get(sn_hw_tk) is not None:
			prev = updated_vocab.get(sn_hw_tk)
			curr = prev + w_hw_sn
			print(f"snHW[{i}]: {sn_hw_tk:<25}w: {w_hw_sn}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[sn_hw_tk] = curr
	print('*'*60)
	for i, sn_tk in enumerate(dframe.snippets_token):
		#print(f"sn[{i}]: {sn_tk:<25}w: {w_sn} | vb_exist? {updated_vocab.get(sn_tk) is not None} ")
		if updated_vocab.get(sn_tk) is not None:
			prev = updated_vocab.get(sn_tk)
			curr = prev + w_sn
			print(f"sn[{i}]: {sn_tk:<25}w: {w_sn}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[sn_tk] = curr
	print('*'*60)
	for i, c_hw_tk in enumerate(dframe.nwp_content_hw_token):
		#print(f"cntHW[{i}]: {c_hw_tk:<25}w: {w_hw_cnt} | vb_exist? {updated_vocab.get(c_hw_tk) is not None} ")
		if updated_vocab.get(c_hw_tk) is not None:
			prev = updated_vocab.get(c_hw_tk)
			curr = prev + w_hw_cnt
			print(f"cntHW[{i}]: {c_hw_tk:<25}w: {w_hw_cnt}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[c_hw_tk] = curr
	print('*'*60)
	for i, c_pt_tk in enumerate(dframe.nwp_content_pt_token):
		#print(f"cntPT[{i}]: {c_pt_tk:<25}w: {w_pt_cnt} | vb_exist? {updated_vocab.get(c_pt_tk) is not None} ")
		if updated_vocab.get(c_pt_tk) is not None:
			prev = updated_vocab.get(c_pt_tk)
			curr = prev + w_pt_cnt
			print(f"cntPT[{i}]: {c_pt_tk:<25}w: {w_pt_cnt}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[c_pt_tk] = curr
	print('*'*60)
	for i, c_tk in enumerate(dframe.nwp_content_lemma_all):
		#print(f"cnt[{i}]: {c_tk:<25}w: {w_cnt} | vb_exist? {updated_vocab.get(c_tk) is not None} ")
		if updated_vocab.get(c_tk) is not None:
			prev = updated_vocab.get(c_tk)
			curr = prev + w_cnt
			print(f"cnt[{i}]: {c_tk:<25}w: {w_cnt}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[c_tk] = curr

	print("#"*150)
	return updated_vocab

def get_newspaper_content(lemmatized_content, vb:Dict[str, int], wg:float=weightContentAppearance):
	updated_vb = dict.fromkeys(vb.keys(), [0, 0])
	# lemmatized_content = [[tk1, tk2, tk3, ...], [tk1, tk2, ...], [tk1, ...], ...]
	#print(f"Found {len(lemmatized_content)} content(s) {type(lemmatized_content)} [[tk1, tk2, tk3, ...], [tk1, tk2, ...], [tk1, ...], ...]".center(130, "-"))
	for ilm, vlm in enumerate( lemmatized_content ): # [[tk1, tk2, tk3, ...], [tk1, tk2, ...], [tk1, ...], ...]
		new_boosts = dict.fromkeys(vb.keys(), 0.0)
		#print(f"cnt[{ilm}] contain(s): {len(vlm)} token(s) {type(vlm)}")
		if vlm: # to ensure list is not empty!
			for iTK, vTK in enumerate( vlm ): # [tk1, tk2, ..., tkN]
				#print(f"tk@idx:{iTK} | {type(vTK)} | {len(vTK)}")
				if vb.get(vTK) is not None:
					new_boosts[vTK] = new_boosts[vTK] + wg				
			new_boosts = {k: v for k, v in new_boosts.items() if v} # get rid of those keys(tokens) with zero values to reduce size
			#print(f"\tcontent[{ilm}] |new_boosts| = {len(new_boosts)}" )
			for k, v in new_boosts.items():
				total_boost = v
				prev_best_boost, prev_best_doc = updated_vb[k]
				if total_boost > prev_best_boost:
					updated_vb[k] = [total_boost, ilm]
	return updated_vb

def get_selected_content(cos_sim, cos_sim_idx, recommended_tokens, df_users_tokens):
	print(f"Selected Content (using top-{len(recommended_tokens)} recommended tokens) INEFFICIENT".center(120, " "))
	nUsers_with_max_cosine = get_nUsers_with_max(cos_sim, cos_sim_idx, df_users_tokens, N=3)
	df = df_users_tokens[df_users_tokens["user_ip"].isin(nUsers_with_max_cosine)]
	print(df.shape)

	# lemmatized_content = [[tk1, tk2, tk3, ...], [tk1, tk2, ...], [tk1, ...], ...]
	for usr, lemmatized_content, content_text in zip(df["user_ip"], df["nwp_content_lemma_separated"], df["nwp_content_raw_text"]):
		user_selected_content_counter, user_selected_content, user_selected_content_idx = 0, None, None
		print(f"{usr} visited {len(lemmatized_content)} document(s) {type(lemmatized_content)}".center(150, "-"))
		for idx_cnt, tks_list in enumerate(lemmatized_content):
			print(f"<> document[{idx_cnt}] "
						f"contain(s) {len(tks_list)} TOKEN(s) {type(tks_list)}"
					)			
			for iTK, vTK in enumerate(recommended_tokens):
				print(f"recTK[{iTK}]: {vTK:<30}=> previous_counter: {user_selected_content_counter:<15}current_counter: {tks_list.count(vTK)}")
				if tks_list.count(vTK) > user_selected_content_counter:
					print(f"bingoo, found with token[{iTK}]: {vTK}: {tks_list.count(vTK)}".center(80, " "))
					user_selected_content_counter = tks_list.count(vTK)
					user_selected_content = content_text[0]
					user_selected_content_idx = idx_cnt

			#print("-"*100)
		print(f"\n{usr} selected content @idx: {user_selected_content_idx} | num_occurrence(s): {user_selected_content_counter}:\n")
		#print(f"\nSelected content:")
		print(user_selected_content)
		print("+"*180)

def get_users_tokens_df(dframe: pd.DataFrame, bow: Dict[str, int]):
	sqFile = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_search_queries.gz")
	snHWFile = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_snippets_hw.gz")
	snFile = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_snippets.gz")
	cntHWFile = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_contents_hw.gz")
	cntPTFile = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_content_pt.gz")
	cntFile = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_contents.gz")
	df_preprocessed = dframe.copy()

	print(f"\n>> Getting {cntFile} ...")	
	try:
		df_preprocessed["nwp_content_ocr_text_tklm"] = load_pickle(fpath=cntFile)
	except:
		print(f"<!> Contents [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['nwp_content_ocr_text'] = df_preprocessed["nwp_content_results"].map(get_raw_cnt, na_action='ignore')
		cnt_list = df_preprocessed["nwp_content_ocr_text"].map(lambda snt: get_lemmatized_cnt(snt, lm=args.lmMethod), na_action='ignore') # inp: "my car is black." => out ["car", "black"]
		df_preprocessed['nwp_content_ocr_text_tklm'] = cnt_list
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		save_pickle(pkl=cnt_list, fname=cntFile)
	
	print(f"\n>> Getting {sqFile} ...")	
	try:
		df_preprocessed["search_query_phrase_tklm"] = load_pickle(fpath=sqFile)
	except:
		print(f"<!> Search query phrases [tokenization + lemmatization]...")
		st_t = time.time()
		sq_list = df_preprocessed["search_query_phrase"].map(lambda lst: get_lemmatized_sqp(lst, lm=args.lmMethod), na_action="ignore")
		df_preprocessed["search_query_phrase_tklm"] = sq_list
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		save_pickle(pkl=sq_list, fname=sqFile)

	print(f"\n>> Getting {snHWFile} ...")	
	try:
		df_preprocessed["search_results_hw_snippets_tklm"] = load_pickle(fpath=snHWFile)
	except:
		print(f"<!> Snippet Highlighted Words [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['search_results_hw_snippets'] = df_preprocessed["search_results"].map(get_raw_snHWs, na_action='ignore')
		snHW_list = df_preprocessed["search_results_hw_snippets"].map(lambda lst: get_lemmatized_snHWs(lst, lm=args.lmMethod), na_action='ignore')
		df_preprocessed['search_results_hw_snippets_tklm'] = snHW_list
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		save_pickle(pkl=snHW_list, fname=snHWFile)

	print(f"\n>> Getting {snFile} ...")	
	try:
		df_preprocessed["search_results_snippets_tklm"] = load_pickle(fpath=snFile)
	except:
		print(f"<!> Snippets [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['search_results_snippets'] = df_preprocessed["search_results"].map(get_raw_sn, na_action='ignore')
		sn_list = df_preprocessed["search_results_snippets"].map(lambda lst: get_lemmatized_sn(lst, lm=args.lmMethod), na_action='ignore')
		df_preprocessed['search_results_snippets_tklm'] = sn_list
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		save_pickle(pkl=sn_list, fname=snFile)
	
	print(f"\n>> Getting {cntHWFile} ...")	
	try:
		df_preprocessed["nwp_content_ocr_text_hw_tklm"] = load_pickle(fpath=cntHWFile)
	except:
		print(f"<!> Content Highlighted Words [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['nwp_content_ocr_text_hw'] = df_preprocessed["nwp_content_results"].map(get_raw_cntHWs, na_action='ignore')
		cntHW_list = df_preprocessed["nwp_content_ocr_text_hw"].map(lambda lst: get_lemmatized_cntHWs(lst, lm=args.lmMethod), na_action='ignore')
		df_preprocessed['nwp_content_ocr_text_hw_tklm'] = cntHW_list
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		save_pickle(pkl=cntHW_list, fname=cntHWFile)

	print(f"\n>> Getting {cntPTFile} ...")	
	try:
		df_preprocessed["nwp_content_ocr_text_pt_tklm"] = load_pickle(fpath=cntPTFile)
	except:
		print(f"<!> Content Parsed Terms [tokenization + lemmatization]...")
		st_t = time.time()
		df_preprocessed['nwp_content_ocr_text_pt'] = df_preprocessed["nwp_content_results"].map(get_raw_cntPTs, na_action='ignore')
		cntPT_list = df_preprocessed["nwp_content_ocr_text_pt"].map(lambda lst: get_lemmatized_cntPTs(lst, lm=args.lmMethod), na_action='ignore')
		df_preprocessed['nwp_content_ocr_text_pt_tklm'] = cntPT_list
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		save_pickle(pkl=cntPT_list, fname=cntPTFile)

	print(f"Original_DF: {dframe.shape} => DF_preprocessed: {df_preprocessed.shape}".center(110, "-"))
	print(f"USERs-TOKENs DataFrame".center(120, " "))
	
	st_t = time.time()
	users_list = list()
	search_query_phrase_tokens_list = list()
	search_results_hw_snippets_tokens_list = list()
	search_results_snippets_tokens_list = list()
	search_results_snippets_raw_texts_list = list()

	nwp_content_pt_tokens_list = list()
	nwp_content_hw_tokens_list = list()
	nwp_content_lemmas_all_list = list()
	nwp_content_raw_texts_list = list()
	nwp_content_lemmas_separated_list = list()
	
	for n, g in df_preprocessed.groupby("user_ip"):
		#print(n)
		users_list.append(n)

		search_query_phrase_tokens_list.append( [tk for tokens in g[g["search_query_phrase_tklm"].notnull()]["search_query_phrase_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		search_results_hw_snippets_tokens_list.append( [tk for tokens in g[g["search_results_hw_snippets_tklm"].notnull()]["search_results_hw_snippets_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		nwp_content_hw_tokens_list.append( [tk for tokens in g[g["nwp_content_ocr_text_hw_tklm"].notnull()]["nwp_content_ocr_text_hw_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		nwp_content_pt_tokens_list.append( [tk for tokens in g[g["nwp_content_ocr_text_pt_tklm"].notnull()]["nwp_content_ocr_text_pt_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		
		# comment for speedup:
		search_results_snippets_tokens_list.append( [ tk for tokens in g[g["search_results_snippets_tklm"].notnull()]["search_results_snippets_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		search_results_snippets_raw_texts_list.append( [ sent for sentences in g[g["search_results_snippets"].notnull()]["search_results_snippets"].values.tolist() if sentences for sent in sentences if sent ] )
	
		#print( len( g[g["search_results_snippets"].notnull()]["search_results_snippets"].values.tolist() ) )
		#print( g[g["search_results_snippets"].notnull()]["search_results_snippets"].values.tolist() )
		nwp_content_lemmas_all_list.append( [tk for tokens in g[g["nwp_content_ocr_text_tklm"].notnull()]["nwp_content_ocr_text_tklm"].values.tolist() if tokens for tk in tokens if tk] ) #[tk1, tk2, tk3, ...]
		nwp_content_lemmas_separated_list.append( [lm for lm in g[g["nwp_content_ocr_text_tklm"].notnull()]["nwp_content_ocr_text_tklm"].values.tolist() if lm ] ) #[ [tk1, tk2, ...], [tk1, tk2, ...], [tk1, tk2, ...], ... ]
		nwp_content_raw_texts_list.append( [sentences for sentences in g[g["nwp_content_ocr_text"].notnull()]["nwp_content_ocr_text"].values.tolist() if sentences ] ) # [cnt1, cnt2, …, cntN]

		#print("#"*150)

	# uncomment for speedup:
	#nwp_content_lemmas_all_list = [f"nwp_content_{i}" for i in range(len(users_list))]
	#search_results_snippets_tokens_list = [f"snippet_{i}" for i in range(len(users_list))]

	print(len(users_list), 
				len(search_query_phrase_tokens_list),
				len(search_results_hw_snippets_tokens_list),
				len(search_results_snippets_tokens_list),
				len(nwp_content_lemmas_all_list),
				len(nwp_content_lemmas_separated_list),
				len(nwp_content_pt_tokens_list),
				len(nwp_content_hw_tokens_list),
				len(nwp_content_raw_texts_list),
				len(search_results_snippets_raw_texts_list),
				)
	#return

	df_user_token = pd.DataFrame(list(zip(users_list, 
																				search_query_phrase_tokens_list, 
																				search_results_hw_snippets_tokens_list, 
																				search_results_snippets_tokens_list, 
																				nwp_content_lemmas_all_list, 
																				nwp_content_lemmas_separated_list,
																				nwp_content_pt_tokens_list, 
																				nwp_content_hw_tokens_list,
																				nwp_content_raw_texts_list,
																				search_results_snippets_raw_texts_list,
																			)
																	),
																columns =['user_ip',
																					'qu_tokens',
																					'snippets_hw_token', 
																					'snippets_token',
																					'nwp_content_lemma_all',
																					'nwp_content_lemma_separated',
																					'nwp_content_pt_token',
																					'nwp_content_hw_token',
																					'nwp_content_raw_text',
																					'snippets_raw_text',
																				]
															)
	print(f"Elapsed_t: {time.time()-st_t:.2f} s | {df_user_token.shape}".center(120, " "))

	print(f"Implicit Feedback".center(100, "-"))
	st_t = time.time()
	df_user_token["user_token_interest"] = df_user_token.apply( lambda x_df: sum_all_tokens_appearance_in_vb(x_df, w_list, bow), axis=1, )	
	df_user_token["usrInt_qu_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="qu_tokens", wg=weightQueryAppearance, vb=bow), axis=1)
	df_user_token["usrInt_sn_hw_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="snippets_hw_token", wg=weightSnippetHWAppearance, vb=bow), axis=1)
	df_user_token["usrInt_sn_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="snippets_token", wg=weightSnippetAppearance, vb=bow), axis=1)
	df_user_token["usrInt_cnt_hw_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="nwp_content_hw_token", wg=weightContentHWAppearance, vb=bow), axis=1)
	df_user_token["usrInt_cnt_pt_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="nwp_content_pt_token", wg=weightContentPTAppearance, vb=bow), axis=1)
	df_user_token["usrInt_cnt_tk"] = df_user_token.apply(lambda x_df: sum_tk_apperance_vb(x_df, qcol="nwp_content_lemma_all", wg=weightContentAppearance, vb=bow), axis=1)
	df_user_token["selected_content"] = df_user_token["nwp_content_lemma_separated"].map(lambda l_of_l: get_newspaper_content(l_of_l, vb=bow, wg=weightContentAppearance), na_action="ignore")
	
	print(f"<Elapsed_t: {time.time()-st_t:.2f} s> | {df_user_token.shape}".center(120, " "))

	#print(type( df_user_token["user_token_interest"].values.tolist()[0] ), type( df_user_token["usrInt_qu_tk"].values.tolist()[0] ))
	#print( len(df_user_token["user_token_interest"].values.tolist()), df_user_token["user_token_interest"].values.tolist() )
	#print( len(df_user_token["usrInt_qu_tk"].values.tolist()), df_user_token["usrInt_qu_tk"].values.tolist() )
	#print(df_user_token.shape, list(df_user_token.columns))
	
	#print(df_user_token.info())	
	#print(df_user_token[["user_ip", "usrInt_qu_tk"]].head())
	#print("#"*100)
	#print(df_user_token[["user_ip", "user_token_interest"]].head())
	
	df_user_token_fname = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_user_tokens_df_{len(bow)}_BoWs.gz")
	save_pickle(pkl=df_user_token, fname=df_user_token_fname)

	print(f"USERs-TOKENs DataFrame".center(120, " "))
	return df_user_token

def get_sparse_matrix(df):
	print(f"Getting Sparse Matrix from DF: {df.shape}".center(110, '-'))
	#print(list(df.columns))
	#print(df.dtypes)

	print( df['user_token_interest'].apply(pd.Series).head(15) )
	print(">"*100)
	df_new = pd.concat( [df["user_ip"], df['user_token_interest'].apply(pd.Series)], axis=1).set_index("user_ip")

	#print(df_new.head(15))
	#print("<"*100)

	##########################Sparse Matrix info##########################
	sparse_matrix = csr_matrix(df_new.values, dtype=np.float32) # (n_usr x n_vb)
	#print("#"*110)
	#print(f"{type(sparse_matrix)} {sparse_matrix.shape} : |tot_elem|: {sparse_matrix.shape[0]*sparse_matrix.shape[1]}")
	#print(f"<> Non-zeros vals: {sparse_matrix.data}")# Viewing stored data (not the zero items)
	#print(sparse_matrix.toarray()[:25, :18])
	#print(f"<> |Non-zero vals|: {sparse_matrix.count_nonzero()}") # Counting nonzeros
	#print("#"*110)
	##########################Sparse Matrix info##########################
	
	sp_mat_user_token_fname = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_user_tokens_sparse_matrix_{sparse_matrix.shape[1]}_BoWs.gz")

	save_pickle(pkl=sparse_matrix, fname=sp_mat_user_token_fname)

	return sparse_matrix

def run(df_inp, qu_phrase, topK=5, normalize_sp_mtrx=False, ):
	print(f">> Running {__file__} with {args.lmMethod.upper()} lemmatizer")
	#BoWs = get_BoWs(dframe=df_inp, fprefix=fprefix, lm=args.lmMethod)
	BoWs = get_cBoWs(dframe=df_inp, fprefix=fprefix, lm=args.lmMethod)
	df_usr_tk = get_users_tokens_df(dframe=df_inp, bow=BoWs)
	del df_inp
	gc.collect()

	try:
		sp_mat_rf = load_pickle(fpath=os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_user_tokens_sparse_matrix_{len(BoWs)}_BoWs.gz"))
	except:
		sp_mat_rf = get_sparse_matrix(df_usr_tk)

	print(f"{type(sp_mat_rf)} (Users-Tokens): {sp_mat_rf.shape} | {sp_mat_rf.toarray().nbytes} | {sp_mat_rf.toarray().dtype}")
	#return

	if normalize_sp_mtrx:
		#sp_mat_rf = normalize(sp_mat_rf, norm="l2", axis=0) # l2 normalize by column -> items
		sp_mat_rf = normalize(sp_mat_rf, norm="l2", axis=1) # l2 normalize by rows -> users
		
	#get_user_n_maxVal_byTK(sp_mat_rf, df_usr_tk, BoWs, )
	#return
	plot_heatmap_sparse(sp_mat_rf, df_usr_tk, BoWs, norm_sp=normalize_sp_mtrx, ifb_log10=False)
	
	#print("#"*150)
	print(f"".center(100,' '))
	query_phrase_tk = get_lemmatized_sqp(qu_list=[qu_phrase], lm=args.lmMethod)
	print(f"Raw Query Phrase: {qu_phrase} contains {len(query_phrase_tk)} lemma(s)\t{query_phrase_tk}")
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

	print(f"Getting users of {np.count_nonzero(query_vector)} token(s) / |QUE_TK|: {len(query_phrase_tk)}".center(120, "-"))
	for iTK, vTK in enumerate(query_phrase_tk):
		if BoWs.get(vTK):
			users_names, users_values_total, users_values_separated = get_users_byTK(sp_mat_rf, df_usr_tk, BoWs, token=vTK)
			plot_users_by(token=vTK, usrs_name=users_names, usrs_value_all=users_values_total, usrs_value_separated=users_values_separated, topUSRs=15, bow=BoWs, norm_sp=normalize_sp_mtrx )
			plot_usersInterest_by(token=vTK, sp_mtrx=sp_mat_rf, users_tokens_df=df_usr_tk, bow=BoWs, norm_sp=normalize_sp_mtrx)

	#cos_sim, cos_sim_idx = get_cs_sklearn(query_vector, sp_mat_rf.toarray(), qu_phrase, query_phrase_tk, df_usr_tk, norm_sp=normalize_sp_mtrx) # qu_ (nItems,) => (1, nItems) -> cos: (1, nUsers)
	cos_sim, cos_sim_idx = get_cs_faiss(query_vector, sp_mat_rf.toarray(), qu_phrase, query_phrase_tk, df_usr_tk, norm_sp=normalize_sp_mtrx) # qu_ (nItems,) => (1, nItems) -> cos: (1, nUsers)

	print(f"Cosine Similarity (1 x nUsers): {cos_sim.shape} {type(cos_sim)} "
				f"Allzero: {np.all(cos_sim.flatten()==0.0)} "
				f"(min, max, sum): ({cos_sim.min()}, {cos_sim.max():.2f}, {cos_sim.sum():.2f})"
			)
	
	if np.all(cos_sim.flatten()==0.0):
		print(f"Sorry, We couldn't find similar results to >> {Fore.RED+Back.WHITE}{qu_phrase}{Style.RESET_ALL} << in our database! Search again!")
		return

	plot_tokens_by_max(cos_sim, cos_sim_idx, sp_mtrx=sp_mat_rf, users_tokens_df=df_usr_tk, bow=BoWs, norm_sp=normalize_sp_mtrx)
	#return
	nUsers, nItems = sp_mat_rf.shape
	print(f"avgRecSysVec (1 x nItems) | nUsers={nUsers} |nItems={nItems}".center(120, " "))
	#print("#"*120)
	#cos = np.random.rand(nUsers).reshape(1, -1)
	#usr_itm = np.random.randint(100, size=(nUsers, nItems))
	#avgrec = np.zeros((1, nItems))
	prev_avgrec = np.zeros((1, nItems))
	#print(f"> avgrec{avgrec.shape}:\n{avgrec}")
	#print()
	
	#print(f"> cos{cos.shape}:\n{cos}")
	#print()

	#print(f"> user-item{usr_itm.shape}:\n{usr_itm}")
	#print("#"*100)
	st_t = time.time()
	for iUser, vUser in enumerate(df_usr_tk['user_ip'].values.tolist()):
		idx_cosine = np.where(cos_sim_idx.flatten()==iUser)[0][0]
		#print(f"argmax(cosine_sim): {idx_cosine} => cos[uIDX: {iUser}] = {cos_sim[0, idx_cosine]}")
		if cos_sim[0, idx_cosine] != 0.0:
			
			print(f"iUser[{iUser}]: {df_usr_tk.loc[iUser, 'user_ip']}".center(140, " "))
			print(f"avgrec (previous): {prev_avgrec.shape} "
						f"(min, max_@(iTK), sum): ({prev_avgrec.min()}, {prev_avgrec.max():.5f}_@(iTK[{np.argmax(prev_avgrec)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(prev_avgrec) )]}), {prev_avgrec.sum():.1f}) "
						f"{prev_avgrec} | Allzero: {np.all(prev_avgrec==0.0)}"
					)
			
			userInterest = sp_mat_rf.toarray()[iUser, :].reshape(1, -1) # 1 x nItems
			
			print(f"<> userInterest[{iUser}]: {userInterest.shape} "
						f"(min, max_@(iTK), sum): ({userInterest.min()}, {userInterest.max():.5f}_@(iTK[{np.argmax(userInterest)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(userInterest) )]}), {userInterest.sum():.1f}) "
						f"{userInterest} | Allzero: {np.all(userInterest==0.0)}"
					)
			
			userInterest_norm = np.linalg.norm(userInterest)
			userInterest = normalize(userInterest, norm="l2", axis=1)
			
			print(f"<> userInterest(norm={userInterest_norm:.3f})[{iUser}]: {userInterest.shape} " 
						f"(min, max_@(iTK), sum): ({userInterest.min()}, {userInterest.max():.5f}_@(iTK[{np.argmax(userInterest)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(userInterest) )]}), {userInterest.sum():.1f}) "
						f"{userInterest} | Allzero: {np.all(userInterest==0.0)}"
					)
			
			if not np.all(userInterest==0.0):
				print(f"\tFound {np.count_nonzero(userInterest.flatten())} non-zero userInterest element(s)")
				print(f"\ttopK TOKENS     user[{iUser}]: {[list(BoWs.keys())[list(BoWs.values()).index( i )] for i in np.flip( np.argsort( userInterest.flatten() ) )[:10] ]}")
				print(f"\ttopK TOKENS val user[{iUser}]: {[v for v in np.flip( np.sort( userInterest.flatten() ) )[:10] ]}")
			
			update_term = (cos_sim[0, idx_cosine] * userInterest)
			avgrec = prev_avgrec + update_term #avgrec = avgrec + (cos_sim[0, idx_cosine] * userInterest)
			prev_avgrec = avgrec
			
			print(f"avgrec (current): {avgrec.shape} "
						f"(min, max_@(iTK), sum): ({avgrec.min()}, {avgrec.max():.5f}_@(iTK[{np.argmax(avgrec)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(avgrec) )]}), {avgrec.sum():.1f}) "
						f"{avgrec} | Allzero: {np.all(avgrec==0.0)}"
					)
			print("-"*150)
			
	avgrec = avgrec / np.sum(cos_sim)
	print(f"Elapsed_t: {time.time()-st_t:.2f} s".center(120, " "))

	print(f"avgRecSys: {avgrec.shape} {type(avgrec)} "
				f"Allzero: {np.all(avgrec.flatten() == 0.0)} "
				f"(min, max_@(iTK), sum): ({avgrec.min()}, {avgrec.max():.5f}"
				f"@(iTK[{np.argmax(avgrec)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(avgrec) )]}), {avgrec.sum():.2f})"
			)

	"""
	print(f"checking original [not sorted] avgRecSys".center(100, "-"))
	print(avgrec.flatten()[:10])
	print("#"*100)
	print(avgrec.flatten()[-10:])
	print(f"checking original [not sorted] avgRecSys".center(100, "-"))
	"""
	f, ax = plt.subplots()
	ax.scatter(	x=np.arange(len(avgrec.flatten())), 
							y=avgrec.flatten(), 
							facecolor="k",
							#s=scales,
							edgecolors='w',
							#alpha=alphas,
							marker=".",
						)
	ax.set_title(f"avgRecSys: max: {avgrec.max():.5f} @(iTK[{np.argmax(avgrec)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(avgrec) )]})")
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_avgRecSys_{nItems}items.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

	print(f"idx:\n{avgrec.flatten().argsort()[-25:]}")
	print([k for i in avgrec.flatten().argsort()[-25:] for k, v in BoWs.items() if v==i ] )
	print(f">> sorted_recsys:\n{np.sort(avgrec.flatten())[-25:]}")


	all_recommended_tks = [k for idx in avgrec.flatten().argsort()[-25:] for k, v in BoWs.items() if (idx not in np.nonzero(query_vector)[0] and v==idx)]
	print(f"TOP-15: (all: {len(all_recommended_tks)}):\n{all_recommended_tks[-15:]}")
	topK_recommended_tokens = all_recommended_tks[-(topK+0):]
	print(f"top-{topK} recommended Tokens: {len(topK_recommended_tokens)}: {topK_recommended_tokens}")
	topK_recommended_tks_weighted_user_interest = [ avgrec.flatten()[BoWs.get(vTKs)] for iTKs, vTKs in enumerate(topK_recommended_tokens)]
	print(f"top-{topK} recommended Tokens weighted user interests: {len(topK_recommended_tks_weighted_user_interest)}: {topK_recommended_tks_weighted_user_interest}")
	#return
	"""
	st_t = time.time()
	get_selected_content(cos_sim, cos_sim_idx, topK_recommended_tokens, df_usr_tk)
	#print(f"Elapsed_t: {time.time()-st_t:.2f} s".center(120, " "))
	"""
	get_nwp_cnt_by_nUsers_with_max(cos_sim, cos_sim_idx, sp_mat_rf, df_usr_tk, BoWs, recommended_tokens=topK_recommended_tokens, norm_sp=normalize_sp_mtrx)

	print(f"Getting users of {len(topK_recommended_tokens)} tokens of top-{topK} RecSys".center(120, "-"))
	for ix, tkv in enumerate(topK_recommended_tokens):
		users_names, users_values_total, users_values_separated = get_users_byTK(sp_mat_rf, df_usr_tk, BoWs, token=tkv)
		
		plot_users_by(token=tkv, usrs_name=users_names, usrs_value_all=users_values_total, usrs_value_separated=users_values_separated, topUSRs=15, bow=BoWs, norm_sp=normalize_sp_mtrx )
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

def get_cs_faiss(QU, RF, query_phrase: str, query_token, users_tokens_df:pd.DataFrame, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original" # unimportant!
	device = "GPU" if torch.cuda.is_available() else "CPU"
	QU = QU.reshape(1, -1).astype(np.float32) # QU: (nItems, ) => (1, nItems)
	RF = RF.astype(np.float32) # RF: (nUsers, nItems) 
	
	print(f"<Faiss> {device} Cosine Similarity: "
			 	f"QUERY: {QU.shape} {type(QU)} {QU.dtype}"
				f" vs. "
				f"REFERENCE: {RF.shape} {type(RF)} {RF.dtype}".center(110, " ")
			)
	"""
	RF = normalize(RF, norm="l2", axis=1)
	#QU = QU / np.linalg.norm(QU)
	QU = normalize(QU, norm="l2", axis=1)
	"""
	faiss.normalize_L2(RF)
	faiss.normalize_L2(QU)
	k=2048-1 if RF.shape[0]>2048 and device=="GPU" else RF.shape[0] # getting k nearest neighbors

	st_t = time.time()
	if torch.cuda.is_available():
		res = faiss.StandardGpuResources() # use a single GPU
		index = faiss.GpuIndexFlatIP(res, RF.shape[1]) 
	else:
		index = faiss.IndexFlatIP(RF.shape[1])
	index.add(RF)
	sorted_cosine, sorted_cosine_idx = index.search(QU, k=k)
	print(f"Elapsed_t: {time.time()-st_t:.3f} s".center(100, " "))

	#print(sorted_cosine_idx.flatten()[:17])
	#print(sorted_cosine.flatten()[:17])
	
	plot_cs(sorted_cosine, sorted_cosine_idx, QU, RF, query_phrase, query_token, users_tokens_df, norm_sp)
	return sorted_cosine, sorted_cosine_idx

def get_cs_sklearn(QU, RF, query_phrase: str, query_token, users_tokens_df:pd.DataFrame, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original"
	QU = QU.reshape(1, -1) #qu_ (nItems,) => (1, nItems) 
	print(f"Sklearn Cosine Similarity QUERY: {QU.shape} vs REFERENCE: {RF.shape}".center(110, " ")) # QU: (nItems, ) => (1, nItems) | RF: (nUsers, nItems) 
	st_t = time.time()
	cos_sim = cosine_similarity(QU, RF) # -> cos: (1, nUsers)
	sorted_cosine = np.flip(np.sort(cos_sim)) # descending
	sorted_cosine_idx = np.flip(cos_sim.argsort()) # descending
	print(f"Elapsed_t: {time.time()-st_t:.3f} s".center(100, " "))
	print(sorted_cosine_idx.flatten()[:17])
	print(sorted_cosine.flatten()[:17])
	plot_cs(sorted_cosine, sorted_cosine_idx, QU, RF, query_phrase, query_token, users_tokens_df, norm_sp)
	return sorted_cosine, sorted_cosine_idx

def plot_cs(cos_sim, cos_sim_idx, QU, RF, query_phrase, query_token, users_tokens_df, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original"
	print(f"Plotting Cosine Similarity {cos_sim.shape} | Raw Query Phrase: {query_phrase} | Query Token(s) : {query_token}")	
	alphas = np.ones_like(cos_sim.flatten())
	scales = 100*np.ones_like(cos_sim.flatten())
	for i, v in np.ndenumerate(cos_sim.flatten()):
		if v==0:
			alphas[i] = 0.05
			scales[i] = 5

	f, ax = plt.subplots()
	ax.scatter(	x=cos_sim_idx.flatten(),
							y=cos_sim.flatten(), 
							facecolor="g", 
							s=scales, 
							edgecolors='w',
							alpha=alphas,
							marker=".",
						)
	

	######################################################################3
	#TODO: must be removed and replaced with!
	N=3
	if np.count_nonzero(cos_sim.flatten()) < N:
		N = np.count_nonzero(cos_sim.flatten())
	
	nUsers_with_max_cosine = users_tokens_df.loc[cos_sim_idx.flatten()[:N], 'user_ip'].values.tolist()
	######################################################################3

	ax.scatter(x=cos_sim_idx.flatten()[:N], y=cos_sim.flatten()[:N], facecolor='none', marker="o", edgecolors="r", s=100)
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
						s=f"Query Phrase: {query_phrase}", 
						fontsize=10.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.text(	x=0.5,
						y=0.91,
						s=f"Query (1 x nItems): {QU.shape} {sp_type} Sparse Matrix (nUsers x nItems): {RF.shape} Cosine Similarity (1 x nUsers): {cos_sim.shape}",
						fontsize=9.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.text(	x=0.5,
						y=0.88,
						s=f"{N}-Max cosine(s): {nUsers_with_max_cosine} : {cos_sim.flatten()[:N]}",
						fontsize=8.5,
						ha="center", 
						color="r",
						transform=f.transFigure,
					)
	
	plt.subplots_adjust(top=0.86, wspace=0.1)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_cosine_{QU.shape[1]}Items_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def get_nUsers_with_max(cos_sim, cos_sim_idx, users_tokens_df:pd.DataFrame, N:int=3):
	if np.count_nonzero(cos_sim.flatten()) < N:
		N = np.count_nonzero(cos_sim.flatten())
	print(f"\n< {N} > user(s) with max cosine similarity:", end=" ")

	topN_max_cosine = cos_sim.flatten()[:N]
	nUsers_with_max_cosine = users_tokens_df.loc[cos_sim_idx.flatten()[:N], 'user_ip'].values.tolist()
	print(nUsers_with_max_cosine, topN_max_cosine)
	return nUsers_with_max_cosine

def get_nwp_cnt_by_nUsers_with_max(cos_sim, cos_sim_idx, sp_mtrx, users_tokens_df, bow, recommended_tokens, norm_sp: bool=False):
	nUsers_with_max_cosine = get_nUsers_with_max(cos_sim, cos_sim_idx, users_tokens_df, N=3)
	print(f"top-{len(recommended_tokens)} recommeded token(s): {recommended_tokens}")
	for recTK_idx, recTK in enumerate(recommended_tokens):
		print(f">> recTK[{recTK_idx}]: {recTK}")
		max_boost = 0.0
		max_boost_idoc = 0
		winner_user = None
		winner_content = None
		for iUSR, vUSR in enumerate(nUsers_with_max_cosine):
			print(vUSR, end="\t")
			tboost, idoc = users_tokens_df[users_tokens_df["user_ip"]==vUSR]["selected_content"].values.tolist()[0].get(recTK)
			print(f"[total_boost, idoc]: [{tboost}, {idoc}]")
			if tboost > max_boost:
				max_boost = tboost
				max_boost_idoc = idoc
				winner_user = vUSR
		if winner_user:
			user_all_doc = users_tokens_df[users_tokens_df["user_ip"]==winner_user]["nwp_content_raw_text"].values.tolist()[0]
			print(f"winner: {winner_user}, idoc: {max_boost_idoc} / ALL:|{len(user_all_doc)}|, {max_boost}".center(120, " "))
			user_best_doc = user_all_doc[max_boost_idoc]
			print(f"\ncontent {type(user_best_doc)} with {len(user_best_doc)} character(s):\n")
			print(user_best_doc)
		print("-"*120)

	#return

def plot_tokens_by_max(cos_sim, cos_sim_idx, sp_mtrx, users_tokens_df, bow, norm_sp: bool=False):
	nUsers_with_max_cosine = get_nUsers_with_max(cos_sim, cos_sim_idx, users_tokens_df, N=3)
	for _, usr in enumerate(nUsers_with_max_cosine):
		tokens_names, tokens_values_total, tokens_values_separated = get_tokens_byUSR(sp_mtrx, users_tokens_df, bow, user=usr)
		plot_tokens_by(	userIP=usr, 
										tks_name=tokens_names, 
										tks_value_all=tokens_values_total, 
										tks_value_separated=tokens_values_separated, 
										topTKs=20,
										bow=bow,
										norm_sp=norm_sp,
									)
	print(f"DONE".center(100, "-"))

def plot_tokens_by(userIP, tks_name, tks_value_all, tks_value_separated, topTKs, bow, norm_sp: bool=False):
	sp_type = "Normalized" if norm_sp else "Original"
	nTokens_orig = len(tks_name)

	if len(tks_name) > topTKs:
		tks_name = tks_name[:topTKs]
		tks_value_all = tks_value_all[:topTKs]
		tks_value_separated = [elem[:topTKs] for elem in tks_value_separated]

	nTokens = len(tks_name)
	print(f"Plotting top-{nTokens} token(s) out of |ALL_UnqTKs = {nTokens_orig}| {userIP}".center(110, "-"))

	f, ax = plt.subplots()
	ax.barh(tks_name, 
					tks_value_all,
					color="#0000ff",
					height=0.4,
				)
	ax.tick_params(axis='x', labelrotation=0, labelsize=7.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
	ax.set_xlabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)
	ax.invert_yaxis()  # labels read top-to-botto
	ax.set_title(f'Top-{nTokens} Unique Tokens / |ALL_UnqTKs = {nTokens_orig}| {userIP}', fontsize=10)
	ax.margins(1e-2, 5e-3)
	ax.spines[['top', 'right']].set_visible(False)
	for container in ax.containers:
		ax.bar_label(	container, 
									#rotation=45, # no rotation for barh 
									fontsize=6.0,
									padding=1.5,
									fmt='%.3f', #if norm_sp else '%.2f',
									label_type='edge',
								)
	ax.set_xlim(right=ax.get_xlim()[1]+1.0, auto=True)
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_usr_{userIP}_topTKs{nTokens}_{len(bow)}_BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

	f, ax = plt.subplots()
	lft = np.zeros(len(tks_name))
	qcol_list = ["Search PHRs", "Snippet HWs", "Snippet Appr", "Content HWs", "Content PRTs", "Content Appr",]
	hbars = list()
	for i, v in enumerate( tks_value_separated ):
		#print(i, tks_name, v)
		hbar = ax.barh(	tks_name, 
										v, 
										color=clrs[i], 
										height=0.65,
										left=lft, 
										edgecolor='w', 
										lw=0.4, 
										label=f"{qcol_list[i]:<20}w: {w_list[i]:<{10}.{3}f}{[f'{val:.3f}' for val in v]}",
									)
		lft += v
		hbars.append(hbar)
		#print(hbar.datavalues)
	
	ax.tick_params(axis='x', labelrotation=0, labelsize=7.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
	
	ax.set_xlabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)
	ax.invert_yaxis()  # labels read top-to-botto
	ax.set_title(f'Top-{nTokens} Unique Tokens / |ALL_UnqTKs = {nTokens_orig}| {userIP}', fontsize=10)
	ax.margins(1e-2, 5e-3)
	ax.spines[['top', 'right']].set_visible(False)
	ax.legend(loc='lower right', fontsize=(115.0/topTKs))
	
	for bar in hbars:
		filtered_lbls = [f"{v:.1f}" if v>=5.0 else "" for v in bar.datavalues]
		ax.bar_label(	container=bar, 
									labels=filtered_lbls, 
									label_type='center', 
									rotation=0.0, 
									fontsize=5.0,
								)
	
	ax.set_xlim(right=ax.get_xlim()[1]+0.5, auto=True)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_usr_{userIP}_topTKs{nTokens}_separated_{len(bow)}_BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_users_by(token, usrs_name, usrs_value_all, usrs_value_separated, topUSRs, bow, norm_sp: bool=False):
	sp_type = "Normalized" if norm_sp else "Original"
	
	nUsers_orig = len(usrs_name)

	if len(usrs_name) > topUSRs:
		usrs_name = usrs_name[:topUSRs]
		usrs_value_all = usrs_value_all[:topUSRs]
		usrs_value_separated = [elem[:topUSRs] for elem in usrs_value_separated]

	nUsers = len(usrs_name)

	print(f"Plotting top-{nUsers} users out of |ALL_USRs = {nUsers_orig} | token: {token}".center(110, " "))

	f, ax = plt.subplots()
	ax.bar(usrs_name, usrs_value_all, color="#a6111122", width=0.58)
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
									fontsize=6.0,
									padding=1.5,
									fmt='%.3f',# if norm_sp else '%.2f', 
									label_type='edge',
									)
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_topUSRs{nUsers}_{len(bow)}_BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

	#print(f"Separated Values".center(100, " "))
	f, ax = plt.subplots()
	btn = np.zeros(len(usrs_name))
	qcol_list = ["Search PHRs", "Snippet HWs", "Snippet Appr", "Content HWs", "Content PRTs", "Content Appr",]
	vbars = list()
	for i, v in enumerate( usrs_value_separated ):
		#print(i, usrs_name, v)
		vbar = ax.bar(usrs_name, v, color=clrs[i], width=0.6, bottom=btn, edgecolor='w', lw=0.4, label=f"{qcol_list[i]:<20}w: {w_list[i]:<{10}.{3}f}{[f'{val:.3f}' for val in v]}")
		btn += v
		vbars.append(vbar)
	
	ax.tick_params(axis='x', labelrotation=90, labelsize=8.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=8.0)
	ax.set_ylabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)	
	ax.set_title(f'Top-{nUsers} User(s) / |ALL_USRs = {nUsers_orig}| for token: {token}', fontsize=11)
	ax.margins(1e-2, 5e-3)
	ax.spines[['top', 'right']].set_visible(False)
	ax.legend(loc="upper right", fontsize=(110.0/topUSRs) )

	for b in vbars:
		filtered_lbls = [f"{v:.1f}" if v>=7.0 else "" for v in b.datavalues]
		ax.bar_label(	b, 
									labels=filtered_lbls, 
									label_type='center', 
									rotation=0.0, 
									fontsize=6.0,
								)
	ax.set_ylim(top=ax.get_ylim()[1]+1.0, auto=True)
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_topUSRs{nUsers}_separated_{len(bow)}BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_usersInterest_by(token, sp_mtrx, users_tokens_df, bow, norm_sp: bool=False):
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
	
	N=3
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

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_usersInterest_{len(bow)}BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_heatmap_sparse(sp_mtrx, df_usr_tk, bow, norm_sp:bool=False, ifb_log10: bool=False):
	name_="sparse_matrix_user_vs_token"
	sp_type = "Normalized" if norm_sp else "Original"

	print(f"{f'{sp_type} Sparse Matrix {sp_mtrx.shape}'.center(100,'-')}")

	print(f"<> Non-zeros vals: {sp_mtrx.data}")# Viewing stored data (not the zero items)
	print(f"<> |Non-zero cells|: {sp_mtrx.count_nonzero()}") # Counting nonzeros
	mtrx = sp_mtrx.toarray() # to numpy array

	####################################################
	if ifb_log10:
		mtrx = np.log10(mtrx, out=np.zeros_like(mtrx), where=(mtrx!=0))
		#mtrx = normalize(mtrx, norm="l2", axis=0) # l2 normalize by column -> items
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
								#cmap="gist_yarg", # 0: white , max -> black
								cmap="GnBu",
								#cmap="gist_gray", # 0: black , max -> white
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

	plt.savefig(os.path.join( RES_DIR, f"heatmap_{sp_type}_log10_{ifb_log10}_SP_{len(bow)}_BoWs.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)
	print(f"Done".center(70, "-"))

def plot_tokens_distribution(sparseMat, users_tokens_df, queryVec, recSysVec, bow, norm_sp: bool=False, topK: int=5):
	sp_type = "Normalized" if norm_sp else "Original"

	print(f"\nPlotting Tokens Distribution in {sp_type} Sparse Matrix (nUsers, nItems): {sparseMat.shape}")
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

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_{args.lmMethod}_top{topK}_recsys_{len(bow)}_BoWs_{sp_type}_SP.png" ), bbox_inches='tight')

	plt.clf()
	plt.close(f)
	print(">> Done!")

def main():
	# usr_tk_dfs = [pd.concat( [df.user_ip, df.user_token_interest.apply(pd.Series).astype('float16')], axis=1 ) for f in glob.glob(os.path.join(dfs_path, "*.gz")) if ( re.search(r'_user_tokens_df_(\d+)_BoWs.gz', f) and (df:=load_df_pkl(fpath=f, ccols=["user_ip", "user_token_interest"])).shape[0]>0 ) ]
	# print(len(usr_tk_dfs))

	# print(usr_tk_dfs[0].info(verbose=True, memory_usage="deep"))
	# print(f"Memory usage of each column in bytes (total column(s)={len(list(usr_tk_dfs.columns))})")
	# print(usr_tk_dfs[0].memory_usage(deep=True, index=False, ))
	# print("#"*100)
	# print(usr_tk_dfs[0].head(10))
	# print("#"*100)
	# print(usr_tk_dfs[0].tail(10))
	# print("#"*100)
	print(f">> concat...")
	st_t = time.time()
	with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
		usr_tk_dfs = pd.concat( [pd.concat( [df.user_ip, df.user_token_interest.apply(pd.Series).astype('float16')], axis=1 ) for f in glob.glob(os.path.join(dfs_path, "*.gz")) if ( re.search(r'_user_tokens_df_(\d+)_BoWs.gz', f) and (df:=load_df_pkl(fpath=f, ccols=["user_ip", "user_token_interest"])).shape[0]>0 ) ] )
	print(f"Elapsed_t: {time.time()-st_t:.2f} s | "
				f"DF: {usr_tk_dfs.shape} | "
				f"unq_users: {len( usr_tk_dfs.user_ip.value_counts() )} | "
				f"unq_tokens: {len( list( usr_tk_dfs.columns.difference(['user_ip'])))}"
			)

	print(f">> groupby...")
	d = dict()
	for n, g in usr_tk_dfs.groupby("user_ip"):
		st_t_f = time.time()
		print(n, end="\t")
		# print(g)
		# print()
		print(g.loc[:, g.columns!="user_ip"].sum().values, end="\t")
		print(f"|non_zeros|: { ( g.loc[:, g.columns!='user_ip'].sum() > 0.0).sum() } / {len(g.loc[:, g.columns!='user_ip'].sum().values)}", end=" ")
		# d[n] = g.loc[:, g.columns.difference(['user_ip'])].sum()
		d[n] = g.loc[:, g.columns!="user_ip"].sum()
		print(f"\t\tElapsed_t: {time.time()-st_t_f:.2f} s")
		# print("#"*100)
	user_token_df = pd.DataFrame.from_dict(d, orient="index")
	print(f"\ttot_Elapsed_t: {time.time()-st_t:.2f} s")

	print("#"*100)
	print(user_token_df.info(verbose=True, memory_usage="deep"))
	print(f"Memory usage of each column in bytes (total column(s)={len(list(user_token_df.columns))})")
	print(user_token_df.memory_usage(deep=True, index=False, ))
	print("#"*100)
	print(user_token_df.head(10))
	print("#"*100)

	# try:
	# 	df_concat_fname = [f for f in os.listdir(dfs_path) if f.endswith("_concat.gz")][0]
	# 	# print(df_concat_fname)
	# 	df_concat_fpath = os.path.join(dfs_path, df_concat_fname)
	# 	# print(df_concat_fpath)
	# 	#df_raw = load_pickle(fpath=df_concat_fpath)
	# 	df_raw = load_df_pkl(fpath=df_concat_fpath)
	# 	#print(df_raw.shape)
	# 	ndfs = int(df_concat_fname[:df_concat_fname.find("_")])
	# 	#print(ndfs)
	# except:
	# 	df_raw, ndfs = get_concat_df(dir_path=args.dsPath)

	# global fprefix, RES_DIR
	# fprefix = f"{ndfs}_dfs_concat_{df_raw.shape[0]}_samples"
	# RES_DIR = make_result_dir(infile=fprefix)
	# # analyze_df(df=df_raw, fname=__file__)
	# # print(fprefix, RES_DIR)
	# run(df_inp=df_raw, qu_phrase=args.qphrase, normalize_sp_mtrx=args.normSP, topK=args.topTKs)

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
	os.system("clear")
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120, " "))
	main()	
	#practice()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120, " "))