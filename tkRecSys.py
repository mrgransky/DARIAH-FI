from utils import *
from nlp_utils import *

parser = argparse.ArgumentParser(	description='User-Item Recommendation system developed based on National Library of Finland (NLF) dataset', 
																	prog='RecSys USER-TOKEN', 
																	epilog='Developed by Farid Alijani',
																)

parser.add_argument('--dsPath', default=dataset_path, type=str) # smallest
parser.add_argument('--qphrase', default="Juha Sipilä Sahalahti", type=str)
parser.add_argument('--lmMethod', default="stanza", type=str)
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

def get_sparse_matrix(df):
	print(f"Sparse Matrix from DF: {df.shape}".center(110, '-'))
	if df.index.inferred_type != 'string':
		df = df.set_index('user_ip')
	sparse_matrix = csr_matrix(df.values, dtype=np.float32) # (n_usr x n_vb)
	print(f"{type(sparse_matrix)} (Users-Tokens): {sparse_matrix.shape} | "
				f"{sparse_matrix.toarray().nbytes} | {sparse_matrix.toarray().dtype} "
				f"|tot_elem|: {sparse_matrix.shape[0]*sparse_matrix.shape[1]} "
				f"|Non-zero vals|: {sparse_matrix.count_nonzero()}"
			)
	print("-"*110)
	sp_mat_user_token_fname = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_user_token_sparse_matrix_{sparse_matrix.shape[1]}_BoWs.gz")
	save_pickle(pkl=sparse_matrix, fname=sp_mat_user_token_fname)
	return sparse_matrix

def get_cs_faiss(QU, RF, query_phrase: str, query_token, users_tokens_df:pd.DataFrame, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original" # unimportant!
	device = "GPU" if torch.cuda.is_available() else "CPU"
	QU = QU.reshape(1, -1).astype(np.float32) # QU: (nItems, ) => (1, nItems)
	RF = RF.astype(np.float32) # RF: (nUsers, nItems) 
	
	print(f"<Faiss> {device} Cosine Similarity: "
			 	f"QUERY: {QU.shape} {type(QU)} {QU.dtype}"
				f" vs. "
				f"REFERENCE: {RF.shape} {type(RF)} {RF.dtype}".center(160, " ")
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
	print(f"Elapsed_t: {time.time()-st_t:.3f} s | {sorted_cosine.shape}".center(100, " "))

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

	# sorted_cosine = np.flip(np.sort(cos_sim)) # descending
	# sorted_cosine_idx = np.flip(cos_sim.argsort()) # descending

	sorted_cosine = cos_sim
	sorted_cosine_idx = cos_sim.argsort()

	print(f"Elapsed_t: {time.time()-st_t:.3f} s".center(100, " "))
	print(sorted_cosine_idx.flatten()[:17])
	print(sorted_cosine.flatten()[:17])
	DF = pd.DataFrame(sorted_cosine)
	DF.to_csv("kantakirjasonni.csv")
	plot_cs(sorted_cosine, sorted_cosine_idx, QU, RF, query_phrase, query_token, users_tokens_df, norm_sp)
	return sorted_cosine, sorted_cosine_idx

def plot_cs(cos_sim, cos_sim_idx, QU, RF, query_phrase, query_token, users_tokens_df, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original"
	print(f"Plotting Cosine Similarity {cos_sim.shape} | Raw Query Phrase: {query_phrase} | Query Lemma(s) : {query_token}")	
	
	if users_tokens_df.index.inferred_type == 'string':
		users_tokens_df = users_tokens_df.reset_index()#.rename(columns = {'index':'user_ip'})

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
	
	# nUsers_with_max_cosine = users_tokens_df.loc[cos_sim_idx.flatten()[:N], 'user_ip'].values
	print(cos_sim.flatten().max())
	print(cos_sim_idx.flatten().max())
	# a = np.argpartition(a, -4)[-4:]
	print()

	# nUsers_with_max_cosine = users_tokens_df.loc[cos_sim_idx.flatten().max(), 'user_ip'].values
	nUsers_with_max_cosine = users_tokens_df.loc[cos_sim_idx.flatten().max(), 'user_ip']
	print(nUsers_with_max_cosine)
	


	print(cos_sim.flatten().max()[:N])
	print(cos_sim_idx.flatten().max()[:N])
	######################################################################3

	# ax.scatter(x=cos_sim_idx.flatten()[:N], y=cos_sim.flatten()[:N], facecolor='none', marker="o", edgecolors="r", s=100)
	ax.scatter(x=cos_sim_idx.flatten().max()[:N], y=cos_sim.flatten().max()[:N], facecolor='none', marker="o", edgecolors="r", s=100)
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

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_cosine_{QU.shape[1]}_nItems_{sp_type}_SP.png" ), bbox_inches='tight')
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
	if users_tokens_df.index.inferred_type == 'string':
		users_tokens_df = users_tokens_df.reset_index()#.rename(columns = {'index':'user_ip'})

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
	if df_usr_tk.index.inferred_type == 'string':
		df_usr_tk = df_usr_tk.reset_index()#.rename(columns = {'index':'user_ip'})

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
								aspect="auto",
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
	# sparse_df = pd.DataFrame(sparseMat.toarray(), index=users_tokens_df["user_ip"])
	sparse_df = users_tokens_df

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
	# print(topK_recommended_tokens)
	# print(recSysVec_indices)

	plt.rcParams["figure.subplot.right"] = 0.8
	f, ax = plt.subplots()	
	quTksLegends = []
	for ix, col in np.ndenumerate(qu_indices):
		# print(ix, col)
		sc1 = ax.scatter(	x=sparse_df.index, 
											y=sparse_df.iloc[:, col],
											label=f"{[k for k, v in bow.items() if v==col]} | {col}",
											marker="H",
											s=260,
											facecolor="none", 
											edgecolors=clrs[::-1][int(ix[0])],
										)
		quTksLegends.append(sc1)

	recLegends = []
	for ix, col in np.ndenumerate(np.flip(recSysVec_indices)):
		# print(ix, col)
		sc2 = ax.scatter(	x=sparse_df.index, 
											y=sparse_df.iloc[:, col],
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
	plt.xticks(	[i for i,_ in enumerate( users_tokens_df.index.values.tolist() ) if i%MODULE==0 ], 
							[f"{v}" for i, v in enumerate( users_tokens_df.index.values.tolist() ) if i%MODULE==0 ],
							rotation=90,
							fontsize=9.0,
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

def get_users_tokens_df():
	user_df_files = glob.glob( dfs_path+'/'+'*_user_df_*_BoWs.gz' )
	print(f">> concatinating {len(user_df_files)} user_df files:\n{user_df_files}")
	st_t = time.time()
	with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
		usr_tk_raw_dfs = pd.concat( [pd.concat( [df.user_ip, df.user_token_interest.apply(pd.Series).astype('float16')], axis=1 ) for f in glob.glob( dfs_path+'/'+'*_user_df_*_BoWs.gz' ) if ( df:=load_pickle(fpath=f) ).shape[0]>0  ] )
	
	print(f"Elapsed_t: {time.time()-st_t:.1f} s | "
				f"DF: {usr_tk_raw_dfs.shape} | "
				f"unq_users: {len( usr_tk_raw_dfs.user_ip.value_counts() )} | "
				f"unq_tokens: {len( list( usr_tk_raw_dfs.columns.difference(['user_ip'])))}".center(150, ' ')
			)
	print(f">> groupby...", end="\t")
	st_t = time.time()
	user_token_df = usr_tk_raw_dfs.groupby("user_ip").sum().astype("float16")
	print(f"Elapsed_t: {time.time()-st_t:.2f} s | DF: {user_token_df.shape}")
	# df_user_token_fname = os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_user_token_sparse_df_{len(user_token_df.columns)}_BoWs.gz")
	user_token_df_fname = os.path.join(dfs_path,f"{fprefix}_lemmaMethod_{args.lmMethod}_user_token_sparse_df_"
																							f"{len( usr_tk_raw_dfs.user_ip.value_counts() )}_nUSRs_x_"
																							f"{len(user_token_df.columns)}_nTKs.gz"
																		)
	save_pickle(pkl=user_token_df, fname=user_token_df_fname)
	save_vocab(	vb={c: i for i, c in enumerate(user_token_df.columns)}, 
							fname=os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_{len(user_token_df.columns)}_vocabs.json"),
						)
	user_token_df = user_token_df.reset_index().rename(columns = {'index': 'user_ip'})
	return user_token_df

def main():
	global fprefix, RES_DIR
	fprefix = f"dfs_concat"
	RES_DIR = make_result_dir(infile=fprefix)
	print(fprefix, RES_DIR)
	normalize_sp_mtrx = False
	topK=args.topTKs
	
	try:
		user_token_df = load_pickle( fpath=glob.glob( dfs_path+'/'+'*user_token_sparse_df_*_nUSRs_x_*_nTKs.gz' )[0] )
	except:
		user_token_df = get_users_tokens_df()

	# print(user_token_df.head(10))
	print(user_token_df)

	user_token_df.loc["ip4571", :].to_csv('out.csv')

	# return

	# print(user_token_df.loc["ip4571", :].head(25))
	# print("#"*100)
	# print(user_token_df.loc["ip4571", :].sort_index(ascending=False))
	# print("#"*100)

	# print(user_token_df.sort_index(ascending=False))
	# print("#"*100)
	pd.set_option('display.max_rows', 1000)
	pd.set_option('display.max_columns', 50)
	pd.set_option('display.width', 1000)
	print(user_token_df.loc["ip4571", :].sort_values(ascending=False, axis=0).head(500))
	print("#"*100)





	# return

	BoWs = {c: i for i, c in enumerate(user_token_df.columns.difference(['user_ip']))}
	
	try:
		sp_mat_rf = load_pickle(fpath=os.path.join(dfs_path, f"{fprefix}_lemmaMethod_{args.lmMethod}_user_token_sparse_matrix_{len(BoWs)}_BoWs.gz"))
	except:
		sp_mat_rf = get_sparse_matrix(user_token_df)

	plot_heatmap_sparse(sp_mat_rf, user_token_df, BoWs, norm_sp=normalize_sp_mtrx, ifb_log10=False)

	qu_phrase = args.qphrase
	query_phrase_tk = get_lemmatized_sqp(qu_list=[qu_phrase], lm=args.lmMethod)
	print(f"<> Raw Input Query Phrase:\t{qu_phrase}\tcontains {len(query_phrase_tk)} lemma(s):\t{query_phrase_tk}")
	query_vector = np.zeros(len(BoWs))
	for qutk in query_phrase_tk:
		#print(qutk, BoWs.get(qutk))
		if BoWs.get(qutk):
			query_vector[BoWs.get(qutk)] += 1.0

	print(f"<> queryVec in vocab\tAllzero: {np.all(query_vector==0.0)} "
				f"( |NonZeros|: {np.count_nonzero(query_vector)} "
				f"@ idx(s): {np.nonzero(query_vector)[0]} ) "
				f"TK(s): {[list(BoWs.keys())[list(BoWs.values()).index(vidx)] for vidx in np.nonzero(query_vector)[0]]}"
				# f"TK(s): {[k for idx in np.nonzero(query_vector)[0] for k, v in BoWs.items() if v==idx]}"
			)

	if np.all( query_vector==0.0 ):
		print(f"Sorry, We couldn't find tokenized words similar to {Fore.RED+Back.WHITE}{qu_phrase}{Style.RESET_ALL} in our BoWs! Search other phrases!")
		return

	# print(f"Getting users of {np.count_nonzero(query_vector)} token(s) / |QUE_TK|: {len(query_phrase_tk)}".center(120, "-"))
	# for iTK, vTK in enumerate(query_phrase_tk):
	# 	if BoWs.get(vTK):
	# 		users_names, users_values_total, users_values_separated = get_users_byTK(sp_mat_rf, user_token_df, BoWs, token=vTK)
	# 		plot_users_by(token=vTK, usrs_name=users_names, usrs_value_all=users_values_total, usrs_value_separated=users_values_separated, topUSRs=15, bow=BoWs, norm_sp=normalize_sp_mtrx )
	# 		plot_usersInterest_by(token=vTK, sp_mtrx=sp_mat_rf, users_tokens_df=user_token_df, bow=BoWs, norm_sp=normalize_sp_mtrx)

	cos_sim, cos_sim_idx = get_cs_sklearn(query_vector, sp_mat_rf.toarray(), qu_phrase, query_phrase_tk, user_token_df, norm_sp=normalize_sp_mtrx) # qu_ (nItems,) => (1, nItems) -> cos: (1, nUsers)
	# cos_sim, cos_sim_idx = get_cs_faiss(query_vector, sp_mat_rf.toarray(), qu_phrase, query_phrase_tk, user_token_df, norm_sp=normalize_sp_mtrx) # qu_ (nItems,) => (1, nItems) -> cos: (1, nUsers)

	print(f"Cosine Similarity (1 x nUsers): {cos_sim.shape} {type(cos_sim)} "
				f"Allzero: {np.all(cos_sim.flatten()==0.0)} "
				f"(min, max, sum): ({cos_sim.min()}, {cos_sim.max():.2f}, {cos_sim.sum():.2f})"
			)
	
	if np.all(cos_sim.flatten()==0.0):
		print(f"Sorry, We couldn't find similar results to >> {Fore.RED+Back.WHITE}{qu_phrase}{Style.RESET_ALL} << in our database! Search again!")
		return

	# plot_tokens_by_max(cos_sim, cos_sim_idx, sp_mtrx=sp_mat_rf, users_tokens_df=user_token_df, bow=BoWs, norm_sp=normalize_sp_mtrx)
	nUsers, nItems = sp_mat_rf.shape
	print(f"avgRecSysVec (1 x nItems) | nUsers: {nUsers} | nItems: {nItems}".center(120, " "))
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

	assert len(user_token_df.index.values.tolist()) == nUsers, f"df_idx: {len(user_token_df.index.values.tolist())} != nUsers: {nUsers}"

	st_t = time.time()
	for iUser in range(nUsers):
		idx_cosine = np.where(cos_sim_idx.flatten()==iUser)[0][0]
		#print(f"argmax(cosine_sim): {idx_cosine} => cos[uIDX: {iUser}] = {cos_sim[0, idx_cosine]}")
		if cos_sim[0, idx_cosine] != 0.0:
			# print(f"iUser[{iUser}]: {user_token_df.index.values.tolist()[iUser]}".center(120, " "))
			# print(f"avgrec (previous): {prev_avgrec.shape} "
			# 			f"(min, max_@(iTK), sum): ({prev_avgrec.min()}, {prev_avgrec.max():.5f}_@(iTK[{np.argmax(prev_avgrec)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(prev_avgrec) )]}), {prev_avgrec.sum():.1f}) "
			# 			f"{prev_avgrec} | Allzero: {np.all(prev_avgrec==0.0)}"
			# 		)			
			userInterest = sp_mat_rf.toarray()[iUser, :].reshape(1, -1) # 1 x nItems
			# print(f"<> userInterest[{iUser}]: {userInterest.shape} "
			# 			f"(min, max_@(iTK), sum): ({userInterest.min()}, {userInterest.max():.5f}_@(iTK[{np.argmax(userInterest)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(userInterest) )]}), {userInterest.sum():.1f}) "
			# 			f"{userInterest} | Allzero: {np.all(userInterest==0.0)}"
			# 		)
			userInterest_norm = np.linalg.norm(userInterest)
			userInterest = normalize(userInterest, norm="l2", axis=1)
			# print(f"<> userInterest(norm={userInterest_norm:.3f})[{iUser}]: {userInterest.shape} " 
			# 			f"(min, max_@(iTK), sum): ({userInterest.min()}, {userInterest.max():.5f}_@(iTK[{np.argmax(userInterest)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(userInterest) )]}), {userInterest.sum():.1f}) "
			# 			f"{userInterest} | Allzero: {np.all(userInterest==0.0)}"
			# 		)
			
			# if not np.all(userInterest==0.0):
			# 	print(f"\tFound {np.count_nonzero(userInterest.flatten())} non-zero userInterest element(s)")
			# 	print(f"\ttopK TOKENS     user[{iUser}]: {[list(BoWs.keys())[list(BoWs.values()).index( i )] for i in np.flip( np.argsort( userInterest.flatten() ) )[:10] ]}")
			# 	print(f"\ttopK TOKENS val user[{iUser}]: {[v for v in np.flip( np.sort( userInterest.flatten() ) )[:10] ]}")
			
			update_term = (cos_sim[0, idx_cosine] * userInterest)
			avgrec = prev_avgrec + update_term #avgrec = avgrec + (cos_sim[0, idx_cosine] * userInterest)
			prev_avgrec = avgrec
			
			# print(f"avgrec (current): {avgrec.shape} "
			# 			f"(min, max_@(iTK), sum): ({avgrec.min()}, {avgrec.max():.5f}_@(iTK[{np.argmax(avgrec)}]: {list(BoWs.keys())[list(BoWs.values()).index( np.argmax(avgrec) )]}), {avgrec.sum():.1f}) "
			# 			f"{avgrec} | Allzero: {np.all(avgrec==0.0)}"
			# 		)
			# print("-"*150)
			
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
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_avgRecSys_{nItems}_nitems.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

	print(f"idx:\n{avgrec.flatten().argsort()[-25:]}")
	print([k for i in avgrec.flatten().argsort()[-25:] for k, v in BoWs.items() if v==i ] )
	print(f">> sorted_recsys:\n{np.sort(avgrec.flatten())[-25:]}")
	st_t = time.time()
	# all_recommended_tks = [list(BoWs.keys())[list(BoWs.values()).index(vidx)] for vidx in avgrec.flatten().argsort() if vidx not in np.nonzero(query_vector)[0]]
	all_recommended_tks = list(map(list(BoWs.keys()).__getitem__, [list(BoWs.values()).index(vidx) for vidx in avgrec.flatten().argsort() if vidx not in np.nonzero(query_vector)[0]]))
	print(f"Elapsed_t: {time.time()-st_t:.2f} s |nTKs={len(all_recommended_tks)}|".center(120, " "))
	
	print(f"TOP-50 / {len(all_recommended_tks)}:\n{all_recommended_tks[-50:]}")
	topK_recommended_tokens = all_recommended_tks[-(topK+0):]
	print(f"top-{topK} recommended Tokens: {len(topK_recommended_tokens)}: {topK_recommended_tokens}")
	topK_recommended_tks_weighted_user_interest = [ avgrec.flatten()[BoWs.get(vTKs)] for iTKs, vTKs in enumerate(topK_recommended_tokens)]
	print(f"top-{topK} recommended Tokens weighted user interests: {len(topK_recommended_tks_weighted_user_interest)}: {topK_recommended_tks_weighted_user_interest}")
	#return
	"""
	st_t = time.time()
	get_selected_content(cos_sim, cos_sim_idx, topK_recommended_tokens, user_token_df)
	#print(f"Elapsed_t: {time.time()-st_t:.2f} s".center(120, " "))
	"""
	# get_nwp_cnt_by_nUsers_with_max(cos_sim, cos_sim_idx, sp_mat_rf, user_token_df, BoWs, recommended_tokens=topK_recommended_tokens, norm_sp=normalize_sp_mtrx)

	# print(f"Getting users of {len(topK_recommended_tokens)} tokens of top-{topK} RecSys".center(120, "-"))
	# for ix, tkv in enumerate(topK_recommended_tokens):
	# 	users_names, users_values_total, users_values_separated = get_users_byTK(sp_mat_rf, user_token_df, BoWs, token=tkv)
	# 	plot_users_by(token=tkv, usrs_name=users_names, usrs_value_all=users_values_total, usrs_value_separated=users_values_separated, topUSRs=15, bow=BoWs, norm_sp=normalize_sp_mtrx )
	# 	plot_usersInterest_by(token=tkv, sp_mtrx=sp_mat_rf, users_tokens_df=user_token_df, bow=BoWs, norm_sp=normalize_sp_mtrx)
	
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

	plot_tokens_distribution(sp_mat_rf, user_token_df, query_vector, avgrec, BoWs, norm_sp=normalize_sp_mtrx, topK=topK)

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