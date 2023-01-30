from utils import *
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='National Library of Finland (NLF) RecSys')
parser.add_argument('--inputDF', default="/home/xenial/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump", type=str) # smallest
args = parser.parse_args()

# how to run:
# python RecSys.py --inputDF /home/xenial/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

def analyze_search_results(df):
	print(f">> Analysing Search Results DF: {df.shape}")
	"""
	print(df.info(verbose=True, memory_usage="deep"))
	print("#"*200)
	#print(df[["user_ip", "query_word", "search_results", "nwp_content_parsed_term"]].head(50))
	#print("<>"*50)
	#print(df["user_ip"].value_counts())

	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1500):
		print(df[(df["user_ip"] == "ip3542")][["user_ip",
																					"timestamp",
																					"query_word", 
																					"nwp_content_parsed_term", 
																					"nwp_content_title", 
																					"nwp_content_issue", 
																					"nwp_content_page",
																					#"referer",
																					#"nwp_content_highlighted_term",
																					]])
	print("#"*200)
	for i in [382, 1251, 2410, 3502, ]: # query phrase: onnettomuus itäkylä
		tst_res = df.loc[i, "search_results"]
		for k, v in tst_res.items(): 
			print(i, 
						k, 
						tst_res.get(k).get("newspaper_title"), 
						tst_res.get(k).get("newspaper_issue"), 
						tst_res.get(k).get("newspaper_page"),
						#tst_res.get(k).get("newspaper_snippet_highlighted_words"),
						tst_res.get(k).get("newspaper_content_ocr_highlighted_words"),						
						)
		#print(json.dumps(tst_res.get("result_0"), indent=1, ensure_ascii=False) )
		print("<>"*50)
	"""
	#return	
	df_cleaned = df.dropna(axis=0, how="any", subset=["query_word"]).reset_index(drop=True)
	print(f">> Removed rows with None Query Phrases: {df_cleaned.shape}")
	#print(df_cleaned.info(verbose=True, memory_usage="deep"))
	#print("%"*90)	
	"""	
	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1100):
		print(df[(df["user_ip"] == "ip3542") & ((df["query_word"] == "onnettomuus itäkylä"))][["search_results"]])
	
	print(df_cleaned[["user_ip", "query_word", "search_results"]].head(50))
	print("<>"*50)

	print("#"*45)
	print(f"< unique > users: {len(df_cleaned['user_ip'].unique())} | query phrases: {len(df_cleaned['query_word'].unique())}")
	print("#"*45)
	idx = np.random.choice(df_cleaned.shape[0]+1)
	print(f"Ex) search results of sample: {idx}")
	
	with pd.option_context('display.max_colwidth', 500):
		print(df_cleaned.loc[idx, ["user_ip", "query_word", "referer"]])

	one_result = df_cleaned.loc[idx, "search_results"]
	
	#print(json.dumps(one_result, indent=1, ensure_ascii=False))
	print(f"{len(one_result)} results: {list(one_result.keys())}")
	print(list(one_result["result_0"].keys()))
	#print("-"*100)
	#return
	"""

	title_issue_pg = []
	usrs = []
	nwp_snippet_hw = []
	nwp_ocr_hw = []
	implicit_feedback = []
	for ii, row in df_cleaned.iterrows():
		if row["search_results"]:
			#print(ii)
			for k, v in row["search_results"].items():
				usrs.append(row["user_ip"])
				title_issue_pg.append(f"{row['search_results'].get(k).get('newspaper_title')}_{row['search_results'].get(k).get('newspaper_issue')}_{row['search_results'].get(k).get('newspaper_page')}")
				if row['search_results'].get(k).get('newspaper_snippet_highlighted_words'):
					nwp_snippet_hw.append(len(row['search_results'].get(k).get('newspaper_snippet_highlighted_words')))
				else:
					nwp_snippet_hw.append(0)
				if row['search_results'].get(k).get('newspaper_content_ocr_highlighted_words'):
					nwp_ocr_hw.append(len(row['search_results'].get(k).get('newspaper_content_ocr_highlighted_words')))
				else:
					nwp_ocr_hw.append(0)
				
	MY_DICT = {
		"user_ip": usrs,
		"title_issue_page": title_issue_pg,
		"snippet_highlighted_words": nwp_snippet_hw,
		"ocr_content_highlighted_words": nwp_ocr_hw,
	}

	df_rec = pd.DataFrame(MY_DICT)
	print(f"<> Creating implicit feedback: {df_rec.shape}")
		
	df_rec["implicit_feedback"] = (0.5 * df_rec["snippet_highlighted_words"] + df_rec["ocr_content_highlighted_words"])
	df_rec["nwp_tip_index"] = df_rec["title_issue_page"].astype("category").cat.codes
	df_rec["user_index"] = df_rec["user_ip"].astype("category").cat.codes

	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1100):
		print(df_rec.head(100))	
	print("*"*100)
	print(df_rec.info(verbose=True, memory_usage="deep"))

	print(f"< unique > users: {len(df_rec['user_index'].unique())} | " 
				f"title_issue_page: {len(df_rec['nwp_tip_index'].unique())} "
				f"=> sparse matrix: {len(df_rec['user_index'].unique()) * len(df_rec['nwp_tip_index'].unique())}")

	imp_fb_sparse_matrix = get_sparse_mtx(df_rec)
	
	usr_similarity_df = get_similarity_df(df_rec, imp_fb_sparse_matrix, method="user-based")
	topN_users("ip67", usr_similarity_df)	
	print("-"*70)

	itm_similarity_df = get_similarity_df(df_rec, imp_fb_sparse_matrix.T, method="item-based")
	topN_nwp_title_issue_page("Karjalatar_135_2", itm_similarity_df)
	print("-"*70)

def get_similarity_df(df, sprs_mtx, method="user-based"):
	method_dict = {"user-based": "user_ip", 
								"item-based": "title_issue_page",
								}
	print(f">> Getting {method} similarity...")

	# 

	similarity = cosine_similarity(sprs_mtx)

	print(type(similarity), similarity.shape)

	sim_df = pd.DataFrame(similarity,
												index=df[method_dict.get(method)].unique(),
												columns=df[method_dict.get(method)].unique(),
												)
	print(sim_df.shape)
	print(sim_df.head(10))
	print("><"*60)

	return sim_df

def get_sparse_mtx(df):
	sparse_mtx = csr_matrix( (df["implicit_feedback"], (df["user_index"], df["nwp_tip_index"],)) ) # num, row, col
	##########################Sparse Matrix info##########################
	print("#"*110)
	print(f"Sparse: {sparse_mtx.shape} : |elem|: {sparse_mtx.shape[0]*sparse_mtx.shape[1]}")
	print(f"<> Non-zeros vals: {sparse_mtx.data}")# Viewing stored data (not the zero items)
	print(sparse_mtx.toarray()[:25, :18])
	print(f"<> |Non-zero vals|: {sparse_mtx.count_nonzero()}") # Counting nonzeros
	print("#"*110)
	##########################Sparse Matrix info##########################
	return sparse_mtx

def topN_nwp_title_issue_page(nwp_tip, sim_df, N=10):
		if nwp_tip not in sim_df.index:
				print(f"Error: Newspaper `{nwp_tip}` not Found!")
				return
		print(f"Top-{N} Newspaper similar to `{nwp_tip}`:")
		similar_newspapers = list(sim_df.sort_values(by=nwp_tip, ascending=False).index[1: N+1])
		similarity_values = list(sim_df.sort_values(by=nwp_tip, ascending=False).loc[:, nwp_tip])[1:N+1]
		for sim_nwp, sim_val in zip(similar_newspapers, similarity_values):
			print(f"\t{sim_nwp} : {sim_val:.3f}")

def topN_users(usr, sim_df, N=10):
		if usr not in sim_df.index:
				print(f"Error: user `{usr}` not Found!")
				return
		print(f"Top-{N} users similar to `{usr}`:")
		similar_users = list(sim_df.sort_values(by=usr, ascending=False).index[1: N+1])
		similarity_values = list(sim_df.sort_values(by=usr, ascending=False).loc[:, usr])[1: N+1]
		for sim_usr, sim_val in zip(similar_users, similarity_values):
			print(f"\t{sim_usr} : {sim_val:.3f}")

def main():
	print(f">> Running {__file__}")
	df = load_df(infile=args.inputDF)
	"""
	print(f"DF: {df.shape}")
	print("%"*140)
	cols = list(df.columns)
	print(len(cols), cols)
	print("#"*150)

	print(df.head(10))
	print("-"*150)
	print(df.tail(10))

	print(df.isna().sum())
	print("-"*150)
	print(df[df.select_dtypes(include=[object]).columns].describe().T)
	"""

	analyze_search_results(df)
	#return

if __name__ == '__main__':
	os.system("clear")
	main()