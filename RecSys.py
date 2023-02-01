import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Colormap as cm

import seaborn as sns

import matplotlib
matplotlib.use("Agg")

from utils import *
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='National Library of Finland (NLF) RecSys')
parser.add_argument('--inputDF', default="~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump", type=str) # smallest
parser.add_argument('--qusr', default="ip69", type=str)
parser.add_argument('--qtip', default="Kristiinan Sanomat_77 A_1", type=str) # smallest

args = parser.parse_args()

# how to run:
# python RecSys.py --inputDF ~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

sz=15
params = {
		'figure.figsize':	(sz*1.0, sz*0.7),  # W, H
		'figure.dpi':		200,
		'figure.autolayout': True,
		#'figure.constrained_layout.use': True,
		'legend.fontsize':	sz*0.8,
		'axes.labelsize':	sz*1.0,
		'axes.titlesize':	sz*1.0,
		'xtick.labelsize':	sz*0.8,
		'ytick.labelsize':	sz*0.8,
		'lines.linewidth' :	sz*0.1,
		'lines.markersize':	sz*0.8,
		'font.size':		sz*1.0,
		'font.family':		"serif",
	}
pylab.rcParams.update(params)

def plot_heatmap(mtrx, name_="user-based"):
	st_t = time.time()
	hm_title = f"{name_} similarity heatmap".capitalize()
	print(f"{hm_title.center(60,'-')}")
	print(type(mtrx), mtrx.shape, mtrx.nbytes)
	RES_DIR = make_result_dir(infile=args.inputDF)

	f, ax = plt.subplots()

	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	im = ax.imshow(mtrx, 
								cmap="viridis",#"magma", # https://matplotlib.org/stable/tutorials/colors/colormaps.html
								)
	cbar = ax.figure.colorbar(im,
														ax=ax,
														label="Similarity",
														orientation="vertical",
														cax=cax,
														ticks=[0.0, 0.5, 1.0],
														)

	ax.set_ylabel(f"{name_.split('-')[0].capitalize()}")
	#ax.set_yticks([])
	#ax.set_xticks([])
	ax.xaxis.tick_top()
	ax.tick_params(axis='x', labelrotation=90, labelsize=10.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=10.0)
	plt.suptitle(f"{hm_title}\n{mtrx.shape[0]} Unique Elements")
	#print(os.path.join( RES_DIR, f'{name_}_similarity_heatmap.png' ))
	plt.savefig(os.path.join( RES_DIR, f"{name_}_similarity_heatmap.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)
	print(f"{f'Elapsed_t: {time.time()-st_t:.3f} sec'.center(60, '-')}")

def analyze_search_results(df):
	print(f">> Analysing Search Results DF: {df.shape}")
	print(df.info(verbose=True, memory_usage="deep"))
	print("#"*200)
	#print(df[["user_ip", "query_word", "search_results", "nwp_content_parsed_term"]].head(50))
	#return
	"""
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
	topN_users(usr=args.qusr, sim_df=usr_similarity_df, dframe=df_cleaned)
	print("-"*70)

	itm_similarity_df = get_similarity_df(df_rec, imp_fb_sparse_matrix.T, method="item-based")
	#topN_nwp_title_issue_page("Karjalatar_135_2", itm_similarity_df)
	topN_nwp_title_issue_page(args.qtip, sim_df=itm_similarity_df)
	
	print("-"*70)

def get_similarity_df(df, sprs_mtx, method="user-based"):
	method_dict = {"user-based": "user_ip", 
								"item-based": "title_issue_page",
								}
	print(f">> Getting {method} similarity...")

	similarity = cosine_similarity(sprs_mtx)
	plot_heatmap(mtrx=similarity.astype(np.float32), 
							name_=method,
							)

	sim_df = pd.DataFrame(similarity.astype(np.float32), 
												index=df[method_dict.get(method)].unique(),
												columns=df[method_dict.get(method)].unique(),
												)
	print(sim_df.shape)
	#print(sim_df.info(verbose=True, memory_usage="deep"))
	print(sim_df.head(25))
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
		sim_df = sim_df.drop(nwp_tip)
		similar_newspapers = list(sim_df.sort_values(by=nwp_tip, ascending=False).index[1: N+1])
		similarity_values = list(sim_df.sort_values(by=nwp_tip, ascending=False).loc[:, nwp_tip])[1:N+1]
		for sim_nwp, sim_val in zip(similar_newspapers, similarity_values):
			print(f"\t{sim_nwp} : {sim_val:.3f}")

def topN_users(usr, sim_df, dframe, N=5):
		if usr not in sim_df.index:
				print(f"User `{usr}` not Found!\t")
				return
		print(f"Top-{N} similar users to `{usr}`:")

		#print(sim_df.sort_values(by=usr, ascending=False))
		#print(f">> dropin row: {usr} ...")
		sim_df = sim_df.drop(usr)

		#print(sim_df.sort_values(by=usr, ascending=False))
		#print("#"*100)
		#print(sim_df.sort_values(by=usr, ascending=False).index[:15])
		
		similar_users = list(sim_df.sort_values(by=usr, ascending=False).index[1: N+1])
		similarity_values = list(sim_df.sort_values(by=usr, ascending=False).loc[:, usr])[1: N+1]
		#print("#"*100)

		similar_users_search_history = get_similar_users_details(similar_users, dframe=dframe)
		qu_usr_search_history = get_similar_users_details([usr], dframe=dframe)

		#print(f"{'Query USER Search Phrase History'.center(100,'-')}")
		#print(len(qu_usr_search_history), qu_usr_search_history)

		#print(f"{'Similar USER Search Phrase History'.center(100,'-')}")
		#print(len(similar_users_search_history), similar_users_search_history)

		for sim_usr, sim_val, usr_hist in zip(similar_users, similarity_values, similar_users_search_history):
			print(f"\t{sim_usr} : {sim_val:.4f}\t(Top Searched Query Phrase: {usr_hist})")
		print(f"Since you {usr} Searched for Query Phrase {qu_usr_search_history}, "
					f"you might also be interested in {similar_users_search_history} Phrases...")
		
def get_similar_users_details(sim_users_list, dframe, qu_usr=False):
	word_search_history = list()
	for usr_i, usr_v in enumerate(sim_users_list):
		#print(usr_i, usr_v)
		#print(f"{'QUERY Phrases'.center(50,'-')}")
		qu_phs = ["".join(elem) for elem in dframe[(dframe["user_ip"] == usr_v)][["query_word"]].values.tolist()]
		#print(qu_phs)
		#print(max(set(qu_phs), key=qu_phs.count))
		#print(f"{'Search Results'.center(50,'-')}")
		#print(dframe[(dframe["user_ip"] == usr_v)][["search_results"]])
		#print()
		if qu_usr:
			word_search_history.append(set(qu_phs))
		else:
			word_search_history.append(max(set(qu_phs), key=qu_phs.count))

	return word_search_history

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