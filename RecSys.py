import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Colormap as cm
import spacy
from colorama import Fore, Style
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

from utils import *
from collections import Counter

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk_modules = ['punkt', 
               'averaged_perceptron_tagger', 
               'stopwords',
               'wordnet',
				'omw-1.4',
                ]
nltk.download('all',
              quiet=True, 
              raise_on_error=True,
              )

# Adapt stop words
#STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
my_custom_stopwords = ['btw', "could've", "n't","'s","—", "i'm", "'m", 
												"i've", "ive", "'d", "i'd", " i'll", "'ll", "'ll", "'re", "'ve", 
												'aldiz', 'baizik', 'bukatzeko', 
												'edota', 'eze', 'ezpabere', 'ezpada', 'ezperen', 'gainera', 
												'gainerontzean', 'guztiz', 'hainbestez', 'horra', 'onların', 'ordea', 
												'osterantzean', 'sha', 'δ', 'δι', 'агар-чи', 'аз-баски', 'афташ', 'бале', 
												'баҳри', 'болои', 'валекин', 'вақте', 'вуҷуди', 'гар', 'гарчанде', 'даме', 'карда', 
												'кошки', 'куя', 'кӣ', 'магар', 'майлаш', 'модоме', 'нияти', 'онан', 'оре', 'рӯи', 
												'сар', 'тразе', 'хом', 'хуб', 'чаро', 'чи', 'чунон', 'ш', 'шарте', 'қадар', 
												'ҳай-ҳай', 'ҳамин', 'ҳатто', 'ҳо', 'ҳой-ҳой', 'ҳол', 'ҳолате', 'ӯим', 'באיזו', 'בו', 'במקום', 
												'בשעה', 'הסיבה', 'לאיזו', 'למקום', 'מאיזו', 'מידה', 'מקום', 'סיבה', 'ש', 'שבגללה', 'שבו', 'תכלית', 'أفعل', 
												'أفعله', 'انفك', 'برح', 'سيما', 'कम', 'से', 'ἀλλ', '’',
												]
STOPWORDS.extend(my_custom_stopwords)
#print(len(STOPWORDS), STOPWORDS)
UNIQUE_STOPWORDS = set(STOPWORDS)
#print(len(UNIQUE_STOPWORDS), UNIQUE_STOPWORDS)


parser = argparse.ArgumentParser(description='National Library of Finland (NLF) RecSys')
parser.add_argument('--inputDF', default="~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump", type=str) # smallest
parser.add_argument('--qusr', default="ip69", type=str)
parser.add_argument('--qtip', default="Kristiinan Sanomat_77 A_1", type=str) # smallest
parser.add_argument('--qphrase', default="ystävä", type=str) # smallest

args = parser.parse_args()

# how to run:
# python RecSys.py --inputDF ~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

sz=13
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


PATTERN_S = re.compile("\'s")  # matches `'s` from text 
PATTERN_RN = re.compile("\\r\\n") #matches `\r` and `\n`
PATTERN_PUNC = re.compile(r"[^\w\s]") # matches all non 0-9 A-z whitespace 

def clean_text(text):
	text = text.lower().strip()
	
	text = re.sub(r"\r\n", "", text)

	#text = re.sub(r"(\n)\1{2,}", " ", text).strip()

	#text = "".join(i for i in text if ord(i)<128)
	#text = re.sub("[^A-Za-z0-9 ]+", "", text) # does not work with äöå...
	#text = re.sub("[^A-ZÜÖÄa-z0-9 ]+", "", text) # äöüÄÖÜß
	text = re.sub(r"\W+|_"," ", text) # replace special characters with space
	text = re.sub("\s+", " ", text)

	return text

def tokenizer(sentence, stopwords=UNIQUE_STOPWORDS, min_words=4, max_words=200, ):
	sentences = sentence.lower()

	wnl = nltk.stem.WordNetLemmatizer()

	tokens = [w for w in nltk.tokenize.word_tokenize(sentences)]
	filtered_tokens = [w for w in tokens if ( w not in stopwords and w not in string.punctuation )]
	lematized_tokens = [wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in nltk.pos_tag(filtered_tokens)]

	return lematized_tokens    

def clean_sentences(df):
	print(f'<> cleaning sentences of {df.shape}')
	df['clean_sentence'] = df['sentence'].apply(clean_text)
	df['tok_lem_sentence'] = df['clean_sentence'].apply(lambda x: tokenizer(x, 
																																					min_words=MIN_WORDS, 
																																					max_words=MAX_WORDS, 
																																					stopwords=STOPWORDS, 
																																					lemmatize=True,
																																				)
																										)
	print(f'<> after cleaned sentences: {df.shape}')
	return df

def extract_best_indices(m, topk, mask=None):
	"""
	Use sum of the cosine distance over all tokens.
	m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
	topk (int): number of indices to return (from high to lowest in order)
	"""
	# return the sum on all tokens of cosinus for each sentence
	if len(m.shape) > 1:
		cos_sim = np.mean(m, axis=0) 
	else: 
		cos_sim = m
	index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
	if mask is not None:
		assert mask.shape == m.shape
		mask = mask[index]
	else:
		mask = np.ones(len(cos_sim))
	mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
	best_index = index[mask][:topk]  
	return best_index

def get_TFIDF_RecSys_rest_api(dframe, qu_phrase="kirjasto", user_name=args.qusr, nwp_title_issue_page_name=args.qtip, topN=5):
	print(f"{'RecSys (TFIDF)'.center(80, '-')}")
	print(list(dframe["nwp_content_results"][4].keys()))
	#print(json.dumps(dframe["nwp_content_results"][4], indent=2, ensure_ascii=False))

	print(f">> Cleaning df: {dframe.shape} with NaN rows..")
	dframe = dframe.dropna(subset=["nwp_content_results"], how='all',).reset_index(drop=True)
	print(f">> Cleaned df: {dframe.shape}")
	fst_lst = [d.get("text") for d in  dframe.loc[:, "nwp_content_results"].values.flatten().tolist() if d.get("text")]
	"""
	fst_lst_cleaned = [clean_text(d.get("text")) for d in  dframe.loc[:, "nwp_content_results"].values.flatten().tolist() if d.get("text")]
	print(f"original".center(60,'-'))
	print(fst_lst[0])
	print("#"*100)
	print(f"clean".center(60,'-'))
	print(fst_lst_cleaned[0])
	print("<>"*100)
	print()

	print(f"original".center(60,'-'))
	print(fst_lst[1286])
	print("#"*100)
	print(f"clean".center(60,'-'))
	print(fst_lst_cleaned[1286])
	print("<>"*100)
	print()

	print(f"original".center(60,'-'))
	print(fst_lst[5429])
	print("#"*100)
	print(f"clean".center(60,'-'))
	print(fst_lst_cleaned[5429])
	print("<>"*100)
	print()
	"""
	fprefix = "_".join(args.inputDF.split("/")[-1].split(".")[:-2]) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	tfidf_vec_fpath = os.path.join(dfs_path, f"{fprefix}_tfidf_vectorizer.lz4")
	tfidf_rf_matrix_fpath = os.path.join(dfs_path, f"{fprefix}_tfidf_matrix_RF.lz4")

	if not os.path.exists(tfidf_rf_matrix_fpath):
		print(f"TFIDF for {len(fst_lst)} documents, might take a while...".center(110, " "))
		st_t = time.time()

		# Fit TFIDF # not time consuming...
		tfidf_vec = TfidfVectorizer(#min_df=5,
															#ngram_range=(1, 2),
															#tokenizer=Tokenizer(),
															tokenizer=tokenizer,
															stop_words=UNIQUE_STOPWORDS,
															)

		tfidf_matrix_rf = tfidf_vec.fit_transform(raw_documents=fst_lst)
		#tfidf_matrix_rf = np.random.choice(10_000, 10_000)

		save_tfidf_vec(tfidf_vec, fname=tfidf_vec_fpath)
		save_tfidf_matrix(tfidf_matrix_rf, fname=tfidf_rf_matrix_fpath)

		print(f"\t\tElapsed_t: {time.time()-st_t:.2f} s")
	else:
		tfidf_vec = load_tfidf_vec(fpath=tfidf_vec_fpath)
		tfidf_matrix_rf = load_tfidf_matrix(fpath=tfidf_rf_matrix_fpath)
	#return

	feat_names = tfidf_vec.get_feature_names_out()
	print(len(feat_names), feat_names[:50])
	
	vocabs = tfidf_vec.vocabulary_
	print(len(feat_names), len(vocabs))
	#print(json.dumps(vocabs, indent=2, ensure_ascii=False))

	# Embed qu_phrase
	tokens = [str(tok) for tok in tokenizer(qu_phrase)]
	print(f">> tokenize >> {qu_phrase} <<\t{len(tokens)} {tokens}")

	tfidf_matrix_qu = tfidf_vec.transform(tokens)
	print(f"RF: {tfidf_matrix_rf.shape}\tQU: {tfidf_matrix_qu.shape}")# (n_sample, n_vocab))
	#print(tfidf_matrix_qu.toarray())

	# Create list with similarity between query and dataset
	kernel_matrix = cosine_similarity(tfidf_matrix_qu, tfidf_matrix_rf)
	print(kernel_matrix.shape)

	# Best cosine distance for each token independantly
	best_index = extract_best_indices(kernel_matrix, topk=topN)
	print(best_index)
	print(f"> You searched for {qu_phrase}\tTop-{topN} Recommendations:")
	#return dframe[["query_word", ""]]
		


def get_TFIDF_RecSys(dframe, qu_phrase="kirjasto", user_name=args.qusr, nwp_title_issue_page_name=args.qtip, topN=5):
	print(f"{'RecSys (TFIDF)'.center(80, '-')}")
	
	print(f">> Cleaning df: {dframe.shape} with NaN rows..")
	dframe = dframe[~dframe["nwp_content_text"].isna()]
	print(f">> Cleaned df: {dframe.shape}")

	# Adapt stop words
	token_stop = tokenizer(' '.join(STOPWORDS), lemmatize=True) # orig: False
	print(f"tokenizer stop words: ({len(token_stop)})")
	st_t = time.time()
	# Fit TFIDF # not time consuming...
	tfidf_vec = TfidfVectorizer(#min_df=5,
																#ngram_range=(1, 2),
																#tokenizer=Tokenizer(),
																tokenizer=tokenizer,
																stop_words=token_stop,
																)
	#TODO: saving as a dump file is highly recommended!
	print(f">> Getting TFIDF RF matrix...")
	#print(dframe['nwp_content_text'].values)
	tfidf_matrix_rf = tfidf_vec.fit_transform(dframe['nwp_content_text'].astype(str).values)
	#tfidf_matrix_rf = tfidf_vec.fit_transform(dframe['clean_nwp_content_text'].values)#~XX sec!
	print(f"\t\tElapsed_t: {time.time()-st_t:.2f} s")

	feat_names = tfidf_vec.get_feature_names_out()

	vocabs = tfidf_vec.vocabulary_
	print(len(feat_names), len(vocabs))
	#print(feat_names[:50])
	#print(json.dumps(vocabs, indent=2, ensure_ascii=False))

	# Embed qu_phrase
	tokens = [str(tok) for tok in tokenizer(qu_phrase)]
	print(f">> tokenize >> {qu_phrase} <<\t{len(tokens)} {tokens}")

	tfidf_matrix_qu = tfidf_vec.transform(tokens)
	print(f"RF: {tfidf_matrix_rf.shape}\tQU: {tfidf_matrix_qu.shape}")# (n_sample, n_vocab))
	#print(tfidf_matrix_qu.toarray())

	# Create list with similarity between query and dataset
	kernel_matrix = cosine_similarity(tfidf_matrix_qu, tfidf_matrix_rf)
	print(kernel_matrix.shape)

	# Best cosine distance for each token independantly
	best_index = extract_best_indices(kernel_matrix, topk=topN)
	print(best_index)
	print(f"> You searched for {qu_phrase}\tTop-{topN} Recommendations:")
	#return dframe[["query_word", ""]]
	
	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1500):
		print(dframe[["query_word", 
									"nwp_content_title", 
									"nwp_content_issue", 
									"nwp_content_page", 
									"nwp_content_highlighted_term",
									"nwp_content_parsed_term",
									"referer",
									]
								].iloc[best_index]
					)
	
	return best_index

def get_basic_RecSys(df, user_name=args.qusr, nwp_title_issue_page_name=args.qtip):
	df_cleaned = df.dropna(axis=0, how="any", subset=["query_word"]).reset_index(drop=True)
	print(f">> Removed rows with None Query Phrases: {df_cleaned.shape}")

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
	
	st_t = time.time()
	usr_similarity_df = get_similarity_df(df_rec, imp_fb_sparse_matrix, method="user-based")
	print(f"<<>> User-based Similarity: {usr_similarity_df.shape}\tElapsed_t: {time.time()-st_t:.2f} s")
	topN_users(usr=user_name, sim_df=usr_similarity_df, dframe=df_cleaned)
	print("<>"*50)

	st_t = time.time()
	itm_similarity_df = get_similarity_df(df_rec, imp_fb_sparse_matrix.T, method="item-based")
	print(f"<<>> Item-based Similarity: {itm_similarity_df.shape}\tElapsed_t: {time.time()-st_t:.2f} s")

	#topN_nwp_title_issue_page("Karjalatar_135_2", itm_similarity_df)
	topN_nwp_title_issue_page(nwp_tip=nwp_title_issue_page_name, sim_df=itm_similarity_df)
	print("-"*70)

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

def get_similarity_df(df, sprs_mtx, method="user-based"):
	method_dict = {"user-based": "user_ip", 
								"item-based": "title_issue_page",
								}
	print(f">> Getting {method} similarity of sparse matrix: {sprs_mtx.shape} | {type(sprs_mtx)}...")

	similarity = cosine_similarity( sprs_mtx )
	#similarity = linear_kernel(sprs_mtx)
	
	plot_heatmap(mtrx=similarity.astype(np.float32), 
							name_=method,
							)

	sim_df = pd.DataFrame(similarity,#.astype(np.float32), 
												index=df[method_dict.get(method)].unique(),
												columns=df[method_dict.get(method)].unique(),
												)
	#print(sim_df.shape)
	#print(sim_df.info(verbose=True, memory_usage="deep"))
	#print(sim_df.head(25))
	#print("><"*60)

	return sim_df

def get_sparse_mtx(df):
	print(f"Getting Sparse Matrix: {df.shape}".center(80, '-'))
	print(list(df.columns))
	print(df.dtypes)
	print(f">> Checking positive indices?")
	assert np.all(df["user_index"] >= 0)
	assert np.all(df["nwp_tip_index"] >= 0)
	print(f">> Done!")

	sparse_mtx = csr_matrix( ( df["implicit_feedback"], (df["user_index"], df["nwp_tip_index"]) ), dtype=np.int8 ) # num, row, col
	#csr_matrix( ( data, (row, col) ), shape=(3, 3))
	##########################Sparse Matrix info##########################
	print("#"*110)
	print(f"Sparse: {sparse_mtx.shape} : |elem|: {sparse_mtx.shape[0]*sparse_mtx.shape[1]}")
	print(f"<> Non-zeros vals: {sparse_mtx.data}")# Viewing stored data (not the zero items)
	#print(sparse_mtx.toarray()[:25, :18])
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
		print(f"User `{usr}` not Found!\tTry another user_ip next time...")
		return
	
	print(f"{'Query USER Search History Phrases'.center(100,'-')}")
	qu_usr_search_history = get_query_user_details(usr_q=usr, dframe=dframe)
	if qu_usr_search_history is None:
		print(f"You have not searched for any specific words/phrases yet... "
					f"=> No recommendation is available at the moment! "
					f"Please continue searching...")
		return
	print(len(qu_usr_search_history), qu_usr_search_history)
	
	print(f"Top-{N} similar users to `{usr}`:")

	print(sim_df.head(20))
	print(f'<>'*50)

	similar_users = list(sim_df.sort_values(by=usr, ascending=False).index[1:]) # excluding usr
	similarity_values = list(sim_df.sort_values(by=usr, ascending=False).loc[:, usr])[1:] # excluding usr
	#similar_users = list(sim_df.sort_values(by=usr, ascending=False).index[1: N+1])
	#similarity_values = list(sim_df.sort_values(by=usr, ascending=False).loc[:, usr])[1: N+1]

	#print(sim_df.sort_values(by=usr, ascending=False).head(20))

	print(len(similar_users), similar_users[:20])
	print(len(similarity_values), similarity_values[:20])
	print("#"*100)
	
	print(f"{f'Similar USER(s) to {usr} | Detail'.center(100,'-')}")
	similar_users_search_phrases_history = get_similar_users_details(similar_users, similarity_values, dframe=dframe, TopN=N)

	print(f"Recommendation Result".center(100,'-'))
	print(f"Since you {Fore.RED}\033[1m{usr}\033[0m{Style.RESET_ALL} "
				f"searched for {len(qu_usr_search_history)} Query Phrase(s):\n"
				f"{Fore.BLUE}{qu_usr_search_history}{Style.RESET_ALL}\n"
				f"you might also be interested in Phrases:\n{Fore.GREEN}{similar_users_search_phrases_history}{Style.RESET_ALL}")
	print(f"Recommendation Result".center(100,'-'))
	
"""
# original implementation for prev dataframe:
def get_similar_users_details(sim_users_list, dframe, qu_usr=False):
	searched_phrases = list()
	print(list(dframe.columns))

	dframe = dframe.dropna(axis=0, how="any", subset=["search_query_phrase"]).reset_index(drop=True)
	for usr_i, usr_v in enumerate(sim_users_list):
		#print(usr_i, usr_v)
		#print(f"{'QUERY Phrases'.center(50,'-')}")
		with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1500):
			#print(dframe[["user_ip", "query_word", "referer", ]].head(50))
			#print(dframe.head(50))
			#print(dframe[(dframe["user_ip"] == usr_v)])
			#print()
			#print(dframe[(dframe["user_ip"] == usr_v)][["search_query_phrase"]])
			#print(f"<>"*10)

		qu_phs = ["".join(elem) for elem in dframe[(dframe["user_ip"] == usr_v)][["query_word"]].values.tolist()]

		#print(f"qu phrases: {qu_phs}")
		#print(max(set(qu_phs), key=qu_phs.count))
		#print(f"{'Search Results'.center(50,'-')}")
		#print(dframe[(dframe["user_ip"] == usr_v)][["search_results"]])
		#print()
		if qu_usr:
			searched_phrases.append(set(qu_phs))
		else:
			searched_phrases.append(max(set(qu_phs), key=qu_phs.count))
		print(f">> search hist: {searched_phrases}")
	return searched_phrases
"""

def get_similar_users_details(sim_users_list, similarity_vals, dframe, TopN=5):
	searched_phrases = list()
	retrieved_users = list()
	retrieved_similarity_values = list()

	for usr_i, usr_v in enumerate(sim_users_list):
		print(f"{'QUERY Phrases'.center(50,'-')}")
		print(usr_i, usr_v)
		qu_phs = ["".join(elem) for elem in dframe[(dframe["user_ip"] == usr_v)][["search_query_phrase"]].values.tolist()]
		print(f"All qu phrases: {len(qu_phs)}")
		unq_qu_phs_dict = Counter(qu_phs)
		print(json.dumps(unq_qu_phs_dict, indent=2, ensure_ascii=False))
		unq_qu_phs_dict.pop("", None)

		if len(unq_qu_phs_dict) > 0:
			#print(max(unq_qu_phs_dict, key=unq_qu_phs_dict.get))
			searched_phrases.append(max(unq_qu_phs_dict, key=unq_qu_phs_dict.get))
			retrieved_users.append(usr_v)
			retrieved_similarity_values.append(similarity_vals[usr_i])
		else:
			print(f"\t<!> Useless user!! Trying next user...")
		print()
		if len(searched_phrases) >= TopN:
			print(f"We found Top-{TopN} results => leave for loop..")
			break
	print(f">> search hist: {searched_phrases}")
	for sim_usr, sim_val, usr_hist in zip(retrieved_users, retrieved_similarity_values, searched_phrases):
		print(f"\t{sim_usr} : {sim_val:.4f}\t(Top Searched Query Phrase: {usr_hist})")
	print()

	return searched_phrases

def get_query_user_details(usr_q, dframe):
	qu_phs = ["".join(elem) for elem in dframe[(dframe["user_ip"] == usr_q)][["search_query_phrase"]].values.tolist()]
	print(f"|phrases|: {len(qu_phs)}")
	
	unq_qu_phs_dict = Counter(qu_phs)

	print(json.dumps(unq_qu_phs_dict, indent=2, ensure_ascii=False))
	unq_qu_phs_dict.pop("", None)
	if len(unq_qu_phs_dict) > 0:
		#print(max(unq_qu_phs_dict, key=unq_qu_phs_dict.get))
		#return max(unq_qu_phs_dict, key=unq_qu_phs_dict.get) # only the one with max occurance!
		return list(unq_qu_phs_dict.keys()) # all searched words/phrases!
	else:
		return

def get_snippet_hw_counts(results_list):
	return [ len(el.get("terms")) if el.get("terms") else 0 for ei, el in enumerate(results_list) ]

def get_content_hw_counts(results_dict):
	hw_count = 0
	if results_dict.get("highlighted_term"):
		hw_count = len(results_dict.get("highlighted_term"))
	return hw_count

def get_search_title_issue_page(results_list):
	return [f'{el.get("bindingTitle")}_{el.get("issue")}_{el.get("pageNumber")}' for ei, el in enumerate(results_list)]

def get_content_title_issue_page(results_dict):
	return f'{results_dict.get("title")}_{results_dict.get("issue")}_{results_dict.get("page")[0]}'

def get_basic_RecSys_rest_api(df, user_name=args.qusr, nwp_title_issue_page_name=args.qtip):
	#print(df.head(30))
	print(f"Search".center(80, "-"))
	df_search = pd.DataFrame()
	df_search["user_ip"] = df.loc[ df['search_results'].notnull(), ['user_ip'] ]

	df_search["search_query_phrase"] = df.loc[ df['search_results'].notnull(), ['search_query_phrase'] ]#.fillna("")
	#df_search['search_query_phrase'] = [','.join(map(str, elem)) if elem else '' for elem in df_search['search_query_phrase']]
	
	df_search["title_issue_page"] = df["search_results"].map(get_search_title_issue_page, na_action='ignore')
	df_search["snippet_highlighted_words"] = df["search_results"].map(get_snippet_hw_counts, na_action='ignore')
	df_search["referer"] = df.loc[ df['search_results'].notnull(), ['referer'] ]

	df_search = df_search.explode(["title_issue_page", "snippet_highlighted_words"],
																ignore_index=True,
																)
	df_search['snippet_highlighted_words'] = df_search['snippet_highlighted_words'].apply(pd.to_numeric)

	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1500):
		print(df_search.head(50))

	print("<>"*50)
	print(f"<Nones> tip: {df_search['title_issue_page'].isna().sum()} | "
				f"qu_phrases: {df_search['search_query_phrase'].isna().sum()} "
				f"total DF: {df_search.shape}")
	print(df_search.info(verbose=True, memory_usage="deep"))
	#return

	print(f"Newspaper Content".center(80, "-"))
	df_content = pd.DataFrame()
	df_content["user_ip"] = df.loc[ df['nwp_content_results'].notnull(), ['user_ip'] ]
	df_content["title_issue_page"] = df["nwp_content_results"].map(get_content_title_issue_page, na_action='ignore')
	df_content["content_highlighted_words"] = df["nwp_content_results"].map(get_content_hw_counts, na_action='ignore')
	df_content['content_highlighted_words'] = df_content['content_highlighted_words'].apply(pd.to_numeric)
	#df_content["referer"] = df.loc[ df['nwp_content_results'].notnull(), ['referer'] ]
	df_content = df_content.reset_index(drop=True)
	print(df_content.head(15))
	print(df_content.info(verbose=True, memory_usage="deep"))
	
	"""
	print(f"Merging".center(80, "-"))
	df_merged = pd.merge(df_search, # left
										df_content, # right
										how='outer', 
										#on=['user_ip','title_issue_page'],
										#on=['user_ip',],
										on=['title_issue_page',],
										suffixes=['_l', '_r'],
										)

	df_merged = df_merged.fillna({'snippet_highlighted_words': 0, 
													'content_highlighted_words': 0,
													}
												)

	df_merged["implicit_feedback"] = (0.5 * df_merged["snippet_highlighted_words"]) + df_merged["content_highlighted_words"]

	print(df_merged.shape)
	print(df_merged["title_issue_page"].isna().sum())
	print(df_merged.head(20))
	print(df_merged.info(verbose=True, memory_usage="deep"))
	print("<>"*100)
	
	#print(df_merged[df_merged['implicit_feedback'].notnull()].tail(60))
	print(f"< unique > tip: {len(df_merged['title_issue_page'].unique())}")
	print(f"< unique > user_ip_l: {len(df_merged['user_ip_l'].unique())}")
	print(f"< unique > user_ip_r: {len(df_merged['user_ip_r'].unique())}")
	"""

	print(f"Concatinating".center(80, "-"))
	df_concat = pd.concat([df_search, df_content],)

	df_concat = df_concat.fillna({'snippet_highlighted_words': 0, 
																'content_highlighted_words': 0,
																'search_query_phrase': '',
																}
															)
	df_concat['search_query_phrase'] = [','.join(map(str, elem)) for elem in df_concat['search_query_phrase']]

	df_concat["implicit_feedback"] = (0.5 * df_concat["snippet_highlighted_words"]) + df_concat["content_highlighted_words"]
	df_concat["implicit_feedback"] = df_concat["implicit_feedback"].astype(np.float32)

	df_concat["nwp_tip_index"] = df_concat["title_issue_page"].fillna('UNAVAILABLE').astype("category").cat.codes
	df_concat["user_index"] = df_concat["user_ip"].astype("category").cat.codes


	print(df_concat.shape)
	print(df_concat["title_issue_page"].isna().sum())
	print(df_concat.head(20))
	print(df_concat.info(verbose=True, memory_usage="deep"))
	print("<>"*100)

	print(f"< unique > users: {len(df_concat['user_ip'].unique())} | " 
				f"title_issue_page: {len(df_concat['nwp_tip_index'].unique())} "
				f"=> sparse matrix: {len(df_concat['user_index'].unique()) * len(df_concat['nwp_tip_index'].unique())}"
				)

	imp_fb_sparse_matrix = get_sparse_mtx(df_concat)
	


	st_t = time.time()
	usr_similarity_df = get_similarity_df(df_concat, imp_fb_sparse_matrix, method="user-based")
	print(f"<<>> User-based Similarity DF: {usr_similarity_df.shape}\tElapsed_t: {time.time()-st_t:.2f} s")

	topN_users(usr=user_name, sim_df=usr_similarity_df, dframe=df_concat)
	print("<>"*50)

	return
	st_t = time.time()
	itm_similarity_df = get_similarity_df(df_concat, imp_fb_sparse_matrix.T, method="item-based")
	print(f"<<>> Item-based Similarity DF: {itm_similarity_df.shape}\tElapsed_t: {time.time()-st_t:.2f} s")

	#topN_nwp_title_issue_page("Karjalatar_135_2", itm_similarity_df)
	topN_nwp_title_issue_page(nwp_tip=nwp_title_issue_page_name, sim_df=itm_similarity_df)
	print("-"*70)

def run_RecSys(df):
	#print(f"Running RecSys for DF: {df.shape}")
	print(f"{f'Running {__file__} for DF: {df.shape}'.center(80, '-')}")

	print(df.info(verbose=True, memory_usage="deep"))
	
	print("#"*100)
	#get_basic_RecSys(df, )
	#get_TFIDF_RecSys(qu_phrase=args.qphrase, dframe=df)
	
	
	#get_basic_RecSys_rest_api(df)
	get_TFIDF_RecSys_rest_api(qu_phrase=args.qphrase, dframe=df)

def main():
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

	run_RecSys(df)
	#return

if __name__ == '__main__':
	os.system("clear")
	main()