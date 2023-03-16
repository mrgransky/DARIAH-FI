import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Colormap as cm
#import spacy
from colorama import Back, Fore, Style
import seaborn as sns

import matplotlib
matplotlib.use("Agg")

from utils import *
from collections import Counter

from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
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
#print(nltk.corpus.stopwords.words('finnish'))

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
parser.add_argument('--inputDF', default=f"{os.path.expanduser('~')}/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump", type=str) # smallest
parser.add_argument('--qusr', default="ip69", type=str)
parser.add_argument('--qtip', default="Kristiinan Sanomat_77 A_1", type=str) # smallest
parser.add_argument('--qphrase', default="ystävä", type=str) # smallest

args = parser.parse_args()

# how to run:
# python RecSys_XXXX.py --inputDF ~/Datasets/Nationalbiblioteket/dataframes/nikeY.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump

def spacy_tokenizer(sentence):
	sentences = sentence.lower()
	sentences = re.sub(r'[~|^|*][\d]+', '', sentences)

	lematized_tokens = [word.lemma_ for word in sp(sentences) if word.lemma_.lower() not in sp.Defaults.stop_words and word.is_punct==False and not word.is_space]
	
	return lematized_tokens

def nltk_tokenizer(sentence, stopwords=UNIQUE_STOPWORDS, min_words=4, max_words=200, ):	
	#print(sentence)
	wnl = nltk.stem.WordNetLemmatizer()

	sentences = sentence.lower()
	sentences = re.sub(r'"|<.*?>|[~|*|^][\d]+', '', sentences)
	sentences = re.sub(r"\W+|_"," ", sentences) # replace special characters with space
	sentences = re.sub("\s+", " ", sentences)

	# works not good: 
	#tokens = [w for w in nltk.tokenize.word_tokenize(sentences)]
	#filtered_tokens = [w for w in tokens if ( w not in stopwords and w not in string.punctuation )]

	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')	
	tokens = tokenizer.tokenize(sentences)
	filtered_tokens = [w for w in tokens if not w in UNIQUE_STOPWORDS]

	#lematized_tokens = [wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in nltk.pos_tag(filtered_tokens)]
	lematized_tokens = [wnl.lemmatize(word,tag[0].lower()) if tag[0].lower() in ['a', 's', 'r', 'n', 'v'] else wnl.lemmatize(word) for word,tag in nltk.pos_tag(filtered_tokens)]

	return lematized_tokens    

def get_qu_phrase_raw_text(phrase_list):
	assert len(phrase_list) == 1
	phrase = phrase_list[0]
	return phrase

def get_snippet_raw_text(search_results_list):
	#snippets_list = [sn.get("textHighlights").get("text") for sn in search_results_list if sn.get("textHighlights").get("text") ] # [["sentA"], ["sentB"], ["sentC"]]
	snippets_list = [sent for sn in search_results_list if sn.get("textHighlights").get("text") for sent in sn.get("textHighlights").get("text")] # ["sentA", "sentB", "sentC"]
	return ' '.join(snippets_list)

def get_complete_BoWs(dframe):
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
		lq = [phrases for phrases in g[g["query_phrase_raw_text"].notnull()]["query_phrase_raw_text"].values.tolist() if len(phrases) > 0]
		ls = [sentences for sentences in g[g["snippet_raw_text"].notnull()]["snippet_raw_text"].values.tolist() if len(sentences) > 0 ]
		lc = [sentences for sentences in g[g["ocr_raw_text"].notnull()]["ocr_raw_text"].values.tolist() if len(sentences) > 0 ]
		ltot = lq + ls + lc
		raw_texts_list.append( ltot )

	print(len(users_list), len(raw_texts_list),)
	df_usr_raw_texts = pd.DataFrame(list(zip(users_list, raw_texts_list,)), 
																	columns =['user_ip', 'raw_text', ])
	df_usr_raw_texts['raw_text'] = [ np.NaN if len(txt) == 0 else txt for txt in df_usr_raw_texts['raw_text'] ]
	
	print(df_usr_raw_texts.info())
	"""
	with pd.option_context('display.max_rows', 300, 'display.max_colwidth', 1500):
		print(df_usr_raw_texts.head(50))
	"""

	#raw_docs_list = df_usr_raw_texts.loc[df_usr_raw_texts["raw_text"].notnull(), "raw_text"].values.flatten().tolist()
	raw_docs_list = [subitem for item in df_usr_raw_texts.loc[df_usr_raw_texts["raw_text"].notnull(), "raw_text"].values.flatten().tolist() for subitem in item]

	print(len(raw_docs_list), type(raw_docs_list))

	fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	tfidf_vec_fpath = os.path.join(dfs_path, f"{fprefix}_tfidf_vectorizer_qu_phrases.lz4")
	tfidf_rf_matrix_fpath = os.path.join(dfs_path, f"{fprefix}_tfidf_matrix_RF_qu_phrases.lz4")
	vocab_json_file = os.path.join(dfs_path, f"{fprefix}_vocabs_qu_phrase.json")

	if not os.path.exists(tfidf_rf_matrix_fpath):
		print(f"Training TFIDF vector for {len(raw_docs_list)} raw words/phrases/sentences, might take a while...".center(110, " "))
		st_t = time.time()

		# Fit TFIDF # not time consuming...
		tfidf_vec = TfidfVectorizer(#min_df=5,
															#ngram_range=(1, 2),
															tokenizer=nltk_tokenizer,
															stop_words=UNIQUE_STOPWORDS,
															)

		tfidf_matrix_rf = tfidf_vec.fit_transform(raw_documents=raw_docs_list)
		#tfidf_matrix_rf = np.random.choice(10_000, 10_000)

		save_pickle(pkl=tfidf_vec, fname=tfidf_vec_fpath)
		save_pickle(pkl=tfidf_matrix_rf, fname=tfidf_rf_matrix_fpath)
		save_vocab(vb=tfidf_vec.vocabulary_, fname=vocab_json_file)

		print(f"\t\tElapsed_t: {time.time()-st_t:.2f} s")
	else:
		tfidf_vec = load_pickle(fpath=tfidf_vec_fpath)
		tfidf_matrix_rf = load_pickle(fpath=tfidf_rf_matrix_fpath)
	#return

	feat_names = tfidf_vec.get_feature_names_out()
	#print(f"1st 100 features:\n{feat_names[:60]}\n")
	
	BOWs = tfidf_vec.vocabulary_ # dict mapping: key: term value: column positions(indices) of features.
	# example:
	# vb = {"example": 0, "is": 1, "simple": 2, "this": 3}
	#	   		example   is         simple     this	
	# 0  		0.377964  0.377964   0.377964   0.377964

	print(f"Feature names: {feat_names.shape}\t{type(feat_names)}")
	print(f"BoWs: {len(BOWs)}\t{type(BOWs)}")

	#save_vocab(vb=BOWs, fname=vocab_json_file)

	return BOWs

def get_bag_of_words(dframe):
	print(f"{'Bag-of-Words'.center(80, '-')}")

	print(f"DF: {dframe.shape}\n{list(dframe.columns)}")
	print(f"Cleaning {dframe['search_query_phrase'].isna().sum()} NaN >> search_query_phrase << rows..")
	
	dframe = dframe.dropna(subset=["search_query_phrase"], how='all',).reset_index(drop=True)
	print(f"\tCleaned df: {dframe.shape}")
	
	dframe['search_query_phrase'] = [','.join(map(str, elem)) for elem in dframe['search_query_phrase']]
	
	raw_phrases_list = dframe.loc[:, "search_query_phrase"].values.flatten().tolist()

	print(raw_phrases_list)
	print(len(raw_phrases_list), type(raw_phrases_list))

	fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	tfidf_vec_fpath = os.path.join(dfs_path, f"{fprefix}_tfidf_vectorizer_qu_phrases.lz4")
	tfidf_rf_matrix_fpath = os.path.join(dfs_path, f"{fprefix}_tfidf_matrix_RF_qu_phrases.lz4")
	vocab_json_file = os.path.join(dfs_path, f"{fprefix}_vocabs_qu_phrase.json")

	if not os.path.exists(tfidf_rf_matrix_fpath):
		print(f"Training TFIDF vector for {len(raw_phrases_list)} query phrases, might take a while...".center(110, " "))
		st_t = time.time()

		# Fit TFIDF # not time consuming...
		tfidf_vec = TfidfVectorizer(#min_df=5,
															#ngram_range=(1, 2),
															tokenizer=nltk_tokenizer,
															stop_words=UNIQUE_STOPWORDS,
															)

		tfidf_matrix_rf = tfidf_vec.fit_transform(raw_documents=raw_phrases_list)
		#tfidf_matrix_rf = np.random.choice(10_000, 10_000)

		save_pickle(pkl=tfidf_vec, fname=tfidf_vec_fpath)
		save_pickle(pkl=tfidf_matrix_rf, fname=tfidf_rf_matrix_fpath)
		save_vocab(vb=tfidf_vec.vocabulary_, fname=vocab_json_file)

		print(f"\t\tElapsed_t: {time.time()-st_t:.2f} s")
	else:
		tfidf_vec = load_pickle(fpath=tfidf_vec_fpath)
		tfidf_matrix_rf = load_pickle(fpath=tfidf_rf_matrix_fpath)
	#return

	feat_names = tfidf_vec.get_feature_names_out()
	#print(f"1st 100 features:\n{feat_names[:60]}\n")
	
	BOWs = tfidf_vec.vocabulary_ # dict mapping: key: term value: column positions(indices) of features.
	# example:
	# vb = {"example": 0, "is": 1, "simple": 2, "this": 3}
	#	   		example   is         simple     this	
	# 0  		0.377964  0.377964   0.377964   0.377964

	print(f"Feature names: {feat_names.shape}\t{type(feat_names)}")
	print(f"BoWs: {len(BOWs)}\t{type(BOWs)}")

	#save_vocab(vb=BOWs, fname=vocab_json_file)

	return BOWs

def count_tokens_vocab(dframe, weights_list, vb):
	w_qu, w_hw_sn, w_sn, w_hw_cnt, w_pt_cnt, w_cnt = weights_list
	updated_vocab = vb.copy()

	for val in dframe["qu_tokens"]:
		if updated_vocab.get(val) is not None:
			updated_vocab[val] = updated_vocab.get(val) + w_qu
	
	for val in dframe["snippets_hw_tokens"]:
		if updated_vocab.get(val) is not None:
			updated_vocab[val] = updated_vocab.get(val) + w_hw_sn

	for val in dframe["snippets_tokens"]:
		if updated_vocab.get(val) is not None:
			updated_vocab[val] = updated_vocab.get(val) + w_sn
	
	for val in dframe["nwp_content_hw_tokens"]:
		if updated_vocab.get(val) is not None:
			updated_vocab[val] = updated_vocab.get(val) + w_hw_cnt

	for val in dframe["nwp_content_pt_tokens"]:
		if updated_vocab.get(val) is not None:
			updated_vocab[val] = updated_vocab.get(val) + w_pt_cnt
	
	for val in dframe["nwp_content_tokens"]:
		if updated_vocab.get(val) is not None:
			updated_vocab[val] = updated_vocab.get(val) + w_cnt

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
	#return [nltk_tokenizer( " ".join(res) )  for res in results_list]
	return [tklm for el in results_list for tklm in nltk_tokenizer(el)]

def tokenize_snippets(results_list):
	return [tklm for el in results_list for tklm in nltk_tokenizer(el)]

def tokenize_nwp_content(sentences):
	return nltk_tokenizer(sentences)

def tokenize_hw_nwp_content(results_list):
	return [tklm for el in results_list for tklm in nltk_tokenizer(el)]

def tokenize_pt_nwp_content(results_list):
	return [tklm for el in results_list for tklm in nltk_tokenizer(el)]

def tokenize_query_phrase(qu_list):
	# qu_list = ['some word in this format with always length 1']
	#print(len(qu_list), qu_list)
	assert len(qu_list) == 1
	return nltk_tokenizer(qu_list[0])

def get_usr_tk_df(dframe, bow):
	fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021

	try:
		df_preprocessed = load_pickle(fpath=os.path.join(dfs_path, f"{fprefix}_df_preprocessed.lz4"))
	except:
		print(f"Updating DF: {dframe.shape} with Tokenized texts".center(100, "-"))
		print(f">> Analyzing query phrases [tokenization + lemmatization]...")
		st_t = time.time()
		dframe["search_query_phrase_tklm"] = dframe["search_query_phrase"].map(tokenize_query_phrase , na_action="ignore")
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

		print(f">> Analyzing highlighted words in snippets [tokenization + lemmatization]...")
		st_t = time.time()
		dframe['search_results_hw_snippets'] = dframe["search_results"].map(get_search_results_hw_snippets, na_action='ignore')
		dframe['search_results_hw_snippets_tklm'] = dframe["search_results_hw_snippets"].map(tokenize_hw_snippets, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

		print(f">> Analyzing highlighted words in newspaper content [tokenization + lemmatization]...")
		st_t = time.time()
		dframe['nwp_content_ocr_text_hw'] = dframe["nwp_content_results"].map(get_nwp_content_hw, na_action='ignore')
		dframe['nwp_content_ocr_text_hw_tklm'] = dframe["nwp_content_ocr_text_hw"].map(tokenize_hw_nwp_content, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

		print(f">> Analyzing parsed terms in newspaper content [tokenization + lemmatization]...")
		st_t = time.time()
		dframe['nwp_content_ocr_text_pt'] = dframe["nwp_content_results"].map(get_nwp_content_pt, na_action='ignore')
		dframe['nwp_content_ocr_text_pt_tklm'] = dframe["nwp_content_ocr_text_pt"].map(tokenize_pt_nwp_content, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")

		"""
		#####################################################
		# comment for speedup:
		print(f">> Analyzing newspaper content [tokenization + lemmatization]...")
		st_t = time.time()
		dframe['nwp_content_ocr_text'] = dframe["nwp_content_results"].map(get_nwp_content_raw_text, na_action='ignore')
		dframe['nwp_content_ocr_text_tklm'] = dframe["nwp_content_ocr_text"].map(tokenize_nwp_content, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		
		print(f">> Analyzing snippets [tokenization + lemmatization]...")
		st_t = time.time()
		dframe['search_results_snippets'] = dframe["search_results"].map(get_search_results_snippet_text, na_action='ignore')
		dframe['search_results_snippets_tklm'] = dframe["search_results_snippets"].map(tokenize_snippets, na_action='ignore')
		print(f"\tElapsed_t: {time.time()-st_t:.2f} s")
		#####################################################
		"""

		dframe_preprocessed_fname = os.path.join(dfs_path, f"{fprefix}_df_preprocessed.lz4")
		save_pickle(pkl=dframe, fname=dframe_preprocessed_fname)
		df_preprocessed = dframe.copy()

	print(df_preprocessed.info())
	print(df_preprocessed.tail(60))
	print(f"<>"*120)
	
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
		#search_results_snippets_tokens_list.append( [tk for tokens in g[g["search_results_snippets_tklm"].notnull()]["search_results_snippets_tklm"].values.tolist() if tokens for tk in tokens if tk] )
		#nwp_content_tokens_list.append([tk for tokens in g[g["nwp_content_ocr_text_tklm"].notnull()]["nwp_content_ocr_text_tklm"].values.tolist() if tokens for tk in tokens if tk] )

	# uncomment for speedup:
	nwp_content_tokens_list = [f"nwp_content_{i}" for i in range(len(users_list))]
	search_results_snippets_tokens_list = [f"snippet_{i}" for i in range(len(users_list))]

	print(len(users_list), 
				len(search_query_phrase_tokens_list),
				len(search_results_hw_snippets_tokens_list),
				len(search_results_snippets_tokens_list),
				len(nwp_content_tokens_list),
				len(nwp_content_pt_tokens_list),
				len(nwp_content_hw_tokens_list),
				)
	#return

	usr_interest_vb = dict.fromkeys(bow.keys(), 0.0)
	print(f"USER-TOKENS DF".center(60, "-"))
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
	
	# list of all weights:
	weightQueryAppearance = 1.0 							# suggested by Jakko: 1.0
	weightSnippetAppearance = 0.25 						# suggested by Jakko: 0.2
	weightSnippetHighlightAppearance = 0.4 		# suggested by Jakko: 0.2
	weightContentAppearance = 0.05 						# suggested by Jakko: 0.05
	weightContentHighlightAppearance = 0.05 	# suggested by Jakko: 0.05
	weightContentParsedAppearance = 0.005			# Did not consider!
	
	w_list = [weightQueryAppearance, 
						weightSnippetHighlightAppearance,
						weightSnippetAppearance,
						weightContentHighlightAppearance,
						weightContentParsedAppearance,
						weightContentAppearance,
					]

	print(f">> Creating Implicit Feedback for user interests")
	df_user_token["user_token_interest"] = df_user_token.apply( lambda x_df: count_tokens_vocab(x_df, w_list, usr_interest_vb.copy()), axis=1, )
	
	print(df_user_token.shape, list(df_user_token.columns))

	print(df_user_token.info())

	df_user_token_fname = os.path.join(dfs_path, f"{fprefix}_user_tokens_df.lz4")
	save_pickle(pkl=df_user_token, fname=df_user_token_fname)

	"""
	with open("ip3540.json", "w") as fw:
		json.dump(df_user_token[df_user_token["user_ip"]=="ip3540"]["user_token_interest"].tolist(), fw, indent=4, ensure_ascii=False)

	with open("ip6843.json", "w") as fw:
		json.dump(df_user_token[df_user_token["user_ip"]=="ip6843"]["user_token_interest"].tolist(), fw, indent=4, ensure_ascii=False)

	with open("ip3539.json", "w") as fw:
		json.dump(df_user_token[df_user_token["user_ip"]=="ip3539"]["user_token_interest"].tolist(), fw, indent=4, ensure_ascii=False)
	"""

	return df_user_token

def get_sparse_matrix(df):
	print(f"Sparse Matrix | DF: {df.shape}".center(80, '-'))
	print(list(df.columns))
	print(df.dtypes)
	df_new = pd.concat( [df["user_ip"], df['user_token_interest'].apply(pd.Series)], axis=1).set_index("user_ip")

	sparse_matrix = csr_matrix(df_new.values, dtype=np.float32)
	print(sparse_matrix.shape, type(sparse_matrix))

	##########################Sparse Matrix info##########################
	print("#"*110)
	print(f"Sparse: {sparse_matrix.shape} : |elem|: {sparse_matrix.shape[0]*sparse_matrix.shape[1]}")
	print(f"<> Non-zeros vals: {sparse_matrix.data}")# Viewing stored data (not the zero items)
	print(sparse_matrix.toarray()[:25, :18])
	print(f"<> |Non-zero vals|: {sparse_matrix.count_nonzero()}") # Counting nonzeros
	print("#"*110)
	##########################Sparse Matrix info##########################
	fprefix = get_filename_prefix(dfname=args.inputDF) # nikeY_docworks_lib_helsinki_fi_access_log_07_02_2021
	sp_mat_user_token_fname = os.path.join(dfs_path, f"{fprefix}_user_tokens_sparse_matrix.lz4")

	save_pickle(pkl=sparse_matrix, fname=sp_mat_user_token_fname)

	return sparse_matrix

def run_RecSys(df_inp, qu_phrase, topK=5):
	print_df_detail(df=df_inp, fname=__file__)
	
	BoWs = get_bag_of_words(dframe=df_inp)
	#BoWs = get_complete_BoWs(dframe=df_inp)
	#return

	try:
		df = load_pickle(fpath=os.path.join(dfs_path, f"{get_filename_prefix(dfname=args.inputDF)}_user_tokens_df.lz4"))
	except:
		df = get_usr_tk_df(dframe=df_inp, bow=BoWs)
	
	print(list(df.columns), df.shape)
	print(df.info())

	try:
		sp_mat_rf = load_pickle(fpath=os.path.join(dfs_path, f"{get_filename_prefix(dfname=args.inputDF)}_user_tokens_sparse_matrix.lz4"))
	except:
		sp_mat_rf = get_sparse_matrix(df)

	print(f"{type(sp_mat_rf)}: {sp_mat_rf.shape}")

	# Embed qu_phrase
	#tokens = [str(tok) for tok in my_tokenizer(qu_phrase)]
	query_phrase_tk = tokenize_query_phrase(qu_list=[qu_phrase])
	print(f"Tokenize(query phrase: >> {qu_phrase} <<) => {len(query_phrase_tk)} {query_phrase_tk}")
	query_vector = np.zeros((1, len(BoWs)))

	for qutk in query_phrase_tk:
		if BoWs.get(qutk):
			query_vector[0, BoWs.get(qutk)] += 1.0 
	
	print(f"query {query_vector.shape}: {query_vector.flatten()} Allzero: {np.all(query_vector.flatten() == 0.0)}") 

	print(f"QUERY_VECTOR: {query_vector.shape} REFERENCE_SPARSE_MATRIX: {sp_mat_rf.shape}") # QU: (1, n_vocabs) | RF: (n_usr, n_vocab) 

	kernel_matrix = cosine_similarity(query_vector, sp_mat_rf) # (1, n_usr)
	
	print(f"kernel{kernel_matrix.shape}: {kernel_matrix.flatten()} Allzero: {np.all(kernel_matrix.flatten() == 0.0)}")
	if np.all(kernel_matrix.flatten() == 0.0):
		print(f"Sorry, We couldn't find similar results to >> {Fore.RED+Back.WHITE}{args.qphrase}{Style.RESET_ALL} << in our database! Search again!")
		return

	#nUsers = 5
	#nItems = 11
	nUsers, nItems = sp_mat_rf.toarray().shape
	#print(f"Users: {nUsers} vs. Tokenzied word Items: {nItems}")
	#cos = np.random.rand(nUsers).reshape(1, -1)
	#usr_itm = np.random.randint(100, size=(nUsers, nItems))
	avgrec = np.zeros((1,nItems))
	#print(f"> avgrec{avgrec.shape}:\n{avgrec}")
	#print()
	
	#print(f"> cos{cos.shape}:\n{cos}")
	#print()

	#print(f"> user-item{usr_itm.shape}:\n{usr_itm}")
	#print("#"*100)
	st_t = time.time()
	for iUser in range(nUsers):
		#print(f"USER: {iUser}")
		#print()
		userInterest = sp_mat_rf.toarray()[iUser, :]
		#print(f"<> userInterest{userInterest.shape}:\n{userInterest}")
		#userInterest = userInterest / np.linalg.norm(userInterest)
		userInterest = np.where(np.linalg.norm(userInterest) != 0, userInterest/np.linalg.norm(userInterest), 0.0)
		#print(f"<> userInterest_norm{userInterest.shape}:\n{userInterest}")
		#print(f"cos[{iUser}]: {kernel_matrix[0, iUser]}")
		#print(f"<> avgrec (B4):{avgrec.shape}\n{avgrec}")
		avgrec = avgrec + kernel_matrix[0, iUser] * userInterest
		#print(f"<> avgrec (After):{avgrec.shape}\n{avgrec}")
		#print()

	#print(f"-"*100)
	avgrec = avgrec / np.sum(kernel_matrix)
	#print(f"avgrec:{avgrec.shape}\n{avgrec}")
	
	#all_topk_match_indeces = avgrec.flatten().argsort()#[-(topK+1):-1]
	#all_topk_matches = np.sort(avgrec.flatten())#[-(topK+1):-1]
	#print(f"ALL top-{topK} idx: {all_topk_match_indeces}\nALL top-{topK} res: {all_topk_matches}")

	# to exclude the query words:
	topk_match_indeces = avgrec.flatten().argsort()[-(topK+1):-1]
	topk_matches = np.sort(avgrec.flatten())[-(topK+1):-1]

	#topk_match_indeces = avgrec.flatten().argsort()[-topK:]
	#topk_matches = np.sort(avgrec.flatten())[-topK:]
	
	#print(f"top-{topK} idx: {topk_match_indeces}\ntop-{topK} res: {topk_matches}")
	topk_recom_tks = [k for idx in topk_match_indeces for k, v in BoWs.items() if v == idx]
	print(f"\t\tElapsed_t: {time.time()-st_t:.2f} s")

	print()
	print(f"Implicit Feedback Recommendation: {f'Unique Users: {nUsers} vs. Tokenzied word Items: {nItems}'}".center(100,'-'))
	print(f"Since you searched for query phrase(s):\t{Fore.BLUE+Back.YELLOW}{args.qphrase}{Style.RESET_ALL}"
				f"\tTokenized + Lemmatized: {query_phrase_tk}\n"
				f"you might also be interested in Phrases:\n{Fore.GREEN}{topk_recom_tks[::-1]}{Style.RESET_ALL}")
	print()
	print(f"{f'Top-{topK} Tokens' : <20}{'Similarity Value' : ^20}")
	for tk, sim_val in zip(topk_recom_tks[::-1], topk_matches[::-1]):
		print(f"{tk : <20} {sim_val:^{20}.{3}f}")
	print()
	print(f"Implicit Feedback Recommendation: {f'Unique Users: {nUsers} vs. Tokenzied word Items: {nItems}'}".center(100,'-'))

def main():
	df_raw = load_df(infile=args.inputDF)
	run_RecSys(df_inp=df_raw, qu_phrase=args.qphrase)
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
	topk_match_indeces = avgrec.flatten().argsort()#[-(topK+1):-1]
	topk_matches = np.sort(avgrec.flatten())#[-(topK+1):-1]

	print(f"top-{topK} idx: {topk_match_indeces}\ntop-{topK} res: {topk_matches}")

if __name__ == '__main__':
	os.system("clear")
	main()
	#practice()
	print()