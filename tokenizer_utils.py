from utils import *

# Define the global MultilingualPipeline object
smp = None

# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
with HiddenPrints():
	import nltk

	# nltk_modules = [
	# 	'punkt',
	# 	'wordnet',
	# 	'averaged_perceptron_tagger', 
	# 	'omw-1.4',
	# 	'stopwords',
	# ]

	# nltk.download(
	# 	# 'all',
	# nltk_modules,
	# 'stopwords',
	# 	quiet=True,
	# 	# raise_on_error=True,
	# )

	import stanza
	from stanza.pipeline.multilingual import MultilingualPipeline
	from stanza.pipeline.core import DownloadMethod

	useless_upos_tags = [
		"PUNCT",
		"CCONJ",
		"SCONJ",
		"SYM",
		"AUX",
		"NUM",
		"DET",
		"ADP",
		"PRON",
		"PART",
		"ADV",
		"INTJ",
		# "X", # foriegn words will be excluded,
	]

	STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
	with open('meaningless_lemmas.txt', 'r') as file_:
		my_custom_stopwords=[line.strip() for line in file_]
	STOPWORDS.extend(my_custom_stopwords)
	UNQ_STW = list(set(STOPWORDS))

# Function to create the MultilingualPipeline object if not already created
def create_multilingual_pipeline(device: str):
	global smp
	if smp is None:
		lang_id_config={
			"langid_lang_subset": [
				'fi', 
				'sv', 
				'en',
				'da',
				# 'nb', 
				'ru',
				'et',
				'de',
				'fr',
			]
		}

		lang_configs = {
			"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
			# "fi": {"processors":"tokenize,lemma,pos,mwt", "package":'tdt',"tokenize_no_ssplit":True}, # TDT
			"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'ftb',"tokenize_no_ssplit":True}, # FTB
			"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
			# "sv": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True}, # errors!!!
			"da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
			# "nb": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
			"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
			"et": {"processors":"tokenize,lemma,pos", "package":'edt',"tokenize_no_ssplit":True},
			"de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True},
			"fr": {"processors":"tokenize,lemma,pos,mwt", "package":'sequoia',"tokenize_no_ssplit":True},
		}
		print(f"Creating Stanza[{stanza.__version__}] < {device} MultilingualPipeline >", end=" ")
		tt = time.time()
		# Create the MultilingualPipeline object
		smp = MultilingualPipeline( 
			lang_id_config=lang_id_config,
			lang_configs=lang_configs,
			download_method=DownloadMethod.REUSE_RESOURCES,
			device=device,
		)
		print(f"Elapsed_t: {time.time()-tt:.3f} sec")

@cache
def stanza_lemmatizer(docs: str="This is a <NORMAL> document!", device=None):
	# Ensure MultilingualPipeline object is created
	create_multilingual_pipeline(device=device)
	try:
		print(f'Stanza[{stanza.__version__} device: {device}] Raw Input:\n{docs}\n')
		st_t = time.time()
		all_ = smp(docs)
		lemmas_list = [ 
			re.sub(r'[";=&#<>_\-\+\^\.\$\[\]]', '', wlm.lower())
			for _, vsnt in enumerate(all_.sentences) 
			for _, vw in enumerate(vsnt.words) 
			if (
				(wlm:=vw.lemma)
				and 5 <= len(wlm) <= 43
				and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<ros>|<eos>|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\^|\s+', wlm) 
				and vw.upos not in useless_upos_tags 
				and wlm not in UNQ_STW
			)
		]
		end_t = time.time()
	except Exception as e:
		print(f"<!> Stanza Error: {e}")
		return
	print( lemmas_list )
	print(f"Found {len(lemmas_list)} lemma(s) in {end_t-st_t:.2f} s".center(140, "-") )
	return lemmas_list

def trankit_lemmatizer(docs, device: str="cuda:0"):
	# print(f'Raw: (len: {len(docs)}) >>{docs}<<')
	# print(f'Raw inp words: { len( docs.split() ) }', end=" ")
	st_t = time.time()
	if not docs:
		return

	# treat all as document
	docs = re.sub(r'\"|<[^>]+>|[~*^][\d]+', '', docs)
	docs = re.sub(r'[%,+;,=&\'*"°^~?!—.•()“”:/‘’<>»«♦■\\\[\]-]+', ' ', docs ).strip()
	
	# print(f'preprocessed: len: {len(docs)}:\n{docs}')
	if ( not docs or len(docs)==0 ):
		return

	all_dict = p(docs)
	#lm = [ tk.get("lemma").lower() for sent in all_dict.get("sentences") for tk in sent.get("tokens") if ( tk.get("lemma") and len(re.sub(r'\b[A-Z](\.| |\:)+|\b[a-z](\.| |\:)+', '', tk.get("lemma") ) ) > 2 and tk.get("upos") not in useless_upos_tags and tk.get("lemma").lower() not in UNQ_STW ) ] 
	lm = [ tk.get("lemma").lower() for sent in all_dict.get("sentences") for tk in sent.get("tokens") if ( tk.get("lemma") and len(re.sub(r'\b[A-Za-z](\.| |:)+', '', tk.get("lemma") ) ) > 2 and tk.get("upos") not in useless_upos_tags and tk.get("lemma").lower() not in UNQ_STW ) ]
	print(f"Elapsed_t: {time.time()-st_t:.3f} sec")
	# print( lm )
	return lm

def spacy_tokenizer(docs, device: str="cuda:0"):
	return None

def nltk_lemmatizer(sentence, device: str="cuda:0"):	
	#print(f'Raw inp ({len(sentence)}): >>{sentence}<<', end='\t')
	if not sentence:
		return
	wnl = nltk.stem.WordNetLemmatizer()

	sentences = sentence.lower()
	sentences = re.sub(r'"|<.*?>|[~|*|^][\d]+', '', sentences)
	sentences = re.sub(r'\b[A-Z](\.| |\:)+|\b[a-z](\.| |\:)+', '', sentences)
	sentences = re.sub(r'["]|[+]|[*]|”|“|\s+|\d', ' ', sentences).strip() # strip() removes leading (spaces at the beginning) & trailing (spaces at the end) characters
	#print(f'preprocessed: {len(sentences)} >>{sentences}<<', end='\t')

	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentences)

	filtered_tokens = [w for w in tokens if not w in UNQ_STW and len(w) > 1 and not w.isnumeric() ]
	# nltk.pos_tag: cheatsheet: pg2: https://computingeverywhere.soc.northwestern.edu/wp-content/uploads/2017/07/Text-Analysis-with-NLTK-Cheatsheet.pdf
	lematized_tokens = [wnl.lemmatize(w, t[0].lower()) if t[0].lower() in ['a', 's', 'r', 'n', 'v'] else wnl.lemmatize(w) for w, t in nltk.pos_tag(filtered_tokens)] 
	#print( list( set( lematized_tokens ) ) )

	return lematized_tokens