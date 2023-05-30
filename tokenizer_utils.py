from utils import *

#with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
with HiddenPrints():
	import nltk
	nltk_modules = ['punkt', 
								'averaged_perceptron_tagger', 
								'stopwords',
								'wordnet',
								'omw-1.4',
								]
	nltk.download(#'all',
								nltk_modules,
								quiet=True, 
								raise_on_error=True,
								)

	import trankit
	p = trankit.Pipeline('finnish-ftb', embedding='xlm-roberta-large', cache_dir=os.path.join(NLF_DATASET_PATH, 'trash'))
	p.add('swedish')
	p.add('russian')
	#p.add('english')
	#p.add('estonian')
	p.set_auto(True)

	# load stanza imports
	import stanza
	from stanza.pipeline.multilingual import MultilingualPipeline
	from stanza.pipeline.core import DownloadMethod

	lang_id_config = {"langid_lang_subset": ['fi', 'sv', 'ru']}
	lang_configs = {"en": {"processors": "tokenize,lemma,pos,depparse"},
											"ru": {"processors": "tokenize,lemma,pos,depparse"},
											"sv": {"processors": "tokenize,lemma,pos,depparse"},
											"fi": {	"processors": "tokenize,lemma,pos,depparse,mwt", 
															"package": 		'ftb',
														},
								}
	stanza_multi_pipeline = MultilingualPipeline(	lang_id_config=lang_id_config, 
																							use_gpu=True,
																							lang_configs=lang_configs,
																							download_method=DownloadMethod.REUSE_RESOURCES,
																					)

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
UNIQUE_STOPWORDS = list(set(STOPWORDS))
#print(f"Unique Stopwords: {len(UNIQUE_STOPWORDS)} | {type(UNIQUE_STOPWORDS)}\n{UNIQUE_STOPWORDS}")
useless_upos_tags = ["PUNCT", "CCONJ", "SYM", "AUX", "NUM", "DET", "ADP", "PRON", "PART", "ADV", "INTJ"]

def spacy_tokenizer(sentence):
	sentences = sentence.lower()
	sentences = re.sub(r'[~|^|*][\d]+', '', sentences)

	lematized_tokens = [word.lemma_ for word in sp(sentences) if word.lemma_.lower() not in sp.Defaults.stop_words and word.is_punct==False and not word.is_space]
	
	return lematized_tokens

def stanza_lemmatizer(docs):
	#print(f'Raw: (len: {len(docs)}) >>{docs}<<')
	if not docs:
		return
	# treat all as document
	#docs = re.sub(r'\"|<.*?>|[~|*|^][\d]+', '', docs)
	# docs = re.sub(r'[%|+|,|;|=|&|\'|*|"|^|~|?|!|—|.|°|•||\(\)|\”\“|:|/|‘|’|<|>|»|„|«|♦|■|\\|\[\]|-]+', ' ', docs ).strip()
	docs = re.sub(r'\"|<[^>]+>|[~*^][\d]+', '', docs)
	docs = re.sub(r'[%,+;,=&\'*"°^~?!—.•()“”:/‘’<>»«♦■\\\[\]-]+', ' ', docs ).strip()
	
	#print(f'preprocessed: (len: {len(docs)}) >>{docs}<<')
	if ( not docs or len(docs)==0 ):
		return

	all_ = stanza_multi_pipeline(docs)
	lm = [ re.sub('#|_','', word.lemma.lower()) for i, sent in enumerate(all_.sentences) for word in sent.words if ( word.lemma and len( re.sub(r'\b[A-Za-z](\.| |:)+', '', word.lemma ) ) > 2 and word.upos not in useless_upos_tags and word.lemma.lower() not in UNIQUE_STOPWORDS ) ]

	#print( lm )
	#print("<>"*70)
	return lm

def trankit_lemmatizer(docs):
	print(f'Raw: (len: {len(docs)}) >>{docs}<<', end='\t')
	if not docs:
		return

	# treat all as document
	docs = re.sub(r'\"|<[^>]+>|[~*^][\d]+', '', docs)
	docs = re.sub(r'[%,+;,=&\'*"°^~?!—.•()“”:/‘’<>»«♦■\\\[\]-]+', ' ', docs ).strip()
		
	print(f'preprocessed: (len: {len(docs)}) >>{docs}<<', end='\t')
	if ( not docs or len(docs)==0 ):
		return

	all_dict = p(docs)
	#lm = [ tk.get("lemma").lower() for sent in all_dict.get("sentences") for tk in sent.get("tokens") if ( tk.get("lemma") and len(re.sub(r'\b[A-Z](\.| |\:)+|\b[a-z](\.| |\:)+', '', tk.get("lemma") ) ) > 2 and tk.get("upos") not in useless_upos_tags and tk.get("lemma").lower() not in UNIQUE_STOPWORDS ) ] 
	lm = [ tk.get("lemma").lower() for sent in all_dict.get("sentences") for tk in sent.get("tokens") if ( tk.get("lemma") and len(re.sub(r'\b[A-Za-z](\.| |:)+', '', tk.get("lemma") ) ) > 2 and tk.get("upos") not in useless_upos_tags and tk.get("lemma").lower() not in UNIQUE_STOPWORDS ) ] 

	print( lm )
	return lm

def nltk_lemmatizer(sentence):	
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

	filtered_tokens = [w for w in tokens if not w in UNIQUE_STOPWORDS and len(w) > 1 and not w.isnumeric() ]
	# nltk.pos_tag: cheatsheet: pg2: https://computingeverywhere.soc.northwestern.edu/wp-content/uploads/2017/07/Text-Analysis-with-NLTK-Cheatsheet.pdf
	lematized_tokens = [wnl.lemmatize(w, t[0].lower()) if t[0].lower() in ['a', 's', 'r', 'n', 'v'] else wnl.lemmatize(w) for w, t in nltk.pos_tag(filtered_tokens)] 
	#print( list( set( lematized_tokens ) ) )

	return lematized_tokens
