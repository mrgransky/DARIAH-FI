from utils import *

# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
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
	lang_configs = {#"en": {"processors":"tokenize,pos,lemma","tokenize_no_ssplit":True},
									"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
									"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
									"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'ftb',"tokenize_no_ssplit":True},
									}
	stanza_multi_pipeline = MultilingualPipeline(
			lang_id_config=lang_id_config,
			lang_configs=lang_configs,
			download_method=DownloadMethod.REUSE_RESOURCES,
	)
	useless_upos_tags = ["PUNCT", "CCONJ", "SYM", "AUX", "NUM", "DET", "ADP", "PRON", "PART", "ADV", "INTJ", "X"]
	STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
	my_custom_stopwords = ['btw', "could've", "n't","'s","—", "i'm", "'m", 
													"i've", "ive", "'d", "i'd", " i'll", "'ll", "'ll", "'re", "'ve", 
													'aldiz', 'baizik', 'bukatzeko', 
													'klo','nro', 'vol', 'amp', 'sid', 'obs',
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
	UNQ_STW = list(set(STOPWORDS))
	#print(f"Unique Stopwords: {len(UNQ_STW)} | {type(UNQ_STW)}\n{UNQ_STW}")
	all_words_list = list()
	all_lemmas_list = list()

def spacy_tokenizer(sentence):
	sentences = sentence.lower()
	sentences = re.sub(r'[~|^|*][\d]+', '', sentences)

	lematized_tokens = [word.lemma_ for word in sp(sentences) if word.lemma_.lower() not in sp.Defaults.stop_words and word.is_punct==False and not word.is_space]
	
	return lematized_tokens

@cache
def stanza_lemmatizer(docs):
	# words_list = list()
	lemmas_list = list()
	st_t = time.time()
	try:
		# print(f'stanza input:[{len(docs)}]:\n{docs}')
		print(f"{f'nWORDS: { len( docs.split() ) }':<15}{str(docs.split()[:5]):<150}", end="")
		all_ = stanza_multi_pipeline(docs)
		# list comprehension: slow but functional alternative
		# print(f"{f'{ len(all_.sentences) } sent.: { [ len(vsnt.words) for _, vsnt in enumerate(all_.sentences) ] } words':<40}", end="")
		lemmas_list = [ re.sub(r'#|_|\-','', wlm.lower()) for _, vsnt in enumerate(all_.sentences) for _, vw in enumerate(vsnt.words) if ( (wlm:=vw.lemma) and len(wlm)>2 and len(wlm)<=35 and not re.search(r"\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\s+", wlm) and vw.upos not in useless_upos_tags and wlm not in UNQ_STW ) ]
		# for _, vsnt in enumerate(all_.sentences):
		# 	for _, vw in enumerate(vsnt.words):
		# 		wlm = re.sub('#|_','', vw.lemma.lower())
		# 		wtxt = vw.text.lower()
		# 		if wtxt in all_words_list and wlm in all_lemmas_list:
		# 			# print(f"Already seen {wtxt} & lemma >>{wlm}<< available")
		# 			lemmas_list.append(wlm)
		# 		elif ( wtxt not in all_words_list and wlm and len(wlm) > 2 and not re.search(r"<eos>|<EOS>|<sos>|<SOS>|<UNK>|<unk>", wlm) and vw.upos not in useless_upos_tags and wlm not in UNQ_STW ):
		# 			# print(f"have not lemmatized: {wtxt}")
		# 			lemmas_list.append( wlm )
		# 			all_lemmas_list.append( wlm )
		# 		all_words_list.append(wtxt)
	
	except Exception as e:
		print(f"<!> Stanza Error: {e}")
		# logging.exception(e)
		return
	print(f"{f'{len(lemmas_list)} Lemma(s)':<20}Elapsed_t: {time.time()-st_t:.3f} s")
	# print( lemmas_list )
	# print("<>"*70)
	del docs, all_
	gc.collect()
	return lemmas_list

def trankit_lemmatizer(docs):
	# print(f'Raw: (len: {len(docs)}) >>{docs}<<')
	# print(f'Raw inp words: { len( docs.split() ) }', end=" ")
	st_t = time.time()
	if not docs:
		return

	# treat all as document
	docs = re.sub(r'\"|<[^>]+>|[~*^][\d]+', '', docs)
	docs = re.sub(r'[%,+;,=&\'*"°^~?!—.•()“”:/‘’<>»«♦■\\\[\]-]+', ' ', docs ).strip()
	
	# print(f'preprocessed: len: {len(docs)}:\n{docs}')
	print(f"{f'preprocessed doc contains { len( docs.split() ) } words':<50}{str(docs.split()[:3]):<60}", end=" ")
	if ( not docs or len(docs)==0 ):
		return

	all_dict = p(docs)
	#lm = [ tk.get("lemma").lower() for sent in all_dict.get("sentences") for tk in sent.get("tokens") if ( tk.get("lemma") and len(re.sub(r'\b[A-Z](\.| |\:)+|\b[a-z](\.| |\:)+', '', tk.get("lemma") ) ) > 2 and tk.get("upos") not in useless_upos_tags and tk.get("lemma").lower() not in UNQ_STW ) ] 
	lm = [ tk.get("lemma").lower() for sent in all_dict.get("sentences") for tk in sent.get("tokens") if ( tk.get("lemma") and len(re.sub(r'\b[A-Za-z](\.| |:)+', '', tk.get("lemma") ) ) > 2 and tk.get("upos") not in useless_upos_tags and tk.get("lemma").lower() not in UNQ_STW ) ]
	print(f"Elapsed_t: {time.time()-st_t:.3f} sec")
	# print( lm )
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

	filtered_tokens = [w for w in tokens if not w in UNQ_STW and len(w) > 1 and not w.isnumeric() ]
	# nltk.pos_tag: cheatsheet: pg2: https://computingeverywhere.soc.northwestern.edu/wp-content/uploads/2017/07/Text-Analysis-with-NLTK-Cheatsheet.pdf
	lematized_tokens = [wnl.lemmatize(w, t[0].lower()) if t[0].lower() in ['a', 's', 'r', 'n', 'v'] else wnl.lemmatize(w) for w, t in nltk.pos_tag(filtered_tokens)] 
	#print( list( set( lematized_tokens ) ) )

	return lematized_tokens
