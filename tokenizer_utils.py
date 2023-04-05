import re
import spacy
import nltk
import trankit
from trankit import Pipeline

#p = Pipeline('auto', embedding='xlm-roberta-large')
p = Pipeline('english', embedding='xlm-roberta-large')
p.add('finnish-ftb')
p.add('swedish')
p.add('russian')
p.add('estonian')
p.set_auto(True)

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
UNIQUE_STOPWORDS = list(set(STOPWORDS))
#print(f"Unique Stopwords: {len(UNIQUE_STOPWORDS)} | {type(UNIQUE_STOPWORDS)}\n{UNIQUE_STOPWORDS}")

def spacy_tokenizer(sentence):
	sentences = sentence.lower()
	sentences = re.sub(r'[~|^|*][\d]+', '', sentences)

	lematized_tokens = [word.lemma_ for word in sp(sentences) if word.lemma_.lower() not in sp.Defaults.stop_words and word.is_punct==False and not word.is_space]
	
	return lematized_tokens

def trankit_lemmatizer(docs):
	print(f"raw inp docs: {docs}", end='\t=>\t')
	if not docs:
				return
	useless_upos_tags = ["PUNCT", "CCONJ", "SYM", "AUX", "NUM", "DET", "ADP", "PRON", "PART", "ADV"]
	# treat all as sentences
	#all_dict = p(docs, is_sent=True)
	#lm = [ re.sub('#', '', tk.get("lemma").lower())  for tk in all_dict.get("tokens") if ( tk.get("lemma") and len(tk.get("lemma")) > 2 and tk.get("upos") not in useless_upos_tags ) ] 
	#lm = [ tk.get("lemma").lower()  for tk in all_dict.get("tokens") if ( tk.get("lemma") and tk.get("upos") not in useless_upos_tags ) ]

	# treat all as document
	docs = re.sub(r'[+]', ' ', docs ) # +kotola+tohmajärvi+nyman
	print(f"preprocessed: {docs}", end=' ')

	all_dict = p(docs)
	lm = [ re.sub('#', '', tk.get("lemma").lower()) for sent in all_dict.get("sentences") for tk in sent.get("tokens") if ( tk.get("lemma") and len(re.sub(r'[A-Za-z][.][\s]+|[A-Za-z][.]+', '', tk.get("lemma") ) ) > 2 and tk.get("upos") not in useless_upos_tags ) ] 
	#lm = [ re.sub('#', '', tk.get("lemma").lower()) for sent in all_dict.get("sentences") for tk in sent.get("tokens") if ( tk.get("lemma") and len( tk.get("lemma") ) > 2 and tk.get("upos") not in useless_upos_tags ) ] 
	#lm = [ tk.get("lemma").lower()  for tk in all_dict.get("tokens") if ( tk.get("lemma") and tk.get("upos") not in useless_upos_tags ) ]
	print(lm)
	return list( set( lm ) )

def nltk_lemmatizer(sentence, stopwords=UNIQUE_STOPWORDS, min_words=4, max_words=200, ):	
	#print(sentence)
	if not sentence:
		return
	wnl = nltk.stem.WordNetLemmatizer()

	sentences = sentence.lower()
	sentences = re.sub(r'"|<.*?>|[~|*|^][\d]+', '', sentences)
	sentences = re.sub(r"\W+|_"," ", sentences) # replace special characters with space
	#sentences = re.sub("\s+", " ", sentences)
	sentences = re.sub("\s+", " ", sentences).strip() # strip() removes leading (spaces at the beginning) & trailing (spaces at the end) characters

	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')	
	tokens = tokenizer.tokenize(sentences)

	#filtered_tokens = [w for w in tokens if not w in stopwords]
	filtered_tokens = [w for w in tokens if not w in stopwords and len(w) > 1 and not w.isnumeric() ]

	lematized_tokens = [wnl.lemmatize(w, t[0].lower()) if t[0].lower() in ['a', 's', 'r', 'n', 'v'] else wnl.lemmatize(w) for w, t in nltk.pos_tag(filtered_tokens)]
	return lematized_tokens