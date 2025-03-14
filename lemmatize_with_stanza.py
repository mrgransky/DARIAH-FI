import enchant
import libvoikko

import stanza
from stanza.pipeline.multilingual import MultilingualPipeline
from stanza.pipeline.core import DownloadMethod
import nltk
import re
import time
import sys
import logging
from utils import *

# logging.getLogger("stanza").setLevel(logging.WARNING) # disable stanza log messages with severity levels of WARNING and higher (ERROR, CRITICAL)

# nltk_modules = [
# 	'punkt',
# 	'stopwords',
# 	'wordnet',
# 	'averaged_perceptron_tagger', 
# 	'omw-1.4',
# ]

nltk.download(
	# 'all', # consume disspace and slow
	# nltk_modules, # not required
	'stopwords',
	quiet=True, 
	# raise_on_error=True,
)
fii_dict = enchant.Dict("fi")
fi_dict = libvoikko.Voikko(language="fi")
sv_dict = enchant.Dict("sv_SE")
sv_fi_dict = enchant.Dict("sv_FI")
en_dict = enchant.Dict("en")
de_dict = enchant.Dict("de")
no_dict = enchant.Dict("no")
da_dict = enchant.Dict("da")
es_dict = enchant.Dict("es")
et_dict = enchant.Dict("et")
cs_dict = enchant.Dict("cs")
cy_dict = enchant.Dict("cy")
fo_dict = enchant.Dict("fo")
fr_dict = enchant.Dict("fr")
ga_dict = enchant.Dict("ga")
hr_dict = enchant.Dict("hr")
hu_dict = enchant.Dict("hu")
is_dict = enchant.Dict("is")
it_dict = enchant.Dict("it")
lt_dict = enchant.Dict("lt")
lv_dict = enchant.Dict("lv")
nl_dict = enchant.Dict("nl")
pl_dict = enchant.Dict("pl")
sl_dict = enchant.Dict("sl")
sk_dict = enchant.Dict("sk")

tt = time.time()
lang_id_config = {
	"langid_lang_subset": [
		'en', 
		'sv', 
		'da', 
		# 'nb',
		'ru', 
		'fi', 
		# 'et', 
		'de', 
		'fr',
	]
}

# # without storing lemmas:
# lang_configs = {
# 	"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
# 	# "fi": {"processors":"tokenize,lemma,pos,mwt", "package":'tdt',"tokenize_no_ssplit":True}, # TDT
# 	"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'ftb',"tokenize_no_ssplit":True}, # FTB
# 	"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
# 	# "sv": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
# 	"da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
# 	# "nb": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True,},
# 	"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
# 	"et": {"processors":"tokenize,lemma,pos", "package":'edt',"tokenize_no_ssplit":True},
# 	"de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True},
# 	# "fr": {"processors":"tokenize,lemma,pos", "package":'sequoia',"tokenize_no_ssplit":True},
# }

# with storing lemmas:
lang_configs = {
	"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True, "lemma_store_results":True},
	# "fi": {"processors":"tokenize,lemma,pos,mwt", "package":'tdt',"tokenize_no_ssplit":True, "lemma_store_results":True}, # TDT
	"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'ftb',"tokenize_no_ssplit":True, "lemma_store_results":True}, # FTB
	"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True, "lemma_store_results":True},
	# "sv": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True, "lemma_store_results":True},
	"da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True, "lemma_store_results":True},
	# "nb": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True,, "lemma_store_results":True},
	"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True, "lemma_store_results":True},
	# "et": {"processors":"tokenize,lemma,pos", "package":'edt',"tokenize_no_ssplit":True, "lemma_store_results":True},
	"de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True, "lemma_store_results":True},
	"fr": {"processors":"tokenize,lemma,pos", "package":'sequoia',"tokenize_no_ssplit":True, "lemma_store_results":True},
}

smp = MultilingualPipeline(	
	lang_id_config=lang_id_config,
	lang_configs=lang_configs,
	download_method=DownloadMethod.REUSE_RESOURCES,
	device="cuda:0",
)
print(f">> smp elasped_t: {time.time()-tt:.3f} sec")

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
	"INTJ", 
	"ADV", 
	# "X", # foriegn words will be excluded,
]

STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())

with open('meaningless_lemmas.txt', 'r') as file_:
	my_custom_stopwords=[line.strip() for line in file_]
STOPWORDS.extend(my_custom_stopwords)
# UNQ_STW = list(set(STOPWORDS))
UNQ_STW = set(STOPWORDS)

# print(enchant.list_languages())
# sys.exit(0)

def stanza_lemmatizer(docs: str="This is a <NORMAL> sentence in document."):
	try:
		print(f'Stanza[{stanza.__version__}] Raw Input:\n{docs}\n')		
		# print(f"{f'nW: { len( docs.split() ) }':<10}{str(docs.split()[:7]):<150}", end="")
		st_t = time.time()
		smp_t = time.time()
		# print(f">> smp elasped_t: {time.time()-smp_t:.3f} sec")

		all_ = smp(docs) # <class 'stanza.models.common.doc.Document'> convertable to Dict
		# print(type(all_))
		# print(all_)
		# print("#"*100)

		# for i, v in enumerate(all_.sentences):
		# 	print(i, v, type(v)) # shows <class 'stanza.models.common.doc.Sentence'> entire dict of id, text, lemma, upos, ...
		# 	for ii, vv in enumerate(v.words):
		# 		# print(ii, vv.text, vv.lemma, vv.upos)
		# 		print(f"<>")
		# 	print('-'*50)

		lemmas_list = [
			re.sub(r'[";&#<>_\-\+\^\.\$\[\]]', '', wlm.lower())
			for _, vsnt in enumerate(all_.sentences) 
			for _, vw in enumerate(vsnt.words) 
			if (
				(wlm:=vw.lemma)
				and 5 <= len(wlm) <= 43
				# and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|\$|\^|<unk>|\s+', wlm) 
				and not re.search(r'\b(?=\d|\w)(?:\w*(?<!\b)(\w)(\1{2,})\w*|\d+\w*|\w*\d\w*)\b|<ros>|<eos>|/|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\?|\^|\s+', wlm) # excludes words containing digits!
				and vw.upos not in useless_upos_tags 
				and wlm not in UNQ_STW
				# and get_nlf_pages(INPUT_QUERY=wlm) # time consuming approach like for loop!
			)
		]
	except Exception as e:
		print(f"<!> Stanza Error: {e}")
		return
	##########################################################################################################################
	# # TODO: remove cols with zero results of NLF (Timeout Session):
	# print(f"raw {len(lemmas_list)} lemma(s):\n{lemmas_list}")
	# print(f"Asynchronous checking of {len(lemmas_list)} lemma(s) for possible ZERO NLF result pages [might take a while]", end="\t")
	# async_st_time = time.time()
	# lemmas_num_NLF_pages_async = asyncio.run(get_num_NLF_pages_asynchronous_run(TOKENs_list=lemmas_list))
	# lemmas_list_tmp = lemmas_list
	# lemmas_num_NLF_pages_async_tmp = lemmas_num_NLF_pages_async
	# lemmas_list = [word for num, word in zip(lemmas_num_NLF_pages_async_tmp, lemmas_list_tmp) if (num and num != 0) ]
	# print(f"Elapsed_t: {time.time()-async_st_time:.2f} sec")
	##########################################################################################################################
	end_t = time.time()
	print( lemmas_list )
	print(f"Found {len(lemmas_list)} lemma(s) in TOTAL lemmatization time: {end_t-st_t:.2f} s".center(140, "-") )
	return lemmas_list

def clean_(docs: str="This is a <NORMAL> string!!", del_misspelled: bool=False):
	# print(f'Raw Input:\n>>{docs}<<')
	if not docs or len(docs) == 0 or docs == "":
		return
	docs = re.sub(r'[\{\}@®¤†±©§½✓%,+–;,=&\'\-$€£¥#*"°^~?!❁—.•()˶“”„:/।|‘’<>»«□™♦_■►▼▲❖★☆¶…\\\[\]]+', ' ', docs )#.strip()
	docs = re.sub(r'\b(?:\w*(\w)(\1{2,})\w*)\b|\d+', " ", docs)#.strip()
	docs = re.sub(
		r'\s{2,}', 
		" ", 
		re.sub(r'\b\w{,2}\b', ' ', docs)
	).strip() # rm words with len() < 3 ex) ö v or l m and extra spaces
	##########################################################################################
	if del_misspelled:
		docs = remove_misspelled_(documents=docs)
	docs = docs.lower()
	##########################################################################################
	# print(f'Cleaned Input:\n{docs}')
	# print(f"<>"*100)
	# # print(f"{f'Preprocessed: { len( docs.split() ) } words':<30}{str(docs.split()[:3]):<65}", end="")
	if not docs or len(docs) == 0 or docs == "":
		return
	return docs

def remove_misspelled_(documents: str="This is a sample sentence."):
	# print(f"Removing misspelled word(s)".center(100, " "))	
	# Split the documents into words
	documents = documents.title()
	if not isinstance(documents, list):
		# print(f"Convert to a list of words using split() command |", end=" ")
		words = documents.split()
	else:
		words = documents
	
	# print(f"Document conatins {len(words)} word(s)")
	t0 = time.time()
	cleaned_words = []
	for word in words:
		# print(word)
		print(
			word,
			fi_dict.spell(word),
			fii_dict.check(word),
			sv_dict.check(word), 
			sv_fi_dict.check(word),
			en_dict.check(word),
			de_dict.check(word),
			da_dict.check(word),
			es_dict.check(word), 
			et_dict.check(word),
			cs_dict.check(word), 
			fr_dict.check(word), 
			# ga_dict.check(word), 
			# hr_dict.check(word), # Croatian
			# hu_dict.check(word), 
		)
		if not (
			fi_dict.spell(word)
			or fii_dict.check(word)
			or sv_dict.check(word)
			or sv_fi_dict.check(word)
			or en_dict.check(word)
			or de_dict.check(word)
			or da_dict.check(word)
			or es_dict.check(word)
			or et_dict.check(word)
			or cs_dict.check(word)
			or fr_dict.check(word)
			# or ga_dict.check(word) # TODO: to be removed!
			# or hr_dict.check(word) # TODO: to be removed!
			# or hu_dict.check(word) # TODO: to be removed!
		):
			print(f"\t\t{word} does not exist")
			pass
		else:
			cleaned_words.append(word)
	# print(cleaned_words)
	# Join the cleaned words back into a string
	cleaned_doc = " ".join(cleaned_words)
	# print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(100, " "))
	return cleaned_doc

# orig_text = '''
# enGliSH with Swedish genealogists
# the reliability helsingin sanomat advertising paper helsingin sanomat only finnish paper with own
# Poitzell Genealogy and Poitzell Family History Information 
# Finnish and Swedish Cellulose Unions helped genealogists
# The Finnish Woodpulp <em>and</em> Board Union, Owners of: <em>Myllykoski</em> Paper <em>and</em> Mechanical wood pulp mill. Establ<<
# Snowball (AA3399), Bargenoch Blue Blood (AA3529), <em>Dunlop Talisman</em> (A 3206), Lessnessock Landseer (A 3408), South Craig
# '''

# orig_text = '''
# sVenskA idrottsförbundet
# vårtermin
# myntgat
# < arbetets krävande natur ><
# Åbo, å Alfred Holmströms advokatkont, slutredovisning (260, 261, 262.) 28 Fredag, kl. 6 e. m. i handl. 
# '''

orig_text = '''
SUomI
Suomen Sosialidemokraatti
kolonia
Naisasia isänmaan elämän kysymyksenä.
naisasia
kolonier
kolonin
naisasian
Hän käänsi suomeksi erään merkkiteoksen uskonpuhdistuksen historian alalta ja julkaisi uudestaan suomentamansa luterilaisen kirkkomme tunnustuskirjat sekä useita Martti Lutherin kirjoja. Latinankieli, kuten yleensä klassilliset kielet, oli vainajalle erityisen harrastuksen esineenä. Monen monet latinankieliset kirjeet, sähkösanomat ja tervehdykset tovereille ovat siitä todistuksena. Mainittakoon tämän yhteydessä vielä Hedbergin valmistama ja julkaisema ansiokas kreikkalais-suomalainen Uuden Testamentin sanakirja, joka etupäässä lienee tuottanutkin hänelle teologian kunniatohtorin arvon vuoden 1907 tohtorinvihkijäisissä.
Saarnamiehenä Hedberg oli etevä ja suosittu, edustaen sellaisenakin horjumatonta evankelisluterilaisen kirkkomme tunnustuksellista kantaa. Loppuijälläänkin vielä Hecjberg Turun tuomiorovastina saarnasi ahkerasti, eikä vain omassa seurakunnassaan, vaan oli hän, kuten leikillisesti itse sanoi, koko tuomiorovastikuntansa papiston yhteinen saarnaapulainen, tarjoutuen saarnoillaan auttamaan milloin missäkin apua kaivattiin.
Maamme perustuslakitaistelussa Hedberg seisoi voimainsa ja mahdollisuuksien mukaan kaiken aikaa horjumatta niiden kanssa ja puolella, jotka pelkäämättä nousivat maatamme ja kansaamme vastaan tähdättyjä sortotoimenpiteitä vastustamaan. Parhaansa mukaan hän pyysi samanlaista henkeä istuttaa tovereihinsakin, missä ikinä hän liikkui. Keskustelut valtiopäivillä niiltä ajoilta ja vainajan esiintyminen sortovuosina Savonlinnan tuomiokapitulissa ovat todistuksena hänen horjumattomasta uskollisuudestaan maamme lainalaista järjestystä kohtaan.
Hedbergin konservatiivisesta katsantokannasta johtui, että hän varsin vastenmielisesti suostui yksikamarisen eduskunnan luomiseen.
'''

cleaned_fin_text = clean_(docs=orig_text, del_misspelled=True)
cleaned_fin_text = stanza_lemmatizer(docs=cleaned_fin_text)