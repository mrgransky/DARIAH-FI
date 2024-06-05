import enchant
import libvoikko
import json
import re
import time
import sys
import logging
import nltk
import trankit
from trankit import Pipeline

# nltk_modules = [
# 	'punkt',
# 	'stopwords',
# 	'wordnet',
# 	'averaged_perceptron_tagger', 
# 	'omw-1.4',
# ]

# nltk.download(
# 	# 'all', # consume disspace and slow
# 	# nltk_modules, # not required
# 	'stopwords',
# 	quiet=True, 
# 	# raise_on_error=True,
# )

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
#tmp = Pipeline('auto', embedding='xlm-roberta-large') # time-consuming and large models (unnecessary languages)
tmp = Pipeline(
	lang='finnish-ftb',
	gpu=True,
	embedding='xlm-roberta-large', 
	cache_dir='/home/farid/datasets/Nationalbiblioteket/trash',
)
tmp.add('english')
tmp.add('swedish')
tmp.add('danish')
tmp.add('russian')
tmp.add('french')
tmp.add('german')
tmp.set_auto(True)
print(f">> tmp elasped_t: {time.time()-tt:.3f} sec")

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

# print(enchant.list_languages())
# sys.exit(0)

def trankit_lemmatizer(docs: str="This is a <NORMAL> sentence in document."):
	try:
		print(f'Trankit[{trankit.__version__}] Raw Input:\n{docs}\n')
		# print(f"{f'nW: { len( docs.split() ) }':<10}{str(docs.split()[:7]):<150}", end="")
		docs = docs.title()
		st_t = time.time()
		all_ = tmp( docs )
		# print(type(all_))
		# print(json.dumps(all_, indent=2, ensure_ascii=False))
		# print("#"*100)

		# for i, v in enumerate(all_.get("sentences")):
		# 	# print(i, v, type(v)) # shows <class 'dict'> entire dict of id, text, lemma, upos, ...
		# 	for ii, vv in enumerate(v.get("tokens")):
		# 		print(ii, vv.get("text"), vv.get("lemma"), vv.get("upos"))
		# 		# print(f"<>")
		# 	print('-'*50)

		lemmas_list = [
			re.sub(r'[";=&#<>_\-\+\^\.\$\[\]]', '', wlm.lower())
			for _, vsnt in enumerate(all_.get("sentences"))
			for _, vw in enumerate(vsnt.get("tokens"))
			if (
				(wlm:=vw.get("lemma"))
				and 5 <= len(wlm) <= 43
				# and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<ros>|<eos>|/|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\^|\s+', wlm) # does not exclude words starting with digits! 
				and not re.search(r'\b(?=\d|\w)(?:\w*(\w)(\1{2,})\w*|\d+\w*)\b|<ros>|<eos>|/|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\^|\s+', wlm)
				
				and vw.get("upos") not in useless_upos_tags 
				and wlm not in UNQ_STW
			)
		]
		end_t = time.time()
	except Exception as e:
		print(f"<!> trankit Error: {e}")
		return
	print( lemmas_list )
	print(f"Found {len(lemmas_list)} lemma(s) in {end_t-st_t:.2f} s".center(140, "-") )
	return lemmas_list

def clean_(docs: str="This is a <NORMAL> string!!", del_misspelled: bool=False):
	# docs = docs.title()
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
			# no_dict.check(word),
			da_dict.check(word),
			es_dict.check(word), 
			et_dict.check(word), # estonian
			cs_dict.check(word), 
			# cy_dict.check(word), 
			# fo_dict.check(word), 
			fr_dict.check(word), 
			ga_dict.check(word), 
			hr_dict.check(word), 
			hu_dict.check(word), 
			# is_dict.check(word), 
			# it_dict.check(word), 
			lt_dict.check(word), 
			lv_dict.check(word), 
			nl_dict.check(word), 
			pl_dict.check(word), 
			sl_dict.check(word), 
			# sk_dict.check(word)
		)
		if not (
			fi_dict.spell(word) or 
			fii_dict.check(word) or 
			sv_dict.check(word) or 
			sv_fi_dict.check(word) or 
			en_dict.check(word) or
			de_dict.check(word) or
			# no_dict.check(word) or
			da_dict.check(word) or
			es_dict.check(word) or
			et_dict.check(word) or # estonian
			cs_dict.check(word) or 
			# cy_dict.check(word) or 
			# fo_dict.check(word) or 
			fr_dict.check(word) or 
			ga_dict.check(word) or 
			hr_dict.check(word) or 
			hu_dict.check(word) or 
			# is_dict.check(word) or 
			# it_dict.check(word) or 
			lt_dict.check(word) or 
			lv_dict.check(word) or 
			nl_dict.check(word) or 
			pl_dict.check(word) or 
			sl_dict.check(word) #or 
			# sk_dict.check(word)
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
# Poitzell Genealogy and Poitzell Family History Information 
# Finnish and Swedish Cellulose Unions helped genealogists
# The Finnish Woodpulp <em>and</em> Board Union, Owners of: <em>Myllykoski</em> Paper <em>and</em> Mechanical wood pulp mill. Establ<<
# Snowball (AA3399), Bargenoch Blue Blood (AA3529), <em>Dunlop Talisman</em> (A 3206), Lessnessock Landseer (A 3408), South Craig
# '''

orig_text = '''
sVenskA
krepp rynkat sidoskärp bruddräkt satin insats ärmar och volang starkt överhängande framstycken lång silkestovs bruddräkt chine lång skärp
brännare med vekar parti amp minut priserna billigaste rettig amp kommissionslager helsingfors cigarrer snus parti till fabrikspriser och konditioner hos 
Korpholmens spetälskehospital
Fängelseföreningen i Finland har af trycket utgifvit sin af centralutskottet afgifna sjette årsberättelse. Denna innehåller: 
1) Föredrag vid fängelseföreningens årsdag den 19 januari af fältprosten K. J. G. Sirelius; 
2) Årsberättelse; 
<em>Knuters</em>, n:o 17 i <em>Hindsby</em>, Sibbo, 160, 161, 162. Korhonens, I., 1&#x2F;2 n:o
Den 2 maj hissar Helsingfors Aktiebanks kontor i Nykarleby flaggan i topp. Kontoret firar 100 års jubileum. 
'''

# orig_text = '''
# SUomI | Vaskivuoren lukio
# kappalaisen virkatalossa syrjäisessä vieremän kylässä oli juhanilla nuorena tilaisuus tutustua samon ihanaan
# Matruusin koulutus
# kuukautissuoja
# antisemitismi ja Ateismi
# venäjää
# res lausuntaa lotta aune puhe kenttäpappi virtanen kuorolaulua vaasan
# Vilho Rokkola | <em>Juho Huppunen</em> | xxxx <em>Levijoki</em>
# Albin Rimppi Toivainen, Juva, Suomi >> Juristi, varatuomari <<< Matts Michelsson, Sukula,
# N:o 45
# rOI M 1 1 US : Antinkatu 15. Fuh«hn 6 52. Av. ia perjantaina lisäksi 6—12 ip. <em>Drumsö<\em> Korkis Vippal kommer från Rågöarna!!!
# Keskuspoliisi
# Kommunistien jouKKowangitfemista tahoilla maassa.
# -!£auqitjciMjd oasat suoranaisena jattona aikai seinnnn tapahtuneille pii osallisuus salaisen fonnnuni stipuolueen toim
# ätytsille
# '''

cleaned_fin_text = clean_(docs=orig_text, del_misspelled=True)
cleaned_fin_text = trankit_lemmatizer(docs=cleaned_fin_text)