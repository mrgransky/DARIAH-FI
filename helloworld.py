import enchant
import libvoikko

import stanza
from stanza.pipeline.multilingual import MultilingualPipeline
from stanza.pipeline.core import DownloadMethod
import nltk
import re
import time
import sys
from functools import cache

nltk_modules = [
	'punkt',
	'stopwords',
	'wordnet',
	'averaged_perceptron_tagger', 
	'omw-1.4',
]
nltk.download(
	#'all',
	nltk_modules,
	quiet=True, 
	raise_on_error=True,
)

lang_id_config = {
	"langid_lang_subset": ['en', 'sv', 'da', 'ru', 'fi', 'de', 'fr']
}
lang_configs = {
	"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
	"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
	"da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
	"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
	"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'ftb',"tokenize_no_ssplit":True},
	"de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True},
	"fr": {"processors":"tokenize,lemma,pos", "package":'sequoia',"tokenize_no_ssplit":True},
}
smp = MultilingualPipeline(	
	lang_id_config=lang_id_config,
	lang_configs=lang_configs,
	download_method=DownloadMethod.REUSE_RESOURCES,
)
useless_upos_tags = ["PUNCT", "CCONJ", "SYM", "AUX", "NUM", "DET", "ADP", "PRON", "PART", "ADV", "INTJ", "X"]
STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
with open('meaningless_lemmas.txt', 'r') as file_:
	my_custom_stopwords=[line.strip() for line in file_]
STOPWORDS.extend(my_custom_stopwords)
UNQ_STW = list(set(STOPWORDS))


print(enchant.list_languages())
# sys.exit(0)

@cache
def stanza_lemmatizer(docs: str="This is a <NORMAL> sentence in document."):
	try:
		print(f'Stanza[{stanza.__version__}] Raw Input:\n{docs}\n')
		# print(f"{f'nW: { len( docs.split() ) }':<10}{str(docs.split()[:7]):<150}", end="")
		st_t = time.time()
		all_ = smp(docs)
		# for i, v in enumerate(all_.sentences):
		# 	print(v)
		# 	for ii, vv in enumerate(v.words):
		# 		print(vv.text, vv.lemma)
		# 	print()

		lemmas_list = [ 
			# re.sub(r'["#_\-]', '', wlm.lower())
			re.sub(r'["#_\-]', '', wlm)
			for _, vsnt in enumerate(all_.sentences) 
			for _, vw in enumerate(vsnt.words) 
			if ( 
					(wlm:=vw.lemma)
					and 5 <= len(wlm) <= 40
					and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|"|#|<unk>|\s+', wlm) 
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

def clean_(docs: str="This is a <NORMAL> string!!", del_misspelled: bool=False):
	print(f'Raw Input:\n>>{docs}<<')
	if not docs or len(docs) == 0 or docs == "":
		return
	docs = re.sub(r'[\{\}@®¤†±©§½✓%,+;,=&\'\-$€£¥#*"°^~?!❁—.•()˶“”„:/।|‘’<>»«□™♦_■►▼▲❖★☆¶…\\\[\]]+', ' ', docs )#.strip()
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
	print(f'Cleaned Input:\n{docs}')
	print(f"<>"*100)
	# # print(f"{f'Preprocessed: { len( docs.split() ) } words':<30}{str(docs.split()[:3]):<65}", end="")
	if not docs or len(docs) == 0 or docs == "":
		return
	return docs

def remove_misspelled_(documents: str="This is a sample sentence."):
	print(f"Removing misspelled word(s)".center(100, " "))
	# Create dictionaries for Finnish, Swedish, and English
	fi_dict = libvoikko.Voikko(language="fi")	
	fii_dict = enchant.Dict("fi")
	sv_dict = enchant.Dict("sv_SE")
	sv_fi_dict = enchant.Dict("sv_FI")
	en_dict = enchant.Dict("en")
	de_dict = enchant.Dict("de")
	no_dict = enchant.Dict("no")
	da_dict = enchant.Dict("da")
	es_dict = enchant.Dict("es")
	et_dict = enchant.Dict("et")
	
	ca_dict = enchant.Dict("ca")
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
	
	# Split the documents into words
	if not isinstance(documents, list):
		print(f"Convert to a list of words using split() command |", end=" ")
		words = documents.split()
	else:
		words = documents
	
	print(f"Document conatins {len(words)} word(s)")
	t0 = time.time()
	cleaned_words = []
	for word in words:
		# print(
		# 	word,
		# 	fi_dict.spell(word),
		# 	fii_dict.check(word), 
		# 	sv_dict.check(word), 
		# 	sv_fi_dict.check(word), 
		# 	en_dict.check(word),
		# 	de_dict.check(word),
		# 	no_dict.check(word),
		# 	da_dict.check(word),
		# 	es_dict.check(word),
		# 	et_dict.check(word)
		# )
		if not (
			fi_dict.spell(word) or 
			fii_dict.check(word) or 
			sv_dict.check(word) or 
			sv_fi_dict.check(word) or 
			en_dict.check(word) or
			de_dict.check(word) or
			no_dict.check(word) or
			da_dict.check(word) or
			es_dict.check(word) or
			et_dict.check(word) or # estonian
			ca_dict.check(word) or
			cs_dict.check(word) or 
			cy_dict.check(word) or 
			fo_dict.check(word) or 
			fr_dict.check(word) or 
			ga_dict.check(word) or 
			hr_dict.check(word) or 
			hu_dict.check(word) or 
			is_dict.check(word) or 
			it_dict.check(word) or 
			lt_dict.check(word) or 
			lv_dict.check(word) or 
			nl_dict.check(word) or 
			pl_dict.check(word) or 
			sl_dict.check(word) or 
			sk_dict.check(word)
		):
			print(f"\t\t{word} does not exist")
			pass
		else:
			cleaned_words.append(word)
	# print(cleaned_words)
	# Join the cleaned words back into a string
	cleaned_doc = " ".join(cleaned_words)
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(100, " "))
	return cleaned_doc

# orig_text = '''
# Två spårvagnar har krockat i Kortedala i Göteborg. <em>Drumsö</em> Korkis Vippal kommer från Rågöarna!!!

# Flera personer har skadats, det är oklart hur allvarligt, skriver polisen på sin hemsida.

# Enligt larmet har en spårvagn kört in i en annan spårvagn bakifrån.

# Räddningstjänsten är på plats med en ”större styrka”.

# På grund av olyckan har all spårvagnstrafik mot Angered ställts in.

# – Vi tittar på att sätta in ersättningstrafik, säger Christian Blomquist, störningskoordinator på Västtrafik, till GP.
# '''

orig_text = """
Vilho Rokkola | <em>Juho Huppunen</em> | xxxx <em>Levijoki</em>
"vanhala nikkilä"~6 | Vanhala Nikkilä - Pietarila ja nykyään
Yrjönpäivää juhlitaan
Albin Rimppi  Toivainen, Juva, Suomi >> Juristi, varatuomari <<< Matts Michelsson, Sukula,
"alina keskinen" - iiiifff Vaili Siviä - Pasi Klemettinen Taustialan Sipilä >>> Taustiala <<<<<<
N. ESPLANADG. 35 Platsagenter: Tammerfors: Vaind Kajanne Kuopio: Kuopion Kemikalikauppa Uleaborg: Oulun Kemikalikauppa
Katso grafiikoista, miten Suomen ja Ruotsin sotilaallinen voima eroaa
Suomi voittaa Ruotsin henkilöstön ja maavoimien kaluston määrässä. Ruotsilla sotilasteknologia on joiltain osin korkeampaa laatua. 
Suomen pääministeri | Helsingin pörssi ja suomen pankki | 
mcchdilmsmi mcchdollffuulsi mcchdollhmj riksdag kräv mcchdollisimmclv mcchdollisimmclv mcchdollisimmclv mcchdollisnn mcchdollisnn mcchdvllffuus mcche mcchelinirrk mcchellnlnk mcchelm mcchk mcchl mcchingunkurmautsenll mcchioistctti mcchtlghrßc mcchnmm mcchowik mcchoofliftmma mcchta mccicl mcciipanf meciipanf mccjsu mecjsu rhythms mxafl faslf faslm fasmiffl faspcnfi fastighetsntmnd. 
N:o 45

rOI M 1 1 US : Antinkatu 15. Fuh«hn 6 52. Av. ia perjantaina lisäksi 6—12 ip. <em>Drumsö<\em> Korkis Vippal kommer från Rågöarna!!!
Vastaava: Ei n KONTTORI: Antinkatu 15 (Kthityksen kirjakau] taina 8-4. Tilauksia, ilmoituksia, kirjapainotöit;

Etsimä
Pälkäneellä ensi sunnuntaina maaliskuun 1 pnä klo 13. Kokoontumispaikat: Onkkaalassa sk-talolle. Valvojat: Matti Heikkilä ja Huugo Aalto. Äimälässä kokoonnutaan J. Lassilaan, valvojat: Kalle Lemola ja Jussi Lassila. Iltasmäellä koululle, valvojat: Eino Tamminen ja Eelis Värilä. Laitikkalassa kokoonnutaan Meijerille, valvojat: Jussi Kaakinen ja Jussi Helmikkala. Kukkolan kylä kokoontuu koululle, valvojat: Heikki Mäkelä ja Aukusti Aspila. Sappeessa kokoonnutaan koululle, valvojat: Tauno Nieminen ja Väinö Hartikkala. Salmentaka kokoontuu kansakoululle, valvojat: August Koivisto ja Kalle Kauppi. Pohjalahtelaiset kokoontuvat koululle, valvojat: Kalle Oivio ja Matti Niemi. Mälkilän kylä kokoontuu Sipilään, valvojat: Lauri Laurila ja Heikki Mattila.
Keskuspoliisi

Kommunistien jouKKowangitfemista tahoilla maassa.
-!£auqitjciMjd oasat suoranaisena jattona aikai seinnnn tapahtuneille pii osallisuus salaisen fonnnuni stipuolueen toim
ätytsille
Siffiffi ilmoitetaan
Pidätettnien lutumääm »ouiee
Suomalaisyrittäjä pidätettiin Mijasissa 
Espanjan Aurinkorannikolla tunnettu pitkän linjan yrittäjä päätyi yllättäen kaltereiden taakse. 
"""

# print(orig_text)
cleaned_fin_text = clean_(docs=orig_text, del_misspelled=True)
cleaned_fin_text = stanza_lemmatizer(docs=cleaned_fin_text)
print(f"Final Cleaned:")
print(cleaned_fin_text)