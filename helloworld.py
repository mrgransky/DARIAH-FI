import enchant
import libvoikko

import stanza
from stanza.pipeline.multilingual import MultilingualPipeline
from stanza.pipeline.core import DownloadMethod
import nltk
import re
import time
import sys

nltk_modules = [
	'punkt',
	'stopwords',
	'wordnet',
	'averaged_perceptron_tagger', 
	'omw-1.4',
]

nltk.download(
	# 'all',
	nltk_modules,
	quiet=True, 
	# raise_on_error=True,
)

lang_id_config = {
	"langid_lang_subset": ['en', 'sv', 'da', 'ru', 'fi', 'et', 'de', 'fr']
}

lang_configs = {
	# "en": {"processors":"tokenize,lemma,pos", "package":'eslspok',"tokenize_no_ssplit":True},
	"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
	# "sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
	"sv": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
	"da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
	"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
	"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'tdt',"tokenize_no_ssplit":True}, # ftb wasn't accurate
	"et": {"processors":"tokenize,lemma,pos", "package":'edt',"tokenize_no_ssplit":True},
	"de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True},
	"fr": {"processors":"tokenize,lemma,pos", "package":'sequoia',"tokenize_no_ssplit":True},
}

smp = MultilingualPipeline(	
	lang_id_config=lang_id_config,
	lang_configs=lang_configs,
	download_method=DownloadMethod.REUSE_RESOURCES,
)

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

def stanza_lemmatizer(docs: str="This is a <NORMAL> sentence in document."):
	try:
		# print(f'Stanza[{stanza.__version__}] Raw Input:\n{docs}\n')
		# print(f"{f'nW: { len( docs.split() ) }':<10}{str(docs.split()[:7]):<150}", end="")
		st_t = time.time()
		all_ = smp(docs)
		for i, v in enumerate(all_.sentences):
			print(i, v)
			for ii, vv in enumerate(v.words):
				print(ii, vv.text, vv.lemma, vv.upos)
			print()

		lemmas_list = [ 
			re.sub(r'["#_\-]', '', wlm.lower())
			for _, vsnt in enumerate(all_.sentences) 
			for _, vw in enumerate(vsnt.words) 
			if ( 
					(wlm:=vw.lemma)
					and 5 <= len(wlm) <= 40
					and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\s+', wlm) 
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
	# docs = docs.title()
	# print(f'Raw Input:\n>>{docs}<<')
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
	# print(f'Cleaned Input:\n{docs}')
	# print(f"<>"*100)
	# # print(f"{f'Preprocessed: { len( docs.split() ) } words':<30}{str(docs.split()[:3]):<65}", end="")
	if not docs or len(docs) == 0 or docs == "":
		return
	return docs

def remove_misspelled_(documents: str="This is a sample sentence."):
	# print(f"Removing misspelled word(s)".center(100, " "))
	# Create dictionaries for Finnish, Swedish, and English
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
		# print(
		# 	word,
		# 	fi_dict.spell(word),
		# 	fii_dict.check(word), 
		# 	sv_dict.check(word), 
		# 	sv_fi_dict.check(word), 
		# 	en_dict.check(word),
		# 	de_dict.check(word),
		# 	# no_dict.check(word),
		# 	da_dict.check(word),
		# 	es_dict.check(word),
		# 	et_dict.check(word),
		# 	cs_dict.check(word), 
		# 	# cy_dict.check(word), 
		# 	# fo_dict.check(word), 
		# 	fr_dict.check(word), 
		# 	ga_dict.check(word), 
		# 	hr_dict.check(word), 
		# 	hu_dict.check(word), 
		# 	# is_dict.check(word), 
		# 	it_dict.check(word), 
		# 	lt_dict.check(word), 
		# 	lv_dict.check(word), 
		# 	nl_dict.check(word), 
		# 	pl_dict.check(word), 
		# 	sl_dict.check(word), 
		# 	sk_dict.check(word)
		# )
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
			it_dict.check(word) or 
			lt_dict.check(word) or 
			lv_dict.check(word) or 
			nl_dict.check(word) or 
			pl_dict.check(word) or 
			sl_dict.check(word) or 
			sk_dict.check(word)
		):
			# print(f"\t\t{word} does not exist")
			pass
		else:
			cleaned_words.append(word)
	# print(cleaned_words)
	# Join the cleaned words back into a string
	cleaned_doc = " ".join(cleaned_words)
	# print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(100, " "))
	return cleaned_doc

# orig_text = '''
# enGliSH
# Cellulose Union, The Finnish Woodpulp <em>and</em> Board Union, Owners of: <em>Myllykoski</em> Paper <em>and</em> Mechanical wood pulp mill. Establ<<
# Snowball (AA3399), Bargenoch Blue Blood (AA3529), <em>Dunlop Talisman</em> (A 3206), Lessnessock Landseer (A 3408), South Craig
# '''

# orig_text = '''
# sVenskA
# Redogörelse från ■ % , Kejserliga Alexanders-Universitetet -■',- . ■' ' ’ 
# för rcktoratstriennium ifrån början af hösttermin 1854 till samma tid 1857, *X . . 
# af Universitetets n. v. Rektor X t ■ ' ‘ Helsingfors, tryckt hos J. C. Frenckell & Son, 1857.
# Styrelseledamot – det här ingår i rollen!
# Hur får du en styrelse som faktiskt bidrar till bolagets framgång och skapar värde för ägarna? 
# I vår bloggserie, Rätt sätt i styrelsearbete, ger vi tips och råd på hur du kan göra skillnad. 
# I detta inlägg, det tredje i vår serie, 
# tydliggör Lena Hasselborn, styrelsecoach på PwC, vad som ingår i rollen som styrelseledamot. 
# Två spårvagnar har krockat i Kortedala i Göteborg. <em>Drumsö</em> Korkis Vippal kommer från Rågöarna!!!

# Flera personer har skadats, det är oklart hur allvarligt, skriver polisen på sin hemsida.

# Enligt larmet har en spårvagn kört in i en annan spårvagn bakifrån.

# Räddningstjänsten är på plats med en ”större styrka”.

# På grund av olyckan har all spårvagnstrafik mot Angered ställts in.

# – Vi tittar på att sätta in ersättningstrafik, säger Christian Blomquist, störningskoordinator på Västtrafik, till GP.
# En styrelseledamot i ett aktiebolag är en person som ingår i bolagets styrelse. 
# I majoriteten av alla svenska aktiebolag finns det enbart en ordinarie ledamot i styrelsen som är ensamt ansvarig för bolaget. 
# Om styrelsen har färre än tre ledamöter måste det även finnas en styrelsesuppleant.
# åbo valde vid årsmöte till ordf hrr Sipilä till viceordf hrr ekholm som styrelseledamöter.
# n:o 3 i Napo by, Storkyro, 166, 167, 168. 
# <em>Knuters</em>, n:o 17 i <em>Hindsby</em>, Sibbo, 160, 161, 162. Korhonens, I., 1&#x2F;2 n:o
# '''

orig_text = """
SUomI
ruottalan koski
Ruottala on kylä Tornion kaupungissa Kaakamajokivarressa Jäämerentien varrella.
Pieni osa kylästä kuuluu Keminmaan kuntaan, mutta suurin osa kylän asukkaista asuu Tornion puolella.
Tampereen Teknillinen Yliopisto
Osoite : Nimi: • I tillin II II 111 il ill II II Ullin lIIHIIIIMIIIIIIIIIIIHIIIIIIIIIIM
arpavihon ostajalle. Sunnuntcnnll näytetään kuuluisaa filminäytelmää „Kamelianai,nen".
Rumanlan kci 2tllwll'teatterissll.
Klliaanin autonkuljetatjat tekemät perheineen ja Mieraineen huomennä klo 12 päimällä automatkan Pllltaniemelle Sutelanperään.
Kllalmia paljastettu. 
Lehtien Bukarestista saamien tietojen mukaan on Rumaniassa päästy uuden kommunistisen järjestön jäljille, jonka tehtävänä on ollut sytyttää maan kaikki kirkot tuleen. 
(Ab Indiana Corporation) v &#x27;«.

VIIVVEI InII I II TA u <em>ADA</em> A Myös on meHIA musHkkl-tnatrv- KhIrRlA JuuLuIAVAnAA mentte]a
Kaikissa suuremmissa kaupungeissa on toimeenpantu pidätyksiä. 
Järjestön johtaja on kuulusteluissa kertonut toimineensa Weinin kommunistikeskuksen antamien ohjeiden mukaan.
Sähköyhtiöt ja asentajat!
Kesäkorj suksiin muuntaja-asemille ja ulkolinjoille sopivat tarvikkeet ostatte meiltä edullisin tukkuhinnoin.
Rauman Sähkö- ja Telefooniinko Urho Tuominen. Kauppak. 22. Puh. 11 43.
UNIVERsITY LIBRARY AT HELsINKI 30 helsink helsingfors helsingfars
J. VALLINKOsKI
<em>TURUN AKATEMIAN</em> VAlTOsKIRJAT
Kuninkaallinen Turun Akatemia 1642—1828
DIE DIssERTATIONEN DER
"vanhala nikkilä"~6 | Vanhala Nikkilä - Pietarila ja nykyään | <em>Michelspiltom</em>.
helsingin teknillinen reaalikoulu
Yrjönpäivää juhlitaan
mcchdilmsmi mcchdollffuulsi mcchdollhmj riksdag kräv mcchdollisimmclv mcchdollisimmclv 
mcchdollisimmclv mcchdollisnn mcchdollisnn mcchdvllffuus 
mcche mcchelinirrk mcchellnlnk mcchelm mcchk mcchl mcchingunkurmautsenll mcchioistctti 
mcchtlghrßc mcchnmm mcchowik mcchoofliftmma mcchta mccicl mcciipanf meciipanf mccjsu mecjsu 
tilallisen tytär Mirja H i dm a n ja tilallinen <em>Veikko Anttila</em>, molemmat Halikosta.
muistcttatpaa!
Salama Teatterissa
rhythms mxafl faslf faslm fasmiffl faspcnfi fastighetsntmnd. "alina keskinen" - iiiifff Vaili Siviä -
Pasi Klemettinen Taustialan Sipilä >>> Taustiala <<<<<<
N. ESPLANADG. 35 Platsagenter: Tammerfors: Vaind Kajanne Kuopio: Kuopion Kemikalikauppa Uleaborg: Oulun Kemikalikauppa
Suomen pääministeri | Helsingin pörssi ja suomen pankki 
res lausuntaa lotta aune puhe kenttäpappi virtanen kuorolaulua vaasan
Vilho Rokkola | <em>Juho Huppunen</em> | xxxx <em>Levijoki</em>
Albin Rimppi Toivainen, Juva, Suomi >> Juristi, varatuomari <<< Matts Michelsson, Sukula,
N:o 45
rOI M 1 1 US : Antinkatu 15. Fuh«hn 6 52. Av. ia perjantaina lisäksi 6—12 ip. <em>Drumsö<\em> Korkis Vippal kommer från Rågöarna!!!
Keskuspoliisi
Kommunistien jouKKowangitfemista tahoilla maassa.
-!£auqitjciMjd oasat suoranaisena jattona aikai seinnnn tapahtuneille pii osallisuus salaisen fonnnuni stipuolueen toim
ätytsille
Siffiffi ilmoitetaan
Pidätettnien lutumääm »ouiee
Malaga-kuvauskielellä kirjoittamaan sananmuodostussäännöstöön.
Etsimä Pohjalahtelaiset kokoontuvat koululle, valvojat: 
Kalle Oivio ja Matti Niemi. Mälkilän kylä kokoontuu Sipilään, valvojat: 
Lauri Laurila ja Heikki Mattila.
(joko kutoen tai ommel- ( len) saatte <em>ryijyn</em> uskomattoman &gt; 
huokealla. <em>Ryijyn</em> valmistaminen on S sitäpaitsi helppoa
n:o 3 i Napo by, Storkyro, 166, 167, 168. 
<em>Knuters</em>, n:o 17 i <em>Hindsby</em>, Sibbo, 160, 161, 162. Korhonens, I., 1&#x2F;2 n:o
Viihtyisä ja valoisa 6. kerroksen koti, hyvien kulkuyhteyksien ja palveluiden äärellä Kalliossa!
"""

cleaned_fin_text = clean_(docs=orig_text, del_misspelled=True)
cleaned_fin_text = stanza_lemmatizer(docs=cleaned_fin_text)