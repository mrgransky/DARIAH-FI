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

# logging.getLogger("stanza").setLevel(logging.WARNING) # disable stanza log messages with severity levels of WARNING and higher (ERROR, CRITICAL)

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
			# re.sub(r'["#_\-]', '', wlm.lower())
			re.sub(r'[";&#<>_\-\+\^\.\$\[\]]', '', wlm.lower())
			for _, vsnt in enumerate(all_.sentences) 
			for _, vw in enumerate(vsnt.words) 
			if (
				(wlm:=vw.lemma)
				and 5 <= len(wlm) <= 43
				and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|\$|\^|<unk>|\s+', wlm) 
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

# orig_text = '''
# sVenskA
# brännare med vekar parti amp minut priserna billigaste rettig amp kommissionslager helsingfors cigarrer snus parti till fabrikspriser och konditioner hos 
# Korpholmens spetälskehospital
# ordf. doktor Arthur Gylling samt drkns ordf. kapten <em>Jacob</em> Lundqvist och v. ordf. gårdsägaren Albert Eriksson
# mellan nordiska länderna enligt vad statsminister erlander meddelat vid samtal med expressen börjar man med
# Kontorsmannaförbundets ombudsman hovrättsauskultanten <em>Sven Sevelius</em>, vilken tillika övertar befattningen som Förlagsföreningens
# Kontorsmannaförbundet 30 år
# Bestyrelsen för allmänna finska utställningen har af landets regering emottagit en tacksägelseskrifvelse i anledning af den lyckliga anordningen af finska allmänna utställningen, 
# hvarjemte till bestyrelsens ordförande, friherre J. A. von Born, öfverlemnats ett exemplar i guld af utställningens prismedalj. — 
# Fängelseföreningen i Finland har af trycket utgifvit sin af centralutskottet afgifna sjette årsberättelse. Denna innehåller: 
# 1) Föredrag vid fängelseföreningens årsdag den 19 januari af fältprosten K. J. G. Sirelius; 
# 2) Årsberättelse; 
# 3) Iustruktion för fängelseföreningens i Finland agenter i Helsingfors; 
# 4) Reglemente för lcke-Officiela Afdelningen.
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
# KffiffiffiffiMffiKäiSKraälSSSäiSiäfiSJSSSSiälJSiffi
# mdccxc. 45 Ö® ettlfllft att, 4-&#x27;to. 
# EnChriftcm <em>Flyttning</em> ur Tiden i Evigheten och dirpl följande SaTiga TilftJnd
# 1 <em>Österbottningen</em> | i GAMLAKARLEBY ffi har stor spridning, w företrädesvis
# <em>Knuters</em>, n:o 17 i <em>Hindsby</em>, Sibbo, 160, 161, 162. Korhonens, I., 1&#x2F;2 n:o
# Den 2 maj hissar Helsingfors Aktiebanks kontor i Nykarleby flaggan i topp. Kontoret firar 100 års jubileum. 
# '''

orig_text = '''
SUomI | Vaskivuoren lukio
Matruusin koulutus
kuukautissuoja
tietopuolisille kursseille. 
Pääsyvaatimuksina on: kansakoulun kurssi, 
terve ruumiin rakenne, ikä mieluummin 18 vuotta...laskento, kauppalaskento, 
tavaran hinnoittaminen, suom. ja routs. 
kauppakirjeenvaihto, kauppaoppi, kauppamaantiede, kielet...Kotkassa 1910. 
Etelä-Suomen kirjapaino Raussilan kansakoulun wesikatto annetaan urakkahuutokaupalla t. k. 12 p...
antisemitismi ja Ateismi
venäjää
Oikeistososialistinen Socialdemokraten huomauttaa,
Oikeistofosialistinen Socialdem«' kraten huomauttaa, 
Muistettavaa Tammikuu. 16. Jäsentenväl. hiihtokilp.. 
Urheiluliitto. 16. Hiihto- ja Kävelyretki, Urheiluliitto. 
15 —16. Kans. painikilp., Toverit, Viipuri. 23. 
Piirin hiihtomest. kilp. Kaiku. 23. Kans. voimistelukilp.,
Huomattavin osa maamme kristil-i listä tyttötyötä on kyllä tässä kirjassa saanut tarpeellisen ja odotetun oppaan. 
kauklahden kirjasto - torvisoittokunta - torwisoittokunta
Iisalmen uimahalli | helsingin tuomiokirkko | helsingin seurakuntayhtymä
jyväskylän sähkövalaistuskeskusasema | tuomiokirkkoseurakunta | helsingin hiippakunta
helsingin sähköwalaistuskeskusasemalla
helsingin sähkövalaistuskeskusasemalla
Mustalaiset - aineistoa romaneista
tiedusteluorganisaatio puolustusvoimat
kalle salonen emil karl karlsson heikki petter emil rask anton
siitä vasemmalle pohjoiseen xvii cois sai yöllä
olemassa kullan temppeli kokoon
mattila palkinto launonen kalle uotila uitto sameli salonen ohra palk salomon ranta
ABO£,£xcudJOH* KIA.MPE, Reg. Acad. Typogr.
HELSINGFORS TEL. 524 &quot; ________________=___B Broderar! <em>OMPELUKONE</em> O&#x2F;V HUSQVARNA SYMASKIN Tammerfors, Helsingfors, Åbo
kartanossa hyvinkäällä astutusmaksu puh hyvinkään sanomain asiamiehille hyvinkään sanomat
URHO SEPPÄLÄ.
mielenkiintoisesti viljan siemenestä oppilas juho pesonen lausui runon ven paavo kolmannen
Pohjois-Hämeen pohjois-hämeen osuuspankki pohjois pohjanmaa
länsi uudenmaan hyvinvointialue
Sanomalehti Länsi-Suomi itä-uudenmaan hyvinvointialue
etelä-karjalan hyvinvointialue
myös ylösnousemus josa nimittäin samalla yli toisen
Ylein. Suomen Maanwiljelyskokous"
Vuonna 1921 sveitsiläinen psykiatri Hermann Rorschach jul* kaisi 
metodiikan ja ensimmäiset tulokset nerokkaasti suunnitellusta psykodiagnostisesta koemenetelmästä, jota sittemmin on totuttu nimittämään hänen mukaansa. 1 )
Tämän jälkeenkin hän jatkoi uuras* tustaan menetelmänsä parantamiseksi, 
mutta ennenaikainen kuolema katkaisi jo seuraavan vuoden alkupuoliskolla yhtäkkiä lupaavan työskentelyn vielä ratkaisemattomien probleemien selvittämiseksi.
229; Jyväskylän seminaari 6 >>ESBO och Björneborg Jyväskylän yliopiston kauppakorkeakoulu
Helsink Jywäskylän yliopisto on Jyväskylässä sijaitseva suomalainen yliopisto. 
Jyväskylän yliopisto on monitieteinen tiedeyliopisto, jossa opiskelijoita on noin 15 000 ja henkilökuntaa noin 2 600. <em>Mattei</em> 
82 Hautajärvi, Juho Juhonp. (Hauta-Junnu)
ruottalan koski
PUOLELLA — KUTEN TAVALLISTA.
Brainerd, tammik. 12 p. — Tn <em>mari</em> McCli iiahan on kumonnut verottajan jiäätöksen, jolla
Ilmankos sitä "yliopistoksi" sanottaisiin.
Ruottala on kylä Tornion kaupungissa Kaakamajokivarressa Jäämerentien varrella.
Pieni osa kylästä kuuluu Keminmaan kuntaan, mutta suurin osa kylän asukkaista asuu Tornion puolella.
Siffiffi ilmoitetaan
Pidätettnien lutumääm »ouiee
HiltulaZta n:o 2.
Muuttokirjaa anoneet: GMnen trpp. <em>Sakari Eronen</em> R&gt;epamäestä n:o t prjytH l perficineen muuttokirjaa
huokealla. <em>Ryijyn</em> valmistaminen on S sitäpaitsi helppoa
n:o 3 i Napo by, Storkyro, 166, 167, 168. 
<em>Knuters</em>, n:o 17 i <em>Hindsby</em>, Sibbo, 160, 161, 162. Korhonens, I., 1&#x2F;2 n:o
Mutta huomattavina osakkaina ovat myöskin belgialaiset, sveitsiläiset, hollantilaiset ja tshekkoslovakialaiset kapitalistit.
SUomI x2
Osoite : Nimi: • I tillin II II 111 il ill II II Ullin lIIHIIIIMIIIIIIIIIIIHIIIIIIIIIIM
arpavihon ostajalle. Sunnuntcnnll näytetään kuuluisaa filminäytelmää „Kamelianai,nen".
Rumanlan kci 2tllwll'teatterissll.
Klliaanin autonkuljetatjat tekemät perheineen ja Mieraineen huomennä klo 12 päimällä automatkan Pllltaniemelle Sutelanperään.
Kllalmia paljastettu. 
Lehtien Bukarestista saamien tietojen mukaan on Rumaniassa päästy uuden kommunistisen järjestön jäljille, jonka tehtävänä on ollut sytyttää maan kaikki kirkot tuleen. 
(Ab Indiana Corporation) v &#x27;«.
Mäkelä-Henriksson och Marja Lounassalo]. 
Hki: Helsingin <em>yliopiston</em> <em>kirjasto</em>, 1973. 15 s. Stencil.
Barck, P. 0., Ett nytt
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
'''

cleaned_fin_text = clean_(docs=orig_text, del_misspelled=True)
cleaned_fin_text = stanza_lemmatizer(docs=cleaned_fin_text)