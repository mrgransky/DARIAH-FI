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
def stanza_lemmatizer(docs):
	try:
		print(f'Stanza[{stanza.__version__}] Raw Input:\n{docs}\n')
		# print(f"{f'nW: { len( docs.split() ) }':<10}{str(docs.split()[:7]):<150}", end="")
		st_t = time.time()
		all_ = smp(docs)
		# list comprehension: slow but functional alternative
		# print(f"{f'{ len(all_.sentences) } sent.: { [ len(vsnt.words) for _, vsnt in enumerate(all_.sentences) ] } words':<40}", end="")
		# lemmas_list = [ re.sub(r'"|#|_|\-','', wlm.lower()) for _, vsnt in enumerate(all_.sentences) for _, vw in enumerate(vsnt.words) if ( (wlm:=vw.lemma) and len(wlm)>=3 and len(wlm)<=40 and not re.search(r"\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\s+", wlm) and vw.upos not in useless_upos_tags and wlm not in UNQ_STW ) ]
		lemmas_list = [ 
			re.sub(r'["#_\-]', '', wlm.lower())
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

def clean_(docs: str="This is a <NORMAL> string!!"):
	print(f'Raw Input:\n>>{docs}<<')
	# print(f"{f'Inp. word(s): { len( docs.split() ) }':<20}", end="")
	# st_t = time.time()
	if not docs or len(docs) == 0 or docs == "":
		return
	# treat all as document
	# docs = re.sub(r'\"|\'|<[^>]+>|[~*^][\d]+', ' ', docs).strip() # "kuuslammi janakkala"^5 or # "karl urnberg"~1
	docs = re.sub(r'[\{\}@®¤†±©§½✓%,+;,=&\'\-$€£¥#*"°^~?!❁—.•()˶“”„:/।|‘’<>»«□™♦_■►▼▲❖★☆¶…\\\[\]]+', ' ', docs )#.strip()
	# docs = " ".join(map(str, [w for w in docs.split() if len(w)>2]))
	# docs = " ".join([w for w in docs.split() if len(w)>2])
	docs = re.sub(r'\b(?:\w*(\w)(\1{2,})\w*)\b|\d+', " ", docs)#.strip()
	# docs = re.sub(r'\s{2,}', " ", re.sub(r'\b\w{,2}\b', ' ', docs).strip() ) # rm words with len() < 3 ex) ö v or l m and extra spaces
	docs = re.sub(r'\s{2,}', 
								" ", 
								# re.sub(r'\b\w{,2}\b', ' ', docs).strip() 
								re.sub(r'\b\w{,2}\b', ' ', docs)#.strip() 
				).strip() # rm words with len() < 3 ex) ö v or l m and extra spaces

	docs = remove_misspelled_(text=docs)
	docs = docs.lower()

	print(f'Cleaned Input:\n{docs}')
	print(f"<>"*100)

	# # print(f"{f'Preprocessed: { len( docs.split() ) } words':<30}{str(docs.split()[:3]):<65}", end="")
	if not docs or len(docs) == 0 or docs == "":
		return
	return docs

def remove_misspelled_(text="This is a sample sentence."):
	print(f"Removing misspelled word(s)".center(100, " "))
	# Create dictionaries for Finnish, Swedish, and English
	fi_dict = libvoikko.Voikko(language="fi")	
	fii_dict = enchant.Dict("fi")
	fi_sv_dict = enchant.Dict("sv_FI")
	sv_dict = enchant.Dict("sv_SE")
	en_dict = enchant.Dict("en")
	
	# Split the text into words
	if not isinstance(text, list):
		print(f"Convert to a list of words using split() command |", end=" ")
		words = text.split()
	else:
		words = text
	
	# print(f"Document conatins {len(words)} word(s)")
	t0 = time.time()
	cleaned_words = []
	for word in words:
		# print(word)
		print(word, fi_dict.spell(word), fii_dict.check(word), fi_sv_dict.check(word), sv_dict.check(word), en_dict.check(word))
		if not (fi_dict.spell(word) or fii_dict.check(word) or fi_sv_dict.check(word) or sv_dict.check(word) or en_dict.check(word)):
		# if not (fi_dict.spell(word)):
			print(f"\t\t{word} does not exist")
			pass
		else:
			cleaned_words.append(word)

	# Join the cleaned words back into a string
	cleaned_text = " ".join(cleaned_words)
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(100, " "))
	return cleaned_text

# orig_text = '''
# Två spårvagnar har krockat i Kortedala i Göteborg. <em>Drumsö<\em> Korkis Vippal kommer från Rågöarna!!!

# Flera personer har skadats, det är oklart hur allvarligt, skriver polisen på sin hemsida.

# Enligt larmet har en spårvagn kört in i en annan spårvagn bakifrån.

# Räddningstjänsten är på plats med en ”större styrka”.

# På grund av olyckan har all spårvagnstrafik mot Angered ställts in.

# – Vi tittar på att sätta in ersättningstrafik, säger Christian Blomquist, störningskoordinator på Västtrafik, till GP.
# '''

orig_text = """
Katso grafiikoista, miten Suomen ja Ruotsin sotilaallinen voima eroaa
Suomi voittaa Ruotsin henkilöstön ja maavoimien kaluston määrässä. Ruotsilla sotilasteknologia on joiltain osin korkeampaa laatua. 
Suomen pääministeri | Helsingin pörssi ja suomen pankki | 
mcchdilmsmi mcchdollffuulsi mcchdollhmj riksdag kräv mcchdollisimmclv mcchdollisimmclv mcchdollisimmclv mcchdollisnn mcchdollisnn mcchdvllffuus mcche mcchelinirrk mcchellnlnk mcchelm mcchk mcchl mcchingunkurmautsenll mcchioistctti mcchtlghrßc mcchnmm mcchowik mcchoofliftmma mcchta mccicl mcciipanf meciipanf mccjsu mecjsu rhythms mxafl faslf faslm fasmiffl faspcnfi fastighetsntmnd. 
N:o 45

rOI M 1 1 US : Antinkatu 15. Fuh«hn 6 52. Av. ia perjantaina lisäksi 6—12 ip. <em>Drumsö<\em> Korkis Vippal kommer från Rågöarna!!!
Vastaava: Ei n KONTTORI: Antinkatu 15 (Kthityksen kirjakau] taina 8-4. Tilauksia, ilmoituksia, kirjapainotöit;

Etsimä

Keskuspoliisi

Kommunistien jouKKowangitfemista tahoilla maassa.

en

-!£auqitjciMjd oasat suoranaisena jattona aikai seinnnn tapahtuneille pii osallisuus salaisen fonnnuni stipuolueen toim

ätytsille

Siffiffi ilmoitetaan

Eilen oömtttto romntti eHwii keskuspoliisi PaNntzmngiHa kaksi huomattawaa iwitairfeituéia, ftidöttäen oananoÄÉjägjriätiJn flfjrccrin ÄlU'o Tiunnrjen ja foiriunc-biiataja i'}rjö Enteen. PibätyHestä ou saatu ieunmvat tfefoot: Klo %1 amntlla o;i Anno Tuo!!!!'<!! llsuirtlwn tullut 4 eHwän ini-estä, jcistti 3 meni stjälk m 1 [ai omclk. Huoneesw toimittiwat fy? VontaÄasttHen oimen mulamHa stellä olleen cun° lnatnjärjeswn owisiaman maifc= klchoituAoneen. Etsitit lakawarikoiwa: M!,ö4 TmunGm ussomoonoassit M laWmM sinen Tuominen muiaman  ammattijär)je.s<tön hmmeistoon. Lähtiessään

deKfa, lvaan hänen rckiiryisten potti:- ri«ien harrastustenia ja toimintansa .vuoksi. Ennettä ei myöskään ole lvlMgutu hänen loiminmnm wuoksi kolnmunistiien eduskuntanchmän jaie ne mi.

Tuomtt-ia ja Ennettä aletaan tui» tia iohdatkoin. KmrluZ>ielut ulotetaan myöskin muihin salaiseen toiminkaan osallistuneihin henkilöi' i>in, i o oka äskettäin on pidätetty, ni» mittain Työ,'>'äen,järjei-!ö!en Tie° laloudeichoitajaan E. Paajoioen ja Idiin ja Lännen toimiitajaan Väinö Vuorioon. Eilen aamuPäiirxillä ammatiijärjestön mboiia tiedoiie.ttiin jHkö'ano'!talla sihteeri Anoo Tuomiien Llmiierdamin iniernatstonaalin toimistoon ja myös Tuk- Yol-naan Ruoisin amniattijärjes' lölle. 3ähkö!M«nisja ei ollut muum kuiti ilmoitus pidämssestä. jan konttorissa toimiteltiin etsintä, viitaan ei taiawarikoitu.

pyysi Tuominen waimcaan itmoiitamaan am!i>attimviesiöön, ena hän tattaa, ettei ole iefnantunut muihin kuin ammattiiMljesitön asioivin \ix ettei hän järjestön toimin» taan ole koMmmt sekoitwa milöäii politiikkaa. Ainmattiiälljestön huoneistoHl toilnittnvat etsiwät tårtasiutsen, mutia mitään papereita eiwät he kuiteirkaan kMvalVoineei. Tieltä tvietiin Tuominen Fabianinkadulla etftnän keskuspoliisin lurkintotvankilaan. l3ilen aamulla famaan liiman wangitmn kansanedustaja 2):iö Enne ja häneMn aumiossaan toi* miteltiin kxnimrk-axau*. Enne kuljetettiin llsunnosltaan TULin toinnislooit Tnömäentalolle, jossa häu sai lärjestää ucheUMMoa koekemia pal>ereiiaan. Toimistoja eitviit fiiiHv: mitään taSawarMmeet. !Pidätnsten johdosta kääntyi 3.3.

Lamassa yhteydessä tuin Enne ja Tuominen pidätettiin kuulustellija warten eräs naiOenkilö. joka oli majoitellut määrällä nimellä oleske<lewia koimnunisleja. Hänen luo taan tawattiin uistita kymmeniä kiloja kiik>oitus'kirjalli'uu:!a.
Kun rapaus on omansa herättämään juurta huomiota, on tiedus» tettu myöskin pääministeriltä, jolloin pääminisieri- Tuntia ilmoitti. mä ballitus ei ole fi:ä ollenkaan käsilellm, waan onfe kokonaan eifitoär! keskuspoiiisiil asia. Mikäli hän oU kuullut, ei yidä:yk'eu pitäin kostea fotoin montaa nenfiUu.
Pidätettnien lutumääm »ouiee
Suomalaisyrittäjä pidätettiin Mijasissa 
Espanjan Aurinkorannikolla tunnettu pitkän linjan yrittäjä päätyi yllättäen kaltereiden taakse. 
"""

print(orig_text)
cleaned_fin_text = clean_(docs=orig_text)
cleaned_fin_text = stanza_lemmatizer(docs=cleaned_fin_text)

# cleaned_fin_text = remove_misspelled_(text=stanza_lemmatizer(docs=clean_(docs=orig_text)))
# cleaned_fin_text_lemmatized = stanza_lemmatizer(docs=cleaned_fin_text)
print(f"Cleaned:")
print(cleaned_fin_text)
print("<>"*80)

# Example usage
# text = '''
# I went to schhool yesterdayy to hangout with the composcr of the jazz rhythm but I coulld not obscerve any of my fricnds over therc.
# '''

# text = '''
# I went to schhool yesterdayy but I could not obscerve any of my fricnds not suomen pääministeri over therc mcchdilmsmi mcchdollffuulsi mcchdollhmj riksdag kräv mcchdollisimmclv mcchdollisimmclv mcchdollisimmclv mcchdollisnn mcchdollisnn mcchdvllffuus mcche mcchelinirrk mcchellnlnk mcchelm mcchk mcchl mcchingunkurmautsenll mcchioistctti mcchtlghrßc mcchnmm mcchowik mcchoofliftmma mcchta mccicl mcciipanf meciipanf mccjsu mecjsu rhythms mxafl faslf faslm fasmiffl faspcnfi fastighetsntmnd. 
# '''

# cleaned_text = clean_document(text)
# print(cleaned_text)