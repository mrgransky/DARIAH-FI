from utils import *

# run using nohup:
# nohup python -u parallel_rest_api.py > rest_api_all_length_batch.out 2>&1 &

# run code:
# python parallel_rest_api.py --vbfpath ~/datasets/Nationalbiblioteket/dataframes_x732/concatinated_732_SPMs_lm_stanza_spMtx_x_9777748_BoWs.json --numslices 1000

# import numpy as np
# import urllib
# import time
# import re

# import aiohttp
# import asyncio

# from typing import List, Set, Dict, Tuple

# import requests
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry
# session = requests.Session()
# retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
# session.mount('https://', HTTPAdapter(max_retries=retries))

# TODO: add parser for vb_fpath:
parser = argparse.ArgumentParser(
	description='User-based Recommendation System developed based on National Library of Finland (NLF) dataset', 
	prog='RecSys Concatenated DFs', 
	epilog='Developed by Farid Alijani',
)
parser.add_argument(
	'--vbfpath',
	type=str, 
	required=True,
	help='Path to vocab.json',
)

parser.add_argument(
	'--numslices',
	type=int, 
	default=600,
	help='Number of Slices to batch large list (def: 100)',
)

args = parser.parse_args()

vb = load_vocab(fname=args.vbfpath)
dataset_fpath = "/".join(args.vbfpath.split("/")[:-1])
print(dataset_fpath)

SEARCH_QUERY_DIGI_URL: str = "https://digi.kansalliskirjasto.fi/search?requireAllKeywords=true&query="
DIGI_HOME_PAGE_URL : str = "https://digi.kansalliskirjasto.fi"

headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}
payload = {
	"authors": [],
	"collections": [],
	"districts": [],
	"endDate": None,
	"exactCollectionMaterialType": "false",
	"formats": [],
	"fuzzy": "false",
	"hasIllustrations": "false",
	"importStartDate": None,
	"importTime": "ANY",
	"includeUnauthorizedResults": "false",
	"languages": [],
	"orderBy": "RELEVANCE",
	"pages": "",
	"publicationPlaces": [],
	"publications": [],
	"publishers": [],
	"queryTargetsMetadata": "false",
	"queryTargetsOcrText": "true",
	"searchForBindings": "false",
	"showLastPage": "false",
	"startDate": None,
	"tags": [],
}

TOKENs_num_NLF_pages = None

# small:
# QUERY_LIST = ['finland', 'öppen', 'komma', 'svensk', 'fartyg', 'regering', 'meddela', 'ryssland', 'övrig', 'london', 'hjältegrav', 'runeberg', 'engelsk', 'vicehär', 'hålla', 'finnas', 'tyskland', 'blott', 'suppleant', 'skriva', 'plats', 'fråga', 'kvinna', 'ärkebiskop', 'färgblindhet', 'stockholm', 'jordfästning', 'hjält', 'finländsk', 'england', 'helsingfors', 'koffert', 'hjälp', 'person', 'besättning', 'imperialism', 'arbetare', 'ledare', 'sänka', 'förening', 'bombardera', 'flykting', 'intresse', 'köpenhamn', 'korrespondent', 'redan', 'procession', 'gammal', 'kyrka', 'folkhjälp', 'medlem', 'släkting', 'värld', 'sphinx', 'styrelse', 'representant', 'sverige', 'neutral', 'understöd', 'vapenbroder', 'vänna', 'soldat', 'förhållande', 'lämna', 'besluta', 'yttervärld', 'sända', 'arbete', 'befolkning', 'maskin', 'följande', 'sydvästkust', 'människa', 'eriksgata', 'förvaltningsråd', 'växla', 'förening', 'insamling', 'belgien', 'vacker', 'ångare', 'strid', 'nämligen', 'utrikesminister', 'spansk', 'kreditanstalt', 'besiktning', 'allmän', 'militär', 'flygare', 'naturligtvis', 'utlåning', 'ingenjör', 'främst', 'börja', 'norrie', 'teatern', 'hembygd', 'amerik', 'hemtrakt']

# # medium:
# QUERY_LIST = ['osake', 'agentuuri', 'huoltotili', 'ahvionsaari', 'hyvittää', 'grundlagsutskottet', 'osuuskassa', 'varioja', 'palovakuutuslaitos', 'huoltokonttori', 'maakuntai', 'antolainausehto', 'osuuskauppaväki', 'pankkiliike', 'korko', 'luotto', 'siltasaarenko', 'jordbruksmaskiner', 'uskela', 'rahalaitos', 'laivapäällystö', 'sinisalo', 'kansallis', 'kassa', 'vakavarainen', 'talletustili', 'pulikko', 'käyttäjä', 'vienti', 'osakeyhtiö', 'ottolaina', 'automa', 'keskus', 'slags', 'föreslag', 'siirtokansa', 'vakuutusvirkailija', 'laululintu', 'osuustoimintakuoro', 'vuorimiehenkatu', 'kpllä', 'pääomatili', 'maksupäivä', 'täydellinen', 'hallitus', 'helsinki', 'vuosi', 'tampere', 'pikkulintu', 'säästövara', 'fabritius', 'tarkoitus', 'vallila', 'redskap', 'veloittaa', 'pääoma', 'säästöpankki', 'siirtomaatavara', 'puolivuosittain', 'seutu', 'töölö', 'velkakirjasaatava', 'hartikainen', 'haliko', 'toiminta', 'lounaissuomi', 'luiskahtaa', 'juhani', 'sähköosoite', 'toimia', 'keskuslainarahasto', 'tehtävä', 'lisäys', 'hukkua', 'kuiti', 'puristus', 'kokous', 'häkki', 'teatteri', 'perustaa', 'yleishyödyllinen', 'pohjola', 'tarvike', 'postitse', 'osuuskassanhoitaja', 'henkivakuutus', 'kerääminen', 'haarakonttori', 'konttori', 'sanottavasti', 'saada', 'honkala', 'karjala', 'täydelleen', 'yrjönkatu', 'liikeapulainen', 'sijoittaminen', 'mikonkatu', 'edullisesti', 'esittää', 'johtaja', 'jäsenmäärä', 'fimbenne', 'toimintataipaleetta', 'sopiva', 'maksu', 'kuulumiset', 'ompelu', 'tilinpito', 'shekinkäyttöoikeus', 'suorittaa', 'pitää', 'gaspe', 'vakuuskysymys', 'isotupa', 'harjoittaa', 'kertomusvuonna', 'säänöinen', 'jäsenkassa', 'maalaisapteekki', 'hmiti', 'hanki', 'huolto', 'maanviljelijä', 'sanomalehtimies', 'keskusliitto', 'velkakirja', 'esitelmä', 'turku', 'vankka', 'vaatia', 'lunastaa', 'eilen', 'selänne', 'markka', 'eloma', 'asiakas', 'perustaja', 'lakka', 'päälle', 'aleksanterinkatu', 'murto', 'tunma', 'agentuuriliike', 'maksaa', 'laivanselvitys', 'kevätkokous', 'osuuskassi', 'luona', 'jäsen', 'hajapiirre', 'suuri', 'jäsenmaksu', 'luennoima', 'muinaishistoria', 'nivelkipu', 'liike', 'tilinpäätös', 'tällöin', 'kallio', 'matta', 'luotontarvi', 'juusto', 'osuustoimintaperiaate', 'lisääntyki', 'tulla', 'naurunremahdus', 'toteuttaminen', 'viitanen', 'soini', 'illanvietto', 'talousalue', 'lehtimäki', 'tavallisesti', 'tenttinen', 'kiinteä', 'turva', 'myöntää', 'lointaa', 'omari', 'osuuskassamies', 'päättää', 'edullinen', 'ottaa', 'ansiokkainen', 'kuulua', 'länsi', 'koskemintaa', 'maito', 'keskustella', 'myydä', 'juokseva', 'parhaiten', 'juhtaa', 'silloin', 'uudistushanke', 'huolinta', 'loppusumma', 'kamppi', 'osuuskassaväki', 'kehitys', 'välittää', 'ylitarkastaja', 'lähettää', 'kurssi', 'rangelli', 'elämänrohkeus', 'tuomari', 'järjestö', 'perustamispäivä', 'kulua', 'pakkari', 'osuuskassaliikki', 'santaoja', 'hoito', 'halikko', 'viime', 'hedelmä', 'toimihenkilö', 'kania', 'reumatismi', 'valita', 'henkilötyyppi', 'ryhtyä', 'jämsä', 'dahlbergi', 'taara', 'puheenjohtaja', 'johtokunta', 'taitella', 'tilintarkastaja', 'tuomela', 'sahlstedt', 'liitto', 'mainita', 'käyttää', 'avoinna', 'toistaiseksi', 'luonnollinen', 'eteläranta', 'tapahtuma', 'lasipalatsi', 'sähinen', 'hyväksyä', 'tuure', 'täysi', 'käräjä', 'päivä', 'lisätä', 'seurata', 'hinta', 'meijeri', 'liikemuoto', 'jakaa', 'tikka', 'ilmoittaa', 'antaa', 'koskeminen', 'hoitokulu', 'alkaa', 'valitsemi', 'kerta', 'piiri', 'nitoa', 'osasto', 'omistaja', 'yhteys', 'määrä', 'rahtaus', 'savonlinta', 'lainata', 'erovuoro', 'osoittaa', 'suoruus', 'esitys', 'erikoinen', 'kuten', 'hanna', 'yksityinen', 'laajentaminen', 'lindegre', 'asianomainen', 'paljon', 'jalka', 'hermokivu', 'laine', 'savonlinna', 'erottaja', 'seuraintalo', 'vuosiylijäämä', 'vuosikertomus', 'edellinen', 'kunnallislehti', 'kannatusmaksu', 'sääminki', 'aarne', 'historiikka', 'nopeasti', 'lukea', 'sihteeri', 'kiinteäkorkoinen', 'yliopistonkatu', 'tanssi', 'käsitellä', 'pirstoutua', 'unettomuus', 'sikatalous', 'maatalous', 'munuainen', 'mahdollinen', 'edelleen', 'tärkeä', 'kruunuhaka', 'kangas', 'kosmeettinen', 'isännöitsijä', 'talletuspaikka', 'odottaa', 'kuorittua', 'lyhyttavara', 'nykytärkeä', 'yleisö', 'hämeentie', 'seura', 'paikka', 'pysyvästi', 'hoitaa', 'osuuskunta', 'tuoda', 'kasso', 'bulevardi', 'iskias', 'alusia', 'tapaan', 'tarkkailija', 'kunta', 'togali', 'tyydyttää', 'vaikuttaa', 'osuusteurastamo', 'toimi', 'alennus', 'jokinen', 'käytyä', 'uudelleen', 'perustamiskirja', 'lähteä', 'varajäsen', 'maksuttomasti', 'paras', 'toimintapiiri', 'metsäalue', 'kerimäki', 'perniö', 'nuorisoseuralainen', 'puoli', 'virtsahappo', 'viinamäki', 'tukkuliike', 'kihti', 'duetto', 'ranki', 'keskikorko', 'numero', 'lopuksi', 'valittu', 'kiikala', 'konttorikone', 'yolanda', 'järjestää', 'alennusmyynti', 'ohjelma', 'vararahasto', 'paikkakunta', 'osuusmaksu', 'pertteli', 'näytelmä', 'kankea', 'tabletti', 'joutua', 'kuolla', 'seuraava', 'kirjapaino', 'kaitila', 'piiritarkastaja', 'opettaja', 'tilivuosi', 'miime', 'autotarvike', 'luennoida', 'lämminhenkinen', 'autonjäähdyttäjätehdas', 'kuulla', 'seppälä', 'päänsärky', 'tarkastaja', 'pyytää', 'suuruinen', 'suunnitella', 'tarkkailu', 'entinen', 'teollisuustarvike', 'osanottaja', 'historiikki', 'netti', 'rautakauppatavaro', 'astua', 'tekstiilitavara', 'räjähtää', 'rahasto', 'tehdä', 'ruumis', 'keskihinta', 'erittää', 'lausunto', 'juhlavieras', 'liikevaihto', 'merkkitapaus', 'tilapäisesti', 'lausua', 'solmiotehdas', 'onnistuneesti', 'kansantanhu', 'merkityksellinen', 'pyyntö', 'alunperin', 'resepti', 'toimittaa', 'sikala', 'ammatillinen', 'paavo', 'toimitalo', 'kehittyä', 'käsinetehdas', 'vapaa', 'loimaa', 'osuuskauppa', 'valtio', 'luennoitsija', 'huomattava', 'kuisma', 'makeistehdas', 'vastata', 'lääke', 'viikko', 'päättyä', 'soitin', 'onnistua', 'pietarsaari', 'juosta', 'jäädä', 'pajula', 'joten', 'tänään', 'käytäntö', 'muuttaa', 'teknokem', 'jatkua', 'laajakantoinen', 'sitten', 'penni', 'laaja', 'rahamäärä', 'obstbaum', 'lieventää', 'kutomateollisuustuste', 'tamperelainen', 'vekseli', 'kiintoisa', 'nasta', 'joukko', 'vaaraton', 'arkipäivä', 'laulu', 'maatalousnäyttely', 'loppu', 'hannu', 'kysymys', 'avata', 'kansa', 'suunta', 'ravintola', 'mieltymys', 'kaava', 'kolkontaipale', 'laukansaari', 'alkuaika', 'pienviljelijä', 'lausunta', 'leipomo', 'ostaa', 'valaiseva', 'vahingollinen', 'osuustoiminta', 'juhlapäivä', 'apteeki']

# # large:
QUERY_LIST = list(vb.keys())[:37690]
print(len(QUERY_LIST))

async def get_recommendation_num_NLF_pages_async(session, REC_TK: str="pollution"):
	st_t = time.time()
	URL = f"{SEARCH_QUERY_DIGI_URL}" + urllib.parse.quote_plus(REC_TK)
	parsed_url = urllib.parse.urlparse(URL)
	parameters = urllib.parse.parse_qs(parsed_url.query, keep_blank_values=True)
	offset_pg = (int(re.search(r'page=(\d+)', URL).group(1)) - 1) * 20 if re.search(r'page=(\d+)', URL) else 0
	search_page_request_url = f"{DIGI_HOME_PAGE_URL}/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	payload["query"] = parameters.get('query', [''])[0] # if unavailable => ""
	payload["requireAllKeywords"] = parameters.get('requireAllKeywords', ['false'])[0] # if unavailable => "false"

	try:
		async with session.post(
			url=search_page_request_url,
			json=payload,
			headers=headers,
		) as response:
				response.raise_for_status()
				res = await response.json()
				TOTAL_NUM_NLF_RESULTs = res.get("totalResults")
				# print(f"Found NLF tot_page(s): {TOTAL_NUM_NLF_RESULTs:<10} in {time.time() - st_t:.1f} sec")
				return TOTAL_NUM_NLF_RESULTs
	except (
			aiohttp.ClientError,
			asyncio.TimeoutError,
		) as e:
			# print(f"<!> ERR < {e} > URL: {URL}")
			return

async def get_num_NLF_pages_asynchronous_run(TOKENs_list: List[str] = ["tk1", "tk2"]):
	async with aiohttp.ClientSession() as session:
		tasks = [
			NUMBER_OF_PAGES
			for tk in TOKENs_list
			if (
				NUMBER_OF_PAGES:=get_recommendation_num_NLF_pages_async(session, REC_TK=tk)
			)
		]
		num_NLF_pages = await asyncio.gather(*tasks)
	return num_NLF_pages

def get_num_NLF_pages(INPUT_QUERY: str="query", INPUT_TOKEN: str="token"):
	st_t = time.time()
	URL = f"{SEARCH_QUERY_DIGI_URL}" + urllib.parse.quote_plus(INPUT_QUERY + " " + INPUT_TOKEN)
	print(f"{URL:<150}", end=" ")
	parsed_url = urllib.parse.urlparse(URL)
	parameters = urllib.parse.parse_qs( parsed_url.query, keep_blank_values=True)
	offset_pg = ( int( re.search(r'page=(\d+)', URL).group(1) )-1)*20 if re.search(r'page=(\d+)', URL) else 0
	search_page_request_url = f"{DIGI_HOME_PAGE_URL}/rest/binding-search/search/binding?offset={offset_pg}&count=20"
	payload["query"] = parameters.get('query')[0] if parameters.get('query') else ""
	payload["requireAllKeywords"] = parameters.get('requireAllKeywords')[0] if parameters.get('requireAllKeywords') else "false"
	try:
		r = session.post(
			url=search_page_request_url,
			json=payload,
			headers=headers,
		)
		r.raise_for_status()  # Raise HTTPError for bad status codes
		res = r.json()
		TOTAL_NUM_NLF_RESULTs = res.get("totalResults")
		print(f"NLF tot_page(s): {TOTAL_NUM_NLF_RESULTs:<6} elapsed_t: {time.time()-st_t:.1f} s")
	except (
			requests.exceptions.RequestException
		) as e:
		# print(f"<!> Error: {e}")
		return
	return TOTAL_NUM_NLF_RESULTs

# # OLD inefficient implementation: 
# start_time = time.time()
# TOKENs_num_NLF_pages = [
# 	# NUMBER_OF_PAGEs
# 	get_num_NLF_pages(INPUT_QUERY=MY_QUERY_PHRASE, INPUT_TOKEN=tk)
# 	for tk in my_list
# 	# if(
# 	# 	(NUMBER_OF_PAGEs:=get_num_NLF_pages(INPUT_QUERY=MY_QUERY_PHRASE, INPUT_TOKEN=tk))
# 	# )
# ]
# print(f"< Traditional approach > TOTAL Elapsed_t: {time.time()-start_time:.1f} sec | Initial TK list: {len(my_list)} | {len(TOKENs_num_NLF_pages)} PAGE(s)")
# print(TOKENs_num_NLF_pages)
# print("#"*100)

# # asynchronous implementation: efficient for small and medium sized lists:
# print(f"< Asynchronous Approach >")
# start_time_async = time.time()
# TOKENs_num_NLF_pages_async = asyncio.run(
# 	get_num_NLF_pages_asynchronous_run(
# 		qu=MY_QUERY_PHRASE, 
# 		TOKENs_list=MY_LIST, 
# 	)
# )
# print(
# 	f">>> TOTAL Elapsed_t: {time.time() - start_time_async:.1f} sec "
# 	f"Initial TK list: {len(MY_LIST)} | {len(TOKENs_num_NLF_pages_async)} PAGE(s) | "
# 	f"nZero(s): {sum(1 for item in TOKENs_num_NLF_pages_async if item==0)} | nNaN(s): {sum(1 for item in TOKENs_num_NLF_pages_async if item is None)}"
# )
# print(len(TOKENs_num_NLF_pages_async), TOKENs_num_NLF_pages_async)

# asynchronous implementation: efficient for LARGE sized lists (>10K):
def get_zero_nlf_pages_asynch_fcn(MY_LIST: List, slices: int=100):
	prev_s = 0
	tk_zero_nlf_pg = list()
	tk_NONEs_nlf_pg = list()
	total_num_batches = int(len(MY_LIST)/slices)+1

	for s in range( total_num_batches ):
		print(f"[batch: {s+1}/{total_num_batches}] previous slice: {prev_s} current slice: {prev_s+slices}")
		my_list = MY_LIST[prev_s:prev_s+slices]
		print(f"< Asynchronous approach > for a list of {len(my_list)} elemensts...")
		start_time_async = time.time()
		TOKENs_num_NLF_pages_async = asyncio.run(
			get_num_NLF_pages_asynchronous_run(
				TOKENs_list=my_list,
			)
		)
		print(
			f"\t>>> Elapsed_t: {time.time() - start_time_async:.1f} sec "
			f"Initial TK list: {len(my_list)} | {len(TOKENs_num_NLF_pages_async)} PAGE(s) | "
			f"nZero(s): {sum(1 for item in TOKENs_num_NLF_pages_async if item==0)} | "
			f"nNaN(s): {sum(1 for item in TOKENs_num_NLF_pages_async if item is None)}"
		)
		prev_s += slices
		# print(len(TOKENs_num_NLF_pages_async), TOKENs_num_NLF_pages_async)
		# print(prev_s)
		# print(my_list)
		tk_zero_nlf_pg_per_slice = [tk for tk, num in zip(my_list, TOKENs_num_NLF_pages_async) if num==0]
		tk_zero_nlf_pg.extend(tk_zero_nlf_pg_per_slice)
		print(
			f"accumulated_ZEROs: {len(tk_zero_nlf_pg)} "
			f"{tk_zero_nlf_pg[-7:]}"
		)
		tk_NONEs_nlf_pg_per_slice = [tk for tk, num in zip(my_list, TOKENs_num_NLF_pages_async) if num is None]
		tk_NONEs_nlf_pg.extend(tk_NONEs_nlf_pg_per_slice)
		print(
			f"accumulated_NONEs: {len(tk_NONEs_nlf_pg)} "
			f"{tk_NONEs_nlf_pg[-7:]}"
		)
		print("-"*180)
	return tk_zero_nlf_pg, tk_NONEs_nlf_pg

def get_all_unique_elements(*args: List[Union[int, float, str]]) -> List[Union[int, float, str]]:
	"""
	Returns a list containing all unique elements from an arbitrary number of input lists.

	Args:
			*args: A variable number of lists containing elements.

	Returns:
			A list containing all unique elements from the input lists.
	"""
	# Combine all elements from input lists into a single set
	all_elements = set(element for arg in args for element in arg)
	return list(all_elements)
##########################################################################################################
# nones will not be taken care of!
# tokens_with_zero_nlf_pages, tokens_with_none_nlf_pages = get_zero_nlf_pages_asynch_fcn(MY_LIST=QUERY_LIST)
# print(
# 	len(total_tokens_with_zero_nlf_pages), 
# 	total_tokens_with_zero_nlf_pages[:150],
# )
##########################################################################################################

# Send NONEs to REST API over and over until disappear!:
print(f">> while loop to get rid of Nones...")
num_TOKENS_NONE_NLF_PAGES = np.inf
qu_lst = QUERY_LIST
total_tokens_with_zero_nlf_pages = list()
while num_TOKENS_NONE_NLF_PAGES > 0:
	current_tokens_with_zero_nlf_pages, current_tokens_with_none_nlf_pages = get_zero_nlf_pages_asynch_fcn(MY_LIST=qu_lst, slices=args.numslices)
	total_tokens_with_zero_nlf_pages.extend(current_tokens_with_zero_nlf_pages)
	qu_lst = current_tokens_with_none_nlf_pages
	num_TOKENS_NONE_NLF_PAGES = len(current_tokens_with_none_nlf_pages)
	print(f"Found {num_TOKENS_NONE_NLF_PAGES} tokens which resulted in None!".center(150, " "))


print(
	len(total_tokens_with_zero_nlf_pages), 
	total_tokens_with_zero_nlf_pages[:150],
)

# TODO: if some meaningless already available, extend and union:
avail_zero_nlf_pages_list = list()
avail_0_nlf_pages_filepath_list = get_files(pth=dataset_fpath+'/'+'tk_x_*_with_zero_NLF_pages.gz')
if len(avail_0_nlf_pages_filepath_list) > 0:
	print(f"Found {len(avail_0_nlf_pages_filepath_list)} {type(avail_0_nlf_pages_filepath_list)} list(s) of zero NLF page files".center(100, "-"))
	for fpath in avail_0_nlf_pages_filepath_list:
		print(fpath)
		lst = load_pickle(fpath=fpath)
		avail_zero_nlf_pages_list.extend(lst)

	print(f"Found {len(avail_zero_nlf_pages_list)} avail_zero_nlf_pages_list: {avail_zero_nlf_pages_list[:10]}")
	duplicate_counts_avail_0_nlf_pages = Counter(avail_zero_nlf_pages_list)
	# print(counts)
	total_duplicates_avail_0_nlf_pages = sum(duplicate_counts_avail_0_nlf_pages>1 for duplicate_counts_avail_0_nlf_pages in duplicate_counts_avail_0_nlf_pages.values())
	print(f"\t\t>>>> duplicates [available zero nlf pages list]: {total_duplicates_avail_0_nlf_pages}")

	total_tokens_with_zero_nlf_pages = get_all_unique_elements(total_tokens_with_zero_nlf_pages, avail_zero_nlf_pages_list)
	print(f"Found {len(total_tokens_with_zero_nlf_pages)} tot_zero_nlf_pages_list: {total_tokens_with_zero_nlf_pages[:10]}")

	duplicate_counts_total_tokens_with_zero_nlf_pages = Counter(total_tokens_with_zero_nlf_pages)
	# print(counts)
	total_duplicates_tot_0_nlf_pages = sum(duplicate_counts_total_tokens_with_zero_nlf_pages>1 for duplicate_counts_total_tokens_with_zero_nlf_pages in duplicate_counts_total_tokens_with_zero_nlf_pages.values())
	print(f"\t\t>>>> duplicates [< TOTAL > zero nlf pages list]: {total_duplicates_tot_0_nlf_pages} <<<<=== MUST BE ZERO ===>>>>")

	print(f"#"*100)

save_pickle(
	pkl=total_tokens_with_zero_nlf_pages,
	fname=os.path.join(dataset_fpath, f"tk_x_{len(total_tokens_with_zero_nlf_pages)}_with_zero_NLF_pages.gz"),
)

if TOKENs_num_NLF_pages:
	print(np.all(TOKENs_num_NLF_pages==TOKENs_num_NLF_pages_async))