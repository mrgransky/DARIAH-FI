from utils import *

# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
with HiddenPrints():
	import nltk
	nltk_modules = ['punkt',
								'stopwords',
								'wordnet',
								'averaged_perceptron_tagger', 
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
	lang_id_config = {"langid_lang_subset": ['en', 'sv', 'da', 'ru', 'fi', 'de', 'fr']}
	lang_configs = {"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
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
	my_custom_stopwords = ['btw', "could've", "n't","'s","—", "i'm", "'m", 
													"i've", "ive", "'d", "i'd", " i'll", "'ll", "'ll", "'re", "'ve", 'oops', 'tmt', 'ocb', 'rdf',
													'aldiz', 'baizik', 'bukatzeko', 'ift', 'jja', 'lrhe', 'iih', 'rno', 'jgfj', 'puh', 'knr', 'rirrh',
													'klo','nro', 'vol', 'amp', 'sid', 'obs', 'annan', 'huom', 'ajl', 'alf', 'frk', 'albi', 'edv', 'ell',
													'inc', 'per', 'ops', 'vpv', 'ojv', 'rva', 'hvr', 'nfn', 'smk', 'lkm', 'quo', 'utf', 'hfvo', 'mim', 'htnm',
													'edota', 'eze', 'ezpabere', 'ezpada', 'ezperen', 'gainera', 'njj', 'aab', 'arb', 'tel', 'fkhi',
													'gainerontzean', 'guztiz', 'hainbestez', 'horra', 'onların', 'ordea', 'hel', 'ake', 'oeb',
													'osterantzean', 'sha', 'δ', 'δι', 'агар-чи', 'аз-баски', 'афташ', 'бале', 'gen', 'fjh', 'fji',
													'баҳри', 'болои', 'валекин', 'вақте', 'вуҷуди', 'гар', 'гарчанде', 'даме', 'карда', 
													'кошки', 'куя', 'кӣ', 'магар', 'майлаш', 'модоме', 'нияти', 'онан', 'оре', 'рӯи', 
													'сар', 'тразе', 'хом', 'хуб', 'чаро', 'чи', 'чунон', 'ш', 'шарте', 'қадар',
													'ҳай-ҳай', 'ҳамин', 'ҳатто', 'ҳо', 'ҳой-ҳой', 'ҳол', 'ҳолате', 'ӯим', 'באיזו', 'בו', 'במקום', 
													'בשעה', 'הסיבה', 'לאיזו', 'למקום', 'מאיזו', 'מידה', 'מקום', 'סיבה', 'ש', 'שבגללה', 'שבו', 'תכלית', 'أفعل', 
													'أفعله', 'انفك', 'برح', 'سيما', 'कम', 'से', 'ἀλλ',
													'rhr', 'trl', 'amt', 'nen', 'mnl', 'krj', 'laj', 'nrn', 'aakv', 'aal', 'oii', 'rji', 'ynn', 'eene', 'eeni', 'eeno', 'eenj', 'eenip', 'eenk',
													'hyv', 'kpl', 'ppt', 'ebr', 'emll', 'cdcdb', 'ecl', 'jjl', 'cdc', 'cdb', 'aga', 'ldr', 'lvg', 'zjo', 'cllcltl', 
													'åbq', 'ordf', 'cuw', 'nrh', 'mmtw', 'crt', 'csök', 'hcrr', 'nvri', 'disp', 'ocll', 'rss', 'aalr', 'aama', 'aamma', 'aamme', 'aammamm', 'aamne', 'aamnie', 'baxk',
													'dii', 'hii', 'hij', 'stt', 'gsj', 'ådg', 'phj', 'nnw', 'wnw', 'nne', 'sij', 'apt', 'iit', 'juu', 'lut', 'aammä', 'aamnli', 'aamo', 'aamtd', 'aamucri', 'basly',
													'afg', 'ank', 'fnb', 'ffc', 'oju', 'kpk', 'kpkm', 'kpkk', 'krs', 'prs', 'osk', 'rka', 'rnu', 'wsw', 'kic', 'aamwa', 'aander', 'aank', 'aanml', 'aanlcapt',
													'atm', 'vmrc', 'swc', 'fjijcn', 'hvmv', 'arb', 'inr', 'cch', 'msk', 'msn', 'tlc', 'vjj', 'jgk', 'mlk', 'aao', 'aaq', 'aaponp', 'aarincn', 'aarinci', 'aarm', 'barp',
													'ffllil', 'adr', 'bea', 'ret', 'inh', 'vrk', 'ang', 'hra', 'nit', 'arr', 'jai', 'enk', 'bjb', 'iin', 'llä', 'aarn', 'aarnc', 'barnfl', 'barnk', 'barnm', 'barnsj',
													'kka', 'tta', 'avu', 'nbl', 'drg', 'sek', 'wrw', 'tjk', 'jssjl', 'ing', 'scn', 'joh', 'yhd', 'uskc', 'enda', 'aabenfa', 'aabaiajajaeftsß', 'barl', 'barnf', 'barne',
													'tyk', 'wbl', 'vvichmann', 'okt', 'yht', 'akt', 'nti', 'mgl', 'vna', 'anl', 'lst', 'hku', 'wllss','ebf', 'adlj', 'asemap', 'asewa', 'bardv', 'barf', 'barj', 'barwa',
													'jan', 'febr', 'aug', 'sept', 'lok', 'nov', 'dec', 
													'aadof', 'ado', 'adoa', 'aadolrl', 'aado', 'adolj', 'aadojf', 'aadolf', 'aadolfa', 'aadolffi', 'aadolfi', 'aadolfin', 'aadolfink', 'aadol', 'abbado', 'adoll', 'beia',
													'jfe', 'ifc', 'flfl', 'bml', 'zibi', 'ibb', 'bbß', 'sfr', 'yjet', 'stm', 'imk', 'sotmt', 'oslfesfl', 'anoth', 'vmo', 'uts', 'abf', 'aajo', 'aajtus', 'aaka', 'beir',
													'mmik', 'pytmkn', 'ins', 'ifk', 'vii', 'ikl', 'lan', 'trt', 'lpu', 'ijj', 'ave', 'lols', 'nyl', 'tav', 'ohoj', 'kph', 'answ', 'ansv', 'anw', 'anv', 'abjfi', 'beim',
													'ver', 'jit', 'sed', 'sih','ltd', 'aan', 'exc', 'tle', 'xjl', 'iti', 'tto', 'otp', 'xxi', 'ing', 'fti', 'fjg', 'fjfj', 'ann', 'ansk', 'ant', 'abh', 'abjne', 'bein',
													'mag', 'jnjejn', 'rvff', 'ast', 'lbo', 'otk', 'mcf', 'prc', 'vak', 'tif', 'nso', 'jyv','jjli', 'ent', 'edv', 'fjiwsi', 'alns', 'andr', 'anf', 'abja', 'ablfa', 'bekl',
													'psr', 'jay', 'ifr', 'jbl', 'iffi', 'ljjlii', 'jirl', 'nen', 'rwa', 'tla', 'mnn', 'jfa', 'omp', 'lnna', 'nyk', 'fjjg', 'taflhßtiß', 'ablbt', 'aajjp', 'afcn', 'beit',
													'via', 'hjo', 'termffl', 'kvrk', 'std', 'utg', 'sir', 'icc', 'kif', 'ooh', 'imi', 'nfl', 'xxiv', 'tik', 'edv', 'fjj', 'tadiißi', 'taershß', 'taes', 'ablgn', 'beji',
													'pnä', 'pna', 'urh', 'ltws', 'ltw', 'ltx', 'ltvfc', 'klp', 'fml', 'fmk', 'hkr', 'cyl', 'hrr', 'mtail', 'hfflt', 'xix', 'aaltofj', 'aaltl', 'aaltos', 'ablq', 'asel',
													'rli', 'mjn', 'flr', 'ffnvbk', 'bjbhj', 'ikf', 'bbh', 'bjete', 'hmx', 'snww', 'dpk', 'hkk', 'aaf', 'aag', 'aagl', 'fjkj', 'aaloßil', 'aals', 'aaltc', 'afcrtu',
													'aah', 'aai', 'aaib', 'aibli', 'aibtt', 'aibg', 'aibi', 'aib', 'aibe', 'aibo', 'aiau', 'axa', 'aiax', 'axb', 'axd', 'axl', 'axx', 'ayf', 'bxi', 'bxa', 'bxs', 'bya', 
													'aalalshifo', 'aalop', 'afchlefca', 'afct', 'byb', 'bäh', 'öna', 'aifk', 'aifo', 'aifu', 'aify', 'aift', 'aife', 'aiwo', 'aiwi', 'aiwe', 'ajci', 'aschw', 'aschw',
													'byatrßm', 'bygk', 'byhr', 'byi', 'byo', 'byr', 'bys', 'bzj', 'bzo', 'bzt', 'bzw', 'bßt', 'bßß', 'bäb', 'bäc', 'bäd', 'ömv', 'aaks', 'aks', 'aalt', 'aaltoc', 'asci',
													'bäi', 'bäik', 'bådh', 'bѣрa', 'caflm', 'caflddg','campm', 'camplnchl', 'camßi', 'canfh', 'jne', 'ium', 'xxviä', 'xys', 'ömä', 'aakrf', 'aalip', 'aalisv', 'asda',
													'xxbx', 'xvy', 'xwr', 'xxii', 'xxix', 'xxo', 'xxm', 'xxl', 'xxv', 'xxä', 'xxvii', 'xyc', 'xxp', 'xxih', 'xxlui', 'xzå', 'önrcr', 'aakr', 'aaliohclmi', 'aalfie', 'ase',
													'ybd', 'ybfi', 'ybh', 'ybj', 'yca', 'yck', 'ycs', 'yey', 'yfcr', 'yfc', 'yfe', 'yffk', 'yff', 'yfht', 'yfj', 'yfl', 'yft', 'öos', 'aaklx', 'aaliltx', 'aalfco', 'alfc',
													'ynnaxy', 'aoaxy', 'aob', 'cxy', 'jmw', 'jmxy', 'msxy', 'msx', 'msy', 'msßas', 'mszsk', 'mta', 'ius', 'tsb', 'öoo', 'öol', 'aakiint', 'aakk', 'aalci', 'aalc', 'alc',
													'ile', 'lle', 'htdle', 'sir','pai', 'päi', 'str', 'ent', 'ciuj', 'homoj', 'quot', 'pro', 'tft', 'lso', 'sii', 'öot', 'öovct', 'öou', 'aakfxmto', 'aalb', 'aale', 'backst',
													'hpl', 'tpv', 'vrpl', 'rpl', 'jtihiap', 'nii', 'nnf', 'aää', 'jofc', 'lxx', 'vmll', 'sll', 'vlli', 'pernstclln', 'nttä', 'npunl', 'aln', 'aakf', 'aakfco', 'aalj', 'baco',
													'aalflm', 'aaltpi', 'aaltqa', 'aalß', 'aalv', 'aalvt', 'aalw', 'taalß', 'aam', 'laam', 'taam', 'taamctli', 'taan', 'taawatwßki', 'taaßffrr', 'taaßxmm', 'afctitf', 'badc',
													'öjj', 'öjo', 'öio', 'öiibi', 'öij', 'öbb', 'öba', 'åvq', 'åvp', 'åvl', 'åyr', 'åjj', 'åji', 'åjk', 'åff', 'åfgr', 'äåg', 'fjlmi', 'aak', 'aakcl', 'aalg', 'manuf', 'badk',
													'ink', 'ksi', 'ctr', 'dec', 'fmf', 'ull', 'prof', 'sco', 'jjö', 'tcnko', 'itx', 'tcnkck', 'kello', 'jnho', 'infji', 'jib', 'ämj', 'aajr', 'aajwu', 'aald', 'adrn', 'bafv',
													'afu', 'ieo', 'ebep', 'tnvr', 'nta', 'tlyllal', 'viv', 'sån', 'hitl', 'vrt', 'pohj', 'nky', 'ope', 'ftm', 'tfflutti', 'aajaj', 'aajl', 'aalf', '8elegram', 'adrrn', 'bafd',
													'lwiki', 'uhi', 'ffiuiru', 'eji', 'iil', 'äbt', 'llimi', 'efl', 'idbt', 'plchäialm', 'xukkalanka', 'aadjf', 'ime', 'tps', 'aaißia', 'aaißial', 'aöd', 'aije', 'aill', 'bagm',
													'vps', 'tys', 'lto', 'pnnfi', 'nfiaiu', 'ilnm', 'cfe', 'hnmßr', 'pfäfflin', 'svk', 'alfr', 'pka', 'avg', 'ångf', 'arf', 'juh', 'pnjßw', 'aais', 'aaiw', 'aöc', 'ailv', 'bagn',
													'ruu', 'sus', 'rur', 'hden', 'kel', 'ppbcsfä', 'pptepbesfä', 'lof', 'adr', 'siv', 'owa', 'osa', 'aagsnyhttcr', 'aahr','aaif', 'aaig', 'aig', 'aaii', 'aaik', 'talak', 'baft',
													'aabd', 'aabx',  'aabn', 'aabnrxii', 'aabr', 'aabt', 'aabä', 'aad', 'aae', 'aafwvwvn', 'aagsßas', 'aagxt', 'aaiißofl', 'aaiiu', 'aiiu', 'aaijsi', 'acl', 'aimc', 'alfs',
													'adsh', 'adslf', 'advtn', 'aed', 'aeap', 'aeb', 'aeen', 'aef', 'aeev', 'aehi', 'aej', 'aeja', 'aek', 'afc', 'axaafcsr', 'aafc', 'aafcfcfc', 'aafci', 'afci', 'aing', 'bachs',
													'tafllwalta', 'taflßan', 'tafmr', 'taftofße', 'taftpo', 'taftßßl', 'tafv', 'tahßßi', 'taij', 'tailctustlvhß', 'tailcvi', 'tairkkß', 'taiwalkoßk', 'tahmß', 'aip', 'alfv',
													'taj', 'tajtsßßsa', 'takaisinßva', 'takhohßacu', 'takll', 'takm', 'taknn', 'taknmnmnn', 'takpi', 'takußtu', 'takxixl', 'takxjuol', 'takxmcl', 'takxul', 'wähä', 'alfi', 'alfm',
													'talamodh', 'talap', 'talapfi', 'alap', 'talarc', 'talarn', 'talarpla', 'talaslcrh', 'talausjärjcskelniä', 'talblti', 'mäßig', 'wähy', 'eii', 'eln', 'aabso', 'aitm', 'aiwcm',
													'tivßti', 'tiyäoliiacß', 'tiß', 'tlelsrtllß', 'tlerpoaaßoack', 'tlerposaßoack', 'tliatalm', 'yliopizto', 'yliopizco', 'ylioppilazlalo', 'yliopr', 'wuotr', 'wägncr', 'asef',
													'wuotiscn', 'wußtusta', 'wva', 'wuwa', 'wuwm', 'wuwwnctv', 'wveb', 'wver', 'wveu', 'wvi', 'wvix', 'wvji', 'wvl', 'wvn', 'wvo', 'wvre', 'wvrqi', 'wvs', 'wvsfn', 'qan', 'asei',
													'wvstnu', 'wvtba', 'wvu', 'wvv', 'wvviv', 'wvvo', 'wvvw', 'wvw', 'wvwv', 'wvxww', 'wvy', 'wvä', 'wwa', 'wvå', 'wwah', 'wwaja', 'wwb', 'wwcamri', 'wwcchmimh', 'wwy', 'baafi',
													'wwckku', 'wwcpbh', 'wwe', 'wwer', 'wwffl', 'wwg', 'wwfi', 'wwi', 'wwimi', 'wwj', 'wwk', 'wwka', 'wwl', 'wwn', 'wwnnm', 'wwo', 'wwol', 'wwpw', 'wwra', 'wwö', 'wwä', 'baahe',
													'wxn', 'wxm', 'wxos', 'wyi', 'wxs', 'wxu', 'wyj', 'wyk', 'wyl', 'wyn', 'wys', 'wzmvm', 'wzsta', 'wßjf', 'wßw', 'wäa', 'wäe', 'wäezg', 'wäg', 'wäh', 'wähcmi', 'wähär', 'bah',
													'wäi', 'wähöh', 'wäisct', 'wäk', 'laad', 'qaa', 'qad', 'qahn', 'qaj', 'adu', 'afef', 'aahde', 'aait', 'aaky', 'aalu', 'aami', 'aame', 'aalä', 'aamp', 'aaml', 'aamt', 'bac',
													'qamd', 'qamdd',  'zaa', 'zaan', 'zab', 'zad', 'zag', 'zai', 'zahn', 'aikx', 'ail', 'ailcimmc', 'ailcs', 'ailclte', 'ailf', 'ailifci', 'aäh', 'aår', 'ahn', 'aho', 'baak',
													'aßtx', 'aßa', 'aßctl', 'aßfn', 'aßj', 'aßl', 'aßmt', 'aßo', 'aßs', 'aßß', 'aßä', 'aäa', 'aäaa', 'aäak', 'aäaw', 'aäbi', 'aäci', 'aäd', 'aäel', 'aäe', 'aåss', 'ahnj',
													'aäi', 'aäj', 'aälj', 'aän', 'aär', 'aäs', 'aät', 'aäu', 'aäv', 'aäy', 'aäö', 'aåaana', 'aåad', 'aåan', 'aådt', 'aåe', 'aåg', 'aåia', 'aågo', 'aål', 'aån', 'aåmp', 'aås',
													'aöf', 'aöjr', 'aöm', 'aön', 'aöoia', 'aöpenhamn', 'aöo', 'aöv', 'aöä', 'hing', 'hingfr', 'lihlm', 'lihn', 'prah', 'vtk', 'wib', 'pif', 'puu', 'mtr', 'luutn', 'ahof',
													'aafa', 'aafi', 'aßi', 'cymecxßyexb', 'deß', 'deßz', 'deßzutom', 'dfi', 'dfl', 'dfs', 'dft', 'dgr', 'dha', 'dhe', 'dhm', 'dif', 'dij', 'dtr', 'dto', 'dtm', 'ahorc',
													'dstta', 'dubb', 'dti', 'dth', 'dtt', 'dtn', 'dtd', 'dtii', 'dti', 'dul', 'dur', 'duu', 'duv', 'dvk', 'dux', 'dvi', 'dwi', 'dyg', 'dyl', 'dyy', 'döfn', 'döf', 'ecn',
													'dtä', 'dtö', 'dub', 'duc', 'dud', 'due', 'duf', 'dug', 'dugl', 'duh', 'dufv', 'eaa', 'eag', 'ead', 'eal', 'ean', 'eam', 'eas', 'eau', 'ebb', 'eci', 'ecl', 'eck', 'ect',
													'edd', 'edg', 'edh', 'edlt', 'edlv', 'edm', 'edsv', 'eeh', 'eei', 'eeii', 'eek', 'eel', 'eem', 'eer', 'ees', 'eet', 'eeu', 'eey', 'efa', 'efe', 'eff', 'efi', 'aaies',
													'eft', 'efu', 'efä', 'efö', 'ega', 'ege', 'egi', 'egl', 'egn', 'egr', 'ehd', 'ehe', 'ehkk', 'ehl', 'ehn', 'ehr', 'eht', 'ehu', 'ehy', 'eia', 'eie', 'eif', 'eig', 'babi',
													'eik', 'eil', 'eim', 'einj', 'eip', 'eir', 'eiy', 'eje', 'ejo', 'ejs', 'ejt', 'eju', 'eka', 'ekr', 'ekt', 'ekv', 'eky', 'elc', 'elel', 'eler', 'elfs', 'elj', 'elm',
													'elt', 'elv', 'ely', 'elö', 'emb', 'eme', 'emd', 'emt', 'emu', 'strn', 'sts', 'stsi', 'stst', 'sttj', 'sttm', 'stu', 'nli', 'smf', 'slßo', 'sna', 'snl', 'sni', 'snk',
													'isswi', 'issw', 'ivi', 'vksl', 'osv', 'ley', 'lfa', 'lfi', 'lfl', 'lfs', 'lft', 'lgs', 'lha', 'lho', 'lhs', 'sns', 'snt', 'snu', 'sntl', 'sof', 'sot', 'amm', 'astt',
													'sotn', 'sowu', 'spee', 'spp', 'spr', 'sps', 'spt', 'sra', 'srbl', 'sri', 'srk', 'sro', 'srs', 'srä', 'ssa', 'ssb', 'sse', 'ssi', 'ssia', 'ssk', 'ssl', 'ssn', 'sso',
													'sson', 'ssr', 'sst', 'ssta', 'sstl', 'ssu', 'ssw', 'stb', 'stc', 'stbk', 'stf', 'stg', 'sti', 'sth', 'stj', 'stl', 'stn', 'ablgt', 'abs', 'ack', 'acc', 'aca', 'avdxhef',
													'stta', 'stte', 'stti', 'stto', 'sttu', 'stty', 'sttä', 'sttö', 'suc', 'sud', 'sug', 'aamn', 'aar', 'abb', 'abc', 'abd', 'aba', 'abn', 'abi', 'abl', 'ablg', 'abt', 'babs',
													'abu', 'ach', 'act', 'adj', 'afd', 'aff', 'afe', 'afi', 'afl', 'afm', 'agd', 'afv', 'aftr', 'aft', 'agi', 'agr', 'agn', 'agt', 'ahd', 'aha', 'aht', 'aif', 'aii', 'astr',
													'aihi', 'aih', 'aiia', 'aij', 'aik', 'ais', 'aiu', 'aix', 'ajä', 'aja', 'aje', 'aji', 'anj', 'anm', 'anr', 'anp', 'öfv', 'öfr', 'åsg', 'ång', 'åkr', 'äär', 'äug', 'aml',
													'ajo', 'akccp', 'akl', 'akn', 'ako', 'alb', 'ald', 'alg', 'alh', 'allg', 'alm', 'alo', 'alw', 'alv', 'ame', 'amn', 'amä', 'amy', 'ääu', 'åby', 'åbr', 'ärd', 'amk', 'avicl',
													'ängf', 'zsa', 'zta', 'ztg', 'yty', 'yva', 'yvp', 'yys', 'yyt', 'ysy', 'yrn', 'yry', 'srr', 'srm', 'yrj', 'ypy', 'yni', 'ynt', 'ymy', 'ylt', 'ylh', 'yky', 'yii', 'avibl',
													'yjm', 'xbo', 'xii', 'xio', 'xioo', 'xiv', 'xll', 'xlo', 'xoo', 'xsi', 'xso', 'xvi', 'xää', 'wuu', 'wuotcc', 'wuot', 'wuotc', 'wro', 'wri', 'wsa', 'wsta', 'wui', 'baabeli',
													'wpk', 'wol', 'wll', 'wlo', 'wmo', 'wmm', 'wno', 'wio', 'wim', 'wij', 'wiin', 'wii', 'wih', 'wid', 'aaw', 'abk', 'abr', 'adl', 'adv', 'aee', 'ael', 'aeg', 'aea', 'astrsm',
													'ael', 'aen', 'aer', 'afa', 'aes', 'aet', 'affh', 'afk', 'afs', 'afo', 'afr', 'afsl', 'ahr', 'aiw', 'aju', 'aka', 'aldr', 'alft', 'allg', 'allm', 'alp', 'alr', 'astk',
													'aoa', 'aol', 'aom', 'aon', 'aoo', 'aor', 'apl', 'apj', 'apn', 'apnl', 'apr', 'arc', 'ard', 'ardr', 'arfd', 'arfw', 'arfv', 'arh', 'ark', 'arj', 'arkl', 'arl', 'asti',
													'arn', 'ars', 'arth', 'aru', 'arvl', 'arw', 'ary', 'asfa', 'asew', 'asp', 'assr', 'asst', 'asw', 'asz', 'atf', 'atg', 'atk', 'atl', 'ats', 'atn', 'atr', 'atv', 'assu',
													'auk', 'aui', 'aul', 'aum', 'aun', 'auo', 'aut', 'avd', 'avb', 'avl', 'avo', 'avs', 'awa', 'awi', 'awo', 'awu', 'awsw', 'ayt', 'ar', 'axi', 'ayd', 'bab', 'balf', 'assy',
													'banl', 'bankpl', 'bba', 'bbi', 'bbl', 'bbo', 'bbä', 'bdn', 'beb', 'bef', 'beh', 'behr', 'behm', 'beij', 'bej', 'bek', 'bekv', 'bel', 'bfi', 'bff', 'bfl', 'bft', 'assrm',
													'bia', 'bex', 'bhrci', 'bib', 'bibi', 'bii', 'iea', 'ёск', 'aabm', 'aabdnsbabbmbfibnhbi', 'aabacm', 'aabchmsm', 'aabme', 'aabbhl', 'aahe', 'aaide', 'aacgioj', 'asca', 'benzl',
													'aaer', 'aaes', 'aafarif', 'aaeio', 'adabl', 'aacnti', 'aaby', 'aabaiyyntl', 'aadut', 'aadla', 'aadtma', 'adt', 'aadtaajsti', 'aaeu', 'aeu', 'aeuu', 'aabdce', 'aabto', 'beo',
													'abg', 'aabg', 'aabcli', 'aabcl', 'aabclt', 'aabu', 'aach', 'aache', 'ache', 'achi', 'aachcn', 'aacce', 'aadbr', 'adbi', 'adb', 'aadbq', 'aabaßfeen', 'aaein', 'aaekcta',
													'aabc', 'lop', 'chr', 'lli', 'lhyww', 'ttll', 'lch', 'lcg', 'lcf', 'lcfv', 'nsö', 'chp', 'pll', 'jvk', 'aahpo', 'aacli', 'aafj', 'aafl', 'aaclönvut', 'ader', 'bedöml', 'beob',
													'aabol', 'aabotr', 'aabo', 'aabne', 'abj', 'aabj', 'aabra', 'aabram', 'tipn', 'cfr', 'pltt', 'vpl', 'pustl', 'cbsa', 'alj', 'aall', 'efo', 'aafo', 'aago', 'aagot', 'asv',
													'qal', 'qam', 'dta', 'itft', 'aabaari', 'aabbcc', 'aabdl', 'aabamiu', 'aabham', 'aabho', 'aabhe', 'aabh', 'aabec', 'aabeck', 'aabergi', 'aabesti', 'aababi', 'aabach', 'befa',
													'aabei', 'aabeii', 'aabempai', 'aabenras', 'aaben', 'aabeu', 'aabenra', 'aabelinp', 'aabel', 'aabeli', 'aabelintr', 'aabelint', 'aabelki', 'aabry', 'aabaajajfa', 'aabtl',
													'aaefitfi', 'aaeh', 'aaeho', 'aaencaßapt', 'aadl', 'aadlf', 'aadintr', 'aacjo', 'aacj', 'aacjjhisnnrv', 'aadw', 'aadwmon', 'aaed', 'aactci', 'aace', 'aabna', 'aabtrt', 'behms',
													'aacftrja', 'aacf', 'aacfo', 'aacfca', 'aacfuak', 'aacftrja', 'aaelgn', 'aael', 'aaeli', 'aaeltpo', 'aaeife', 'aacxflmh', 'efr', 'luu', 'ino', 'aabs', 'aaeo', 'aihp', 'baabe',
													'aabnd', 'aafe', 'aacv', 'aaci', 'aadaminp', 'aacne', 'aactraejcs', 'aann', 'aaddodaannddnnandd', 'aacmk', 'aeuo', 'aaboc', 'aabldtech', 'affi', 'aaffi', 'aahfi', 'aiimi', 'beok',
													'aadiiri', 'aabcpk', 'aabcu', 'aabwl', 'aaccid', 'aabtek', 'acjg', 'aahanno', 'ljd', 'tts', 'gna', 'agm', 'aagmt', 'agsi', 'aeto', 'aetr', 'aihp', 'aiht', 'aiifa', 'aiigi', 'bepo',
													'aiija', 'aiilqwi', 'aiimpi', 'alaj', 'allf', 'allr', 'alms', 'alml', 'almy', 'alpg', 'alul', 'alå', 'alé', 'alö', 'amis', 'ajn', 'ajll', 'ajm', 'ajj', 'ajk', 'ajg', 'ajh', 'bergn',
													'aacdmo', 'aacd', 'aacdsöri', 'aadgl', 'aadll', 'aem', 'aaem', 'aadu', 'aaduaa', 'aadvlkae', 'aadwmn', 'aadä', 'aaea', 'aaeac', 'aaee', 'aaeie', 'aaceßtkeßy', 'aiid', 'arji',
													'aacti', 'aaclo', 'aacno', 'aacnos', 'saacl', 'aacxb', 'aac', 'aacca', 'aaco', 'aacßon',  'aack', 'aacl', 'baacl', 'iaaclu', 'laaclu', 'aacbe',  'aacla', 'aacamfa', 'aaca', 'berh',
													'aahß', 'aaia', 'aaiani', 'aaie', 'aaiia', 'aaifi', 'aifi', 'aaifn', 'aaiel', 'aaiio', 'aaiitj', 'aaija', 'aaikslla', 'aaila', 'aaile', 'aaili', 'aaeb', 'jbmä', 'ndt', 'asch',
													'aair', 'aairp', 'aaiu', 'aaiuatw', 'aaiumwmm', 'aaiuu', 'aaiuub', 'aaiv', 'aaiyy', 'aaizo', 'aaiß', 'aajala', 'aajja', 'aajmf', 'aajne', 'amnsms', 'ampk', 'amr', 'amå', 'assw',
													'ablqu', 'abmnkd', 'abnn', 'aabdf', 'ajfr', 'ajs', 'ajr', 'ajy', 'ajt', 'ajv', 'ajw', 'ajx', 'ajß', 'akc', 'akg', 'akf', 'akk', 'akj', 'akkc', 'akr', 'aksl', 'amfa', 'amfi', 'aste',
													'abnu', 'aboa', 'aboae', 'aboas', 'aboba', 'abobo', 'abobsen', 'abodefgb', 'abodetgh', 'abol', 'abor', 'abot', 'abpe', 'aabp', 'abp', 'abq', 'abv', 'abrh', 'abrg', 'aabaasr', 'asr',
													'abta', 'absä', 'abwaf', 'abzu', 'abßbf', 'acb', 'acbe', 'accii', 'accilltl', 'acd', 'ace', 'acdi', 'acfh', 'acfa', 'acfh', 'acgna', 'acgu', 'achr', 'aci', 'acht', 'aabis', 'aas',
													'acjjaflbioßaro', 'ackm', 'acm', 'acn', 'aco', 'acr', 'acq', 'acu', 'acw', 'acy', 'acwo', 'acls', 'acrv', 'acssr', 'aacs', 'acs', 'actc', 'acts', 'actni', 'acto', 'acva', 'acxe',
													'adc', 'adce', 'adcirc', 'adclc', 'adcle', 'adclf', 'adclf', 'adde' 'addr', 'afj', 'afjr', 'afjm', 'afkw', 'afky', 'afllßbku', 'afll', 'afnn', 'aabyc', 'aabmm', 'aabalsnmne', 'bedrifn',
													'aflf', 'aflfa', 'aflfaf', 'aflfc', 'aflfeoßa', 'aflidn', 'aflj', 'aflimr', 'aflk', 'afllfo', 'afllijli', 'aflr', 'afltr', 'aflu', 'aflv', 'afn', 'afna', 'afnd', 'afne', 'aabde', 'bem',
													'afnftfl', 'afp', 'afowfa', 'afpo', 'afrt', 'afsfvm', 'afsg', 'afsk', 'afst', 'aftbl', 'aftc', 'aftfa', 'aftfld', 'aftfl', 'aftl', 'aftni', 'aftp', 'aftpnf', 'aftti', 'afttcacr', 'benk',
													'afttm', 'aftx', 'afty', 'afuf', 'afuntci', 'afuwa', 'afz', 'afä', 'agb', 'agbmr', 'agc', 'agcr', 'agct', 'agctll', 'agcutcr', 'agcn', 'agfcl', 'agft', 'agf', 'agj', 'agjl', 'arits',
													'agjls', 'aglksr', 'aglxvä', 'agrr', 'ags', 'agsb', 'agsjm', 'agsk', 'agsm', 'agtl', 'agtm', 'ahc', 'ahcs', 'aaim', 'aim', 'aaini', 'aain', 'aio', 'aamujplv', 'amg', 'amhe', 'amkv',
													'aafl', 'aage', 'aaftc', 'aafsr', 'aaent', 'aaen', 'aacmak', 'aacoßi', 'aade', 'aadho', 'aadltu', 'aadev', 'aaenklo', 'aaesßgo', 'aaberrci', 'aadarr', 'adar', 'aabox', 'aabllff', 'benn',
													'abca', 'bcd', 'abcd', 'abcde', 'abcdefgb', 'abcdefgh', 'aabcd', 'aabcdefgh', 'aabcdefghi', 'aabcdefghij', 'bce', 'bcf', 'bcem', 'bcero', 'bcgraf', 'bch', 'bcha', 'bci', 'bcij', 'bck',
													'bbc', 'bbe', 'bbf', 'bbime', 'bbj', 'bbla', 'bbw', 'bcbäck', 'bclsmki', 'bcm', 'bco', 'bcrg', 'bcrgh', 'bcrgi', 'bcta', 'bcti', 'bcu', 'bcy', 'bde', 'bded', 'bdes', 'bdh', 'bdio', 'bdiv',
													'abedefgb', 'abedefgh', 'abedetgb', 'abrn', 'abro', 'absa', 'aczi', 'acz', 'aabiergh', 'bborg', 'bbp', 'bbr', 'bbs', 'bbt', 'bbte', 'bbu', 'bdla', 'bdit', 'bdv', 'beba', 'bebo', 'bedj',
													'aabbau', 'aabba', 'aabbad', 'aabbe', 'aabbergetleatern', 'aabber', 'aabb', 'aabbs', 'aabbsk', 'aabbss', 'aabbsa', 'aabbsaa', 'aabbt', 'aabberg', 'aabbl', 'bbm', 'abbm', 'aabbmbi', 'aabbo', 
													'aabakkc', 'aabad', 'aaba', 'aabakkc', 'aaballu', 'aabasdigas', 'aabaimv', 'aabait', 'aabaja', 'aabauiu', 'aabasfoe', 'aabab', 'aababflaa', 'aaberg', 'aabete', 'aabnn', 'aabsl', 'bec',
													'aabalf', 'aabf', 'aabffl', 'aabfd', 'aabfo', 'aabfos', 'aabfsa', 'abbd', 'abbi', 'abbo', 'abbor', 'aabend', 'aabara', 'aabca', 'aabcabiibi', 'aabani', 'aabitfu', 'aaßaa', 'becf', 'becli',
													'aabafs', 'aabafli', 'aabaffccii', 'aabaffe', 'aabenta', 'aabakko', 'aabate', 'ablgs', 'abma', 'accls', 'achb', 'achccx', 'achcfjo', 'acrjljl', 'acrma', 'acte', 'acxn', 'acßyn', 'befehl',
													'aabenjaa', 'aaband', 'aabanv', 'aabattu', 'aabejbro', 'aabaek', 'aabala', 'aabebkramlare', 'aabenraag', 'aabenraagi', 'aabaljoki', 'aett', 'aetl', 'aaeto', 'aabojiobtb', 'arit', 'befi',
													'aabatjec', 'aabcn', 'aabcnra', 'aabcnras', 'aaballan', 'aabdatonta', 'aabcbm', 'aabci', 'aabekatt', 'aabeq', 'acä', 'adalrnina', 'aabmwwb', 'aatam', 'aatc', 'aatav', 'aate', 'baa', 'beffi',
													'aahu', 'ahu', 'ahv', 'aahvi', 'aahv', 'aahve', 'ahw', 'ahö', 'aaiami', 'aaho', 'aahi', 'aahopf', 'aahoagglg', 'aaiaiea', 'aiai', 'aaiamlac', 'aaiant', 'aaica', 'aic', 'aabdoo', 'aör',
													'aaisji', 'aaku', 'aalanotf', 'aalbcrse', 'aial', 'aiao', 'aiar', 'aias', 'aaielswa', 'aicetv', 'aict', 'aicm', 'aicu', 'aicmallc', 'aicßa', 'aabrcnn', 'aanaj', 'arabl', 'arai', 'aäo',
													'aäaau', 'aabaau', 'aaeaau', 'aau', 'aauhunln', 'aaujo', 'aaukk', 'aaul', 'aaun', 'aaua', 'aave', 'aawa', 'aawi', 'aawasa', 'aawo', 'aaws', 'aawu', 'aawwi', 'aclv', 'adcm', 'arg', 'aäm',
													'aral', 'arap', 'arbk', 'arbfte', 'arbo', 'arccn', 'arck', 'arci', 'arct', 'arcmi', 'arct', 'arcy', 'areg', 'arer', 'arfb', 'arff', 'areu', 'arfip', 'arfma', 'arfo', 'arha', 'ari', 'aza',
													'aaxo', 'axo', 'aaxifabni', 'aay', 'aaäri', 'aaé',  'abbar', 'abbé', 'abcnt', 'abdu', 'abef', 'abedsfg', 'abedsffb', 'abjj', 'abjo', 'abko', 'abka', 'abld', 'acme', 'aetv', 'araf', 'arie',
													'saadf', 'adf', 'aadfl', 'aadff', 'adfn', 'adi', 'adici', 'adjrmb', 'adjcr', 'adju', 'adlcrcroutz', 'adreß', 'adre', 'aemo', 'aeng', 'aaeng', 'aend', 'aenr', 'aeoo', 'aete', 'ara', 'ays',
													'aema', 'aaema', 'aeme', 'aemu', 'aewi', 'aeva', 'aeur', 'afaift', 'afar', 'afat', 'afas', 'afamaf', 'afbcrg', 'afiec', 'aftitn', 'agardh', 'agarc', 'agdn', 'agkr', 'agne', 'aqr', 'aqu',
													'affo', 'affsr', 'aaftro', 'affl', 'afh', 'afflr', 'affä', 'affä', 'affrd', 'afgd', 'afgejrd', 'afgfi', 'afgi', 'afgt', 'afgll', 'afgifw', 'afhä', 'afia', 'aficr', 'afii', 'apy', 'apä',
													'afkfs', 'aflfl', 'aflfljt', 'aflfmci', 'aflnva', 'aflp', 'afma', 'afrd', 'afrds', 'afss', 'afssemw', 'afta', 'afts', 'aftu', 'afut', 'afwe', 'agaf', 'agaplt', 'aanc', 'anc', 'aput', 'aysr',
													'ahe', 'aher', 'ahel', 'ahft', 'ahfa', 'ahfli', 'adahj', 'ahj', 'adahl', 'stahlhclm', 'ahl', 'ahlha', 'ahlho', 'ahli', 'ahlm', 'ahlqv', 'ahlqvi', 'ahlst', 'aifca', 'aifr', 'apum', 'apur',
													'ahm', 'aahmhmnßumrumhaammhmnn', 'aahmtkse', 'ahmff', 'ahme', 'aiig', 'aiis', 'aiif', 'aiik', 'aiie', 'aiit', 'aiirli', 'aiir', 'aiio', 'aiji', 'aijj', 'ailpg', 'ainc', 'aacxabixb', 'aabhf',
													'sbte', 'bbens', 'bfebj', 'rsbok', '"^[azäöü].+mäßig$"', 'aaboe', 'aabo', 'aaboam', 'aaboah', 'aabmzr', 'aabz', 'aacu', 'aabellap', 'aacie', 'ahtci', 'ahtb', 'aabni', 'aanbi', 'anb', 'anbt',
													'taißa', 'aads', 'aadsk', 'ads', 'adw', 'adwcni', 'aacxala', 'aabia', 'aabio', 'aabi', 'aabil', 'aabittaisitii', 'aabika', 'aabk', 'aababvaav', 'aabq', 'aabßaa', 'aaboth', 'aprl', 'aprll',
													'aabehbhahiihiibmbif', 'aabelja', 'aabe', 'aablkko', 'aabll', 'aabllsj', 'aablta', 'aabl', 'aabctj', 'aabgl', 'aabaflbi', 'aabmv', 'aabrr', 'aabsbhhw', 'aabesjtti', 'aabhava', 'aprilkl',
													'aabaaieililtt', 'aabama', 'aabava', 'adamjo', 'aabdiskh', 'aabolf', 'aabccb', 'aabadae', 'aabiu', 'aabalsnmne', 'aabö', 'aabakfe', 'aabaanlja', 'aaboßiiaaßda', 'aabrahamaabraham', 'aps',
													'aabrhatw', 'aaboxi', 'aabaife', 'aacatjji', 'aacaos', 'aabbd', 'aahc', 'aahcm', 'aachenirf', 'aabmsfc', 'ffood', 'cpizooti', 'aabborgi', 'aaln', 'aalo', 'aabta', 'aabtaßliait', 'apu',
													'aaberra', 'iaa', 'iaaf', 'iaafr', 'iaaf', 'abm', 'abmi', 'aabmi', 'aabmsp', 'aadalsg', 'aadr', 'aadru', 'ahtcc', 'aaeka', 'aaeeva', 'aacha', 'aamup', 'aand', 'aefo', 'aedo', 'appl', 'ayy',
													'aaioaj', 'aaionk', 'aaio', 'aaha', 'aano', 'aans', 'aant', 'aanv', 'aanvi', 'aanwilj', 'aape', 'aapp', 'aara', 'aarfe', 'aarr', 'aars', 'aarv', 'aasv', 'aatma', 'aatna', 'acis', 'apul',
													'aatl', 'aatnu', 'aato', 'aatlnt', 'aabaatl', 'aat', 'aatt', 'aatr', 'aaue', 'aauo', 'aaut', 'aav', 'aawe', 'aaö', 'aax', 'aaßoabi', 'abov', 'aborr', 'abso', 'accep', 'accp', 'appi', 'azaa',
													'acwi', 'acuo', 'adh', 'adg', 'adk', 'adma', 'aadmmqn', 'aadmf', 'aadmk', 'adm', 'adn', 'adnn', 'adnexltl', 'aadnmukko', 'adni', 'advb', 'advo', 'ady', 'aeglr', 'aeia', 'aeji', 'apoc', 'aze',
													'aejioß', 'aeiila', 'aejme', 'aela', 'aeli', 'aeni', 'aemb', 'aess', 'affa', 'afgaf', 'afgafa', 'afgafs', 'afgl', 'afle', 'afrr', 'afsla', 'afy', 'afö', 'aföc', 'anfr', 'anh', 'anlc', 'azi',
													'ahal', 'ahar', 'ahaa', 'ahcc', 'ahdi', 'ahde', 'ahdo', 'ahea', 'ahfe', 'ahhu', 'ahir', 'ahk', 'ahlb', 'ahle', 'ahlmi', 'ahlqu', 'ahwe', 'ahä', 'aang', 'andl', 'ands', 'andt', 'andst', 'azy',
													'annl', 'annll', 'annln', 'anodipatterlirlipterlerlerlir', 'antc', 'anth', 'antl', 'antw', 'anvc', 'anz', 'anwa', 'anwo', 'anyo', 'anä', 'anå', 'anö', 'aoao', 'apfo', 'apg', 'apk', 'apo', 
													'aäg', 'berl', 'berlln', 'berlip', 'bernd', 'bernh', 'bes', 'besl', 'besv', 'besw', 'betaln', 'betj', 'betl', 'betr', 'betts', 'betz', 'beuf', 'beus', 'beu', 'beun', 'bfa', 'bfc', 'bfck',
													'aoc', 'aoch', 'aod', 'aoe', 'aog', 'aogo', 'aoh', 'aohi', 'aoi', 'aoia', 'aoio', 'aoite', 'aoiu', 'aoj', 'aok', 'aoli', 'aolm', 'aop', 'aot', 'aou', 'aouo', 'aoy', 'apah', 'apai', 'apal',
													'aiima', 'arje', 'arkg', 'arkchäiw', 'arkk', 'arklv', 'arkp', 'arkt', 'arlb', 'arlcip', 'arldp', 'arlv', 'arll', 'arls', 'armcija', 'armf', 'armp', 'arms', 'arnd', 'arndts', 'aro', 'arp',
													'arpfte', 'arpe', 'arpi', 'arrac', 'arpp', 'arpl', 'arppc', 'arrmn', 'arrt', 'artj', 'artl', 'artt', 'artz', 'artturj', 'artturl', 'arwatcn', 'arwe', 'arwi', 'arwo', 'arß', 'asb', 'asbe',
													'asf', 'asfo', 'asg', 'ash', 'asi', 'asiap', 'asic', 'asie', 'asig', 'asik', 'asioij', 'asja', 'asl', 'aslo', 'asli', 'asm', 'asmn', 'asn', 'asnnt', 'aso', 'asnt', 'asps', 'assrr', 'assrrn',
													'atb', 'atc', 'atd', 'ate', 'atco', 'atelj', 'atfa', 'atfc', 'atfo', 'atfu', 'ath', 'atha', 'athl', 'atho', 'atjs', 'ato', 'atni', 'atp', 'atqve', 'atsk', 'atsl', 'atst', 'attl', 'aua', 'aub',
													'auc', 'aud', 'aucr', 'aue', 'auff', 'auffa', 'aufl', 'aufr', 'augn', 'auh', 'auhs', 'auin', 'auiu', 'auj', 'aujr', 'aulol', 'aull', 'aulnoy', 'aunr', 'aupp', 'auoa', 'auoi', 'auqu', 'aure',
													'aurd', 'aurell', 'aurlnk', 'ause', 'ausa', 'ausr', 'auss', 'aust', 'austr', 'ausv', 'auta', 'ausu', 'autf', 'autc', 'autn', 'autof', 'autohurj', 'autol', 'autr', 'auu', 'auue', 'avbr', 'avcin',
													'avf', 'avh', 'avk', 'avj', 'avn', 'avpa', 'avr', 'avsl', 'avto', 'avwo', 'awama', 'awance', 'awau', 'awe', 'awel', 'awll', 'awnn', 'awwl', 'aww', 'axcll', 'axe', 'axu', 'axh', 'ayn', 'ayo', 'ayr',
													'bahn', 'bahk', 'bahl', 'bahn', 'bahr', 'bahs', 'baht', 'baj', 'bakb', 'bakkcn', 'bakp', 'bal', 'balba', 'bald', 'balfe', 'baltls', 'baltlcum', 'baltlm', 'bamsj', 'bamk', 'bandl', 'banq', 'bantj',
													'bfe', 'bfhde', 'bga', 'bgi', 'bgl', 'bgx', 'bhb', 'bhba', 'bhea', 'bhhi', 'bhm', 'bhnberg', 'bho', 'bhr', 'bhrström', 'bhti', 'bibb', 'bibe', 'bibl', 'bic', 'bie', 'bif', 'biffi', 'bifl', 'bigi',
													'bigl', 'bign', 'bigt', 'biib', 'bih', 'biha', 'biia', 'biig', 'biihi', 'biihho', 'biij', 'biijett', 'biim', 'biis', 'biir', 'bijl', 'bijl', 'bik', 'biki', 'bilb', 'bilj', 'bim', 'bimi', 'biml',
													'binn', 'bip', 'biq', 'birn', 'biu', 'biv', 'biuv', 'bix', 'bja', 'bjg', 'bji', 'bjilp', 'bjj', 'bjjf', 'bjm', 'bjo', 'bjp', 'bjr', 'bjt', 'bju', 'bjup', 'bjå', 'bjö', 'bka', 'bkamp', 'bki', 'bkk',
													'bkm', 'bkqvis', 'bkr', 'bkt', 'bkti', 'bkv', 'bkx', 'bla', 'bladh', 'bladj', 'blai', 'blal', 'blb', 'blc', 'blces', 'blej', 'blf', 'blfl', 'blgt', 'blifw', 'blifwa', 'blii', 'bliiv', 'blifg',
													'blk', 'bll', 'bllbao', 'blle', 'bllj', 'blljaardl', 'bln', 'blne', 'blo', 'blol', 'blomq', 'blomqmi', 'blomqu', 'blr', 'blra', 'bls', 'blt', 'blta', 'bltr', 'blu', 'bluc', 'blvd', 'bly', 'bma', 
													'bme', 'bnksg', 'bnl', 'bnlo', 'bnmi', 'bnn', 'bno', 'bnr', 'bnra', 'bnse', 'bnsi', 'boa', 'bobr', 'boc', 'bocf', 'boct', 'bodn', 'bodg', 'boee', 'bofl', 'boh', 'bohl', 'bohn', 'bohm', 'boij',
													'bmi', 'bmk', 'bmm', 'bmmi', 'bmn', 'bmni', 'bmnnsväg', 'bmno', 'bmo', 'bmt', 'bmu', 'bmv', 'bmy', 'bmß', 'bna', 'bnand', 'bnd', 'bndtr', 'bne', 'bnei', 'bnenk', 'bnge', 'bngtig', 'bni', 'bnii',
													'boi', 'bokf', 'bokh', 'bolh', 'bolj', 'boncxßa', 'boo', 'booa', 'bonz', 'bop', 'bordl', 'bouqu', 'bov', 'bpg', 'bpa', 'bpemh', 'boßtad', 'bpl', 'bpo', 'bpr', 'bpra', 'bpw', 'bpå', 'bqg', 'bqo',
													'brah', 'bran', 'brb', 'brcslau', 'brcssel', 'brd', 'bre', 'breij', 'brefw', 'brehm', 'brej', 'brejl', 'brg', 'brii', 'brj', 'brja', 'brl', 'brlk', 'brlnma', 'brn', 'brng', 'brnk', 'brp', 'brr',
													'brsd', 'brsi', 'brt', 'brst', 'brtft', 'bru', 'brv', 'bryd', 'bsa', 'bso', 'bsb', 'bsd', 'bsg', 'bsh', 'bsi', 'bsj', 'bso', 'bsr', 'bsrn', 'bss', 'bsso', 'bst', 'bsä', 'bsö', 'bta', 'btb', 'btbk',
													'bte', 'btg', 'bth', 'bti', 'btik', 'btjrfan', 'btl', 'btm', 'btn', 'bto', 'btos', 'btr', 'btri', 'btrlini', 'bts', 'btst', 'btt', 'btta', 'btti', 'btto', 'bttä', 'btu', 'bty', 'btä', 'bua', 'bub',
													'buc', 'budg', 'budb', 'budj', 'bue', 'buf', 'buff', 'buffl', 'buh', 'buha', 'bugt', 'buhe', 'buhr', 'buhl', 'buia', 'buk', 'bul', 'bulba', 'bulg', 'bulew', 'buli', 'bulw', 'bum', 'buol', 'buom',
													'burr', 'butelj', 'butl', 'buu', 'buul', 'buv', 'buvbja', 'buä', 'bva', 'bvgd', 'bvi', 'bvo', 'bvr', 'bvtt', 'bvvi', 'bvvä', 'bvä', 'bwana', 'bxb', 'bxc', 'bxio', 'bxso', 'byv', 'byt', 'byx', 'byy',
													'bähr', 'bäl', 'bäg', 'bäm', 'bän', 'böf', 'böb', 'böc', 'böd', 'bög', 'bögh', 'böh', 'böhl', 'böhlc', 'böhm', 'böj', 'böl', 'böo', 'böp', 'caa', 'caai', 'caari', 'cad', 'cabl',
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
	try:
		print(f'\nstanza raw input:\n{docs}\n')
		# print(f"{f'nW: { len( docs.split() ) }':<10}{str(docs.split()[:7]):<150}", end="")
		st_t = time.time()
		all_ = smp(docs)
		# list comprehension: slow but functional alternative
		# print(f"{f'{ len(all_.sentences) } sent.: { [ len(vsnt.words) for _, vsnt in enumerate(all_.sentences) ] } words':<40}", end="")
		lemmas_list = [ re.sub(r'#|_|\-','', wlm.lower()) for _, vsnt in enumerate(all_.sentences) for _, vw in enumerate(vsnt.words) if ( (wlm:=vw.lemma) and len(wlm)>=3 and len(wlm)<=40 and not re.search(r"\b(?:\w*(\w)(\1{2,})\w*)\b|<eos>|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\s+", wlm) and vw.upos not in useless_upos_tags and wlm not in UNQ_STW ) ]
		end_t = time.time()
	except Exception as e:
		print(f"<!> Stanza Error: {e}")
		return
	print( lemmas_list )
	print(f"{len(lemmas_list)} lemma(s) | Elapsed_t: {end_t-st_t:.3f} s".center(150, "-") )
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