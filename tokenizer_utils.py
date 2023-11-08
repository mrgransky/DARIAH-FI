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
													'gainerontzean', 'guztiz', 'hainbestez', 'horra', 'onların', 'ordea', 'hel', 'aac', 'ake', 'oeb',
													'osterantzean', 'sha', 'δ', 'δι', 'агар-чи', 'аз-баски', 'афташ', 'бале', 'gen', 'fjh', 'fji',
													'баҳри', 'болои', 'валекин', 'вақте', 'вуҷуди', 'гар', 'гарчанде', 'даме', 'карда', 
													'кошки', 'куя', 'кӣ', 'магар', 'майлаш', 'модоме', 'нияти', 'онан', 'оре', 'рӯи', 
													'сар', 'тразе', 'хом', 'хуб', 'чаро', 'чи', 'чунон', 'ш', 'шарте', 'қадар',
													'ҳай-ҳай', 'ҳамин', 'ҳатто', 'ҳо', 'ҳой-ҳой', 'ҳол', 'ҳолате', 'ӯим', 'באיזו', 'בו', 'במקום', 
													'בשעה', 'הסיבה', 'לאיזו', 'למקום', 'מאיזו', 'מידה', 'מקום', 'סיבה', 'ש', 'שבגללה', 'שבו', 'תכלית', 'أفعل', 
													'أفعله', 'انفك', 'برح', 'سيما', 'कम', 'से', 'ἀλλ',
													'rhr', 'trl', 'amt', 'nen', 'mnl', 'krj', 'laj', 'nrn', 'aakv', 'aal', 'oii', 'rji', 'ynn', 'eene', 'eeni', 'eeno', 'eenj', 'eenip', 'eenk',
													'hyv', 'kpl', 'ppt', 'ebr', 'emll', 'cdcdb', 'ecl', 'jjl', 'cdc', 'cdb', 'aga', 'ldr', 'lvg', 'zjo', 'cllcltl', 'adnexltl', 'adolj',
													'åbq', 'ordf', 'cuw', 'nrh', 'mmtw', 'crt', 'csök', 'hcrr', 'nvri', 'disp', 'ocll', 'rss', 'aalr', 'aama', 'aamma', 'aamme', 'aammamm', 'aamne', 'aamnie',
													'dii', 'hii', 'hij', 'stt', 'gsj', 'ådg', 'phj', 'nnw', 'wnw', 'nne', 'sij', 'apt', 'iit', 'juu', 'lut', 'aammä', 'aamnli', 'aamo', 'aamtd', 'aamucri',
													'afg', 'ank', 'fnb', 'ffc', 'oju', 'kpk', 'kpkm', 'kpkk', 'krs', 'prs', 'osk', 'rka', 'rnu', 'wsw', 'kic', 'aamwa', 'aander', 'aank', 'aanml', 'aanlcapt',
													'atm', 'vmrc', 'swc', 'fjijcn', 'hvmv', 'arb', 'inr', 'cch', 'msk', 'msn', 'tlc', 'vjj', 'jgk', 'mlk', 'aao', 'aaq', 'aaponp', 'aarincn', 'aarinci', 'aarm',
													'ffllil', 'adr', 'bea', 'ret', 'inh', 'vrk', 'ang', 'hra', 'nit', 'arr', 'jai', 'enk', 'bjb', 'iin', 'llä', 'aarn', 'aarnc', 'aato',
													'kka', 'tta', 'avu', 'nbl', 'drg', 'sek', 'wrw', 'tjk', 'jssjl', 'ing', 'scn', 'joh', 'yhd', 'uskc', 'enda', 'aabenfa', 'aabaiajajaeftsß',
													'tyk', 'wbl', 'vvichmann', 'okt', 'yht', 'akt', 'nti', 'mgl', 'vna', 'anl', 'lst', 'hku', 'wllss','ebf', 'aadmmqn', 'aadmf', 'aadmk', 'adm', 'adlj', 'adnn',
													'jan', 'febr', 'aug', 'sept', 'lok', 'nov', 'dec', 
													'jfe', 'ifc', 'flfl', 'bml', 'zibi', 'ibb', 'bbß', 'sfr', 'yjet', 'stm', 'imk', 'sotmt', 'oslfesfl', 'anoth', 'vmo', 'uts', 'abf', 'aajo', 'aajtus', 'aaka',
													'mmik', 'pytmkn', 'ins', 'ifk', 'vii', 'ikl', 'lan', 'trt', 'lpu', 'ijj', 'ave', 'lols', 'nyl', 'tav', 'ohoj', 'kph', 'answ', 'ansv', 'anw', 'anv', 'abjfi',
													'ver', 'jit', 'sed', 'sih','ltd', 'aan', 'exc', 'tle', 'xjl', 'iti', 'tto', 'otp', 'xxi', 'ing', 'fti', 'fjg', 'fjfj', 'ann', 'ansk', 'ant', 'abh', 'abjne',
													'mag', 'jnjejn', 'rvff', 'ast', 'lbo', 'otk', 'mcf', 'prc', 'vak', 'tif', 'nso', 'jyv','jjli', 'ent', 'edv', 'fjiwsi', 'alns', 'andr', 'anf', 'abja', 'ablfa',
													'psr', 'jay', 'ifr', 'jbl', 'iffi', 'ljjlii', 'jirl', 'nen', 'rwa', 'tla', 'mnn', 'jfa', 'omp', 'lnna', 'nyk', 'fjjg', 'taflhßtiß', 'ablbt', 'aajjp', 'afcn',
													'via', 'hjo', 'termffl', 'kvrk', 'std', 'utg', 'sir', 'icc', 'kif', 'ooh', 'imi', 'nfl', 'xxiv', 'tik', 'edv', 'fjj', 'tadiißi', 'taershß', 'taes', 'ablgn',
													'pnä', 'pna', 'urh', 'ltws', 'ltw', 'ltx', 'ltvfc', 'klp', 'fml', 'fmk', 'hkr', 'cyl', 'hrr', 'mtail', 'hfflt', 'xix', 'aaltofj', 'aaltl', 'aaltos', 'ablq',
													'rli', 'mjn', 'flr', 'ffnvbk', 'bjbhj', 'ikf', 'bbh', 'bjete', 'hmx', 'snww', 'dpk', 'hkk', 'aaf', 'aag', 'aagl', 'fjkj', 'aaloßil', 'aals', 'aaltc', 'afcrtu',
													'aah', 'aai', 'aaib', 'axa', 'axb', 'axd', 'axl', 'axx', 'ayf', 'bxi', 'bxa', 'bxs', 'bya', 'byb', 'bäh', 'öna', 'aalalshifo', 'aalop', 'afchlefca', 'afct',
													'byatrßm', 'bygk', 'byhr', 'byi', 'byo', 'byr', 'bys', 'bzj', 'bzo', 'bzt', 'bzw', 'bßt', 'bßß', 'bäb', 'bäc', 'bäd', 'ömv', 'aaks', 'aks', 'aalt', 'aaltoc',
													'bäi', 'bäik', 'bådh', 'bѣрa', 'caflm', 'caflddg','campm', 'camplnchl', 'camßi', 'canfh', 'jne', 'ium', 'xxviä', 'xys', 'ömä', 'aakrf', 'aalip', 'aalisv',
													'xxbx', 'xvy', 'xwr', 'xxii', 'xxix', 'xxo', 'xxm', 'xxl', 'xxv', 'xxä', 'xxvii', 'xyc', 'xxp', 'xxih', 'xxlui', 'xzå', 'önrcr', 'aakr', 'aaliohclmi', 'aalfie',
													'ybd', 'ybfi', 'ybh', 'ybj', 'yca', 'yck', 'ycs', 'yey', 'yfcr', 'yfc', 'yfe', 'yffk', 'yff', 'yfht', 'yfj', 'yfl', 'yft', 'öos', 'aaklx', 'aaliltx', 'aalfco',
													'ynnaxy', 'aoaxy', 'aob', 'cxy', 'jmw', 'jmxy', 'msxy', 'msx', 'msy', 'msßas', 'mszsk', 'mta', 'ius', 'tsb', 'öoo', 'öol', 'aakiint', 'aakk', 'aalci', 'aalc',
													'ile', 'lle', 'htdle', 'sir','pai', 'päi', 'str', 'ent', 'ciuj', 'homoj', 'quot', 'pro', 'tft', 'lso', 'sii', 'öot', 'öovct', 'öou', 'aakfxmto', 'aalb', 'aale',
													'hpl', 'tpv', 'vrpl', 'jtihiap', 'nii', 'nnf', 'aää', 'jofc', 'lxx', 'vmll', 'sll', 'vlli', 'pernstclln', 'nttä', 'npunl', 'aln', 'aakf', 'aakfco', 'aalj', 
													'aalflm', 'aaltpi', 'aaltqa', 'aalß', 'aalv', 'aalvt', 'aalw', 'taalß', 'aam', 'laam', 'taam', 'taamctli', 'taan', 'taawatwßki', 'taaßffrr', 'taaßxmm', 'afctitf',
													'öjj', 'öjo', 'öio', 'öiibi', 'öij', 'öbb', 'öba', 'åvq', 'åvp', 'åvl', 'åyr', 'åjj', 'åji', 'åjk', 'åff', 'åfgr', 'äåg', 'fjlmi', 'aak', 'aakcl', 'aalg',
													'ink', 'ksi', 'ctr', 'dec', 'fmf', 'ull', 'prof', 'sco', 'jjö', 'tcnko', 'itx', 'tcnkck', 'kello', 'jnho', 'infji', 'jib', 'ämj', 'aajr', 'aajwu', 'aald',
													'afu', 'ieo', 'ebep', 'tnvr', 'nta', 'tlyllal', 'viv', 'sån', 'hitl', 'vrt', 'pohj', 'nky', 'ope', 'ftm', 'tfflutti', 'aajaj', 'aajl', 'aalf',
													'lwiki', 'uhi', 'ffiuiru', 'eji', 'iil', 'äbt', 'llimi', 'efl', 'idbt', 'plchäialm', 'xukkalanka', 'aacxb', 'aadjf', 'ime', 'tps', 'aaißia', 'aaißial', 'aöd',
													'vps', 'tys', 'lto', 'pnnfi', 'nfiaiu', 'ilnm', 'cfe', 'hnmßr', 'pfäfflin', 'svk', 'alfr', 'pka', 'avg', 'ångf', 'arf', 'juh', 'pnjßw', 'aais', 'aaiw', 'aöc',
													'ruu', 'sus', 'rur', 'hden', 'kel', 'ppbcsfä', 'pptepbesfä', 'lof', 'adr', 'siv', 'owa', 'osa', 'aagsnyhttcr', 'aahr','aaif', 'aaig', 'aaii', 'aaik', 'talak',
													'aabd', 'aabx', 'aaco', 'aacßon', 'aad', 'aae', 'aafwvwvn', 'aagsßas', 'aagxt', 'aaiißofl', 'aaiiu', 'aaijsi', 'taißa', 'aads', 'aadsk', 'ads', 'adw', 'adwcni', 
													'adsh', 'adslf', 'advtn', 'aed', 'aeap', 'aeb', 'aeen', 'aef', 'aeev', 'aehi', 'aej', 'aeja', 'aek', 'afc', 'axaafcsr', 'aafc', 'aafcfcfc', 'aafci', 'afci',
													'tafllwalta', 'taflßan', 'tafmr', 'taftofße', 'taftpo', 'taftßßl', 'tafv', 'tahßßi', 'taij', 'tailctustlvhß', 'tailcvi', 'tairkkß', 'taiwalkoßk', 'tahmß',
													'taj', 'tajtsßßsa', 'takaisinßva', 'takhohßacu', 'takll', 'takm', 'taknn', 'taknmnmnn', 'takpi', 'takußtu', 'takxixl', 'takxjuol', 'takxmcl', 'takxul', 'wähä',
													'talamodh', 'talap', 'talapfi', 'talarc', 'talarn', 'talarpla', 'talaslcrh', 'talausjärjcskelniä', 'talblti', 'mäßig', 'aabn', 'aabnrxii', 'aabr', 'aabt', 'wähy',
													'tivßti', 'tiyäoliiacß', 'tiß', 'tlelsrtllß', 'tlerpoaaßoack', 'tlerposaßoack', 'tliatalm', 'yliopizto', 'yliopizco', 'ylioppilazlalo', 'yliopr', 'wuotr', 'wägncr',
													'wuotiscn', 'wußtusta', 'wva', 'wuwa', 'wuwm', 'wuwwnctv', 'wveb', 'wver', 'wveu', 'wvi', 'wvix', 'wvji', 'wvl', 'wvn', 'wvo', 'wvre', 'wvrqi', 'wvs', 'wvsfn', 'qan'
													'wvstnu', 'wvtba', 'wvu', 'wvv', 'wvviv', 'wvvo', 'wvvw', 'wvw', 'wvwv', 'wvxww', 'wvy', 'wvä', 'wwa', 'wvå', 'wwah', 'wwaja', 'wwb', 'wwcamri', 'wwcchmimh', 'wwy',
													'wwckku', 'wwcpbh', 'wwe', 'wwer', 'wwffl', 'wwg', 'wwfi', 'wwi', 'wwimi', 'wwj', 'wwk', 'wwka', 'wwl', 'wwn', 'wwnnm', 'wwo', 'wwol', 'wwpw', 'wwra', 'wwö', 'wwä',
													'wxn', 'wxm', 'wxos', 'wyi', 'wxs', 'wxu', 'wyj', 'wyk', 'wyl', 'wyn', 'wys', 'wzmvm', 'wzsta', 'wßjf', 'wßw', 'wäa', 'wäe', 'wäezg', 'wäg', 'wäh', 'wähcmi', 'wähär',
													'wäi', 'wähöh', 'wäisct', 'wäk', 'aack', 'acl', 'aacl', 'baacl', 'iaaclu', 'iaa', 'iaaf', 'iaafr', 'iaaf', 'laaclu', 'laad', 'qaa', 'qad', 'qahn', 'qaj', 'adu', 'afef',
													'qamd', 'qamdd', 'saacl', 'zaa', 'zaan', 'zab', 'zad', 'zag', 'zai', 'zahn', 'aikx', 'ail', 'ailcimmc', 'ailcs', 'ailclte', 'ailf', 'ailifci', 'aäh', 'aår', 'ahn', 'aho',
													'aßtx', 'aßa', 'aßctl', 'aßfn', 'aßj', 'aßl', 'aßmt', 'aßo', 'aßs', 'aßß', 'aßä', 'aäa', 'aäaa', 'aäak', 'aäaw', 'aäbi', 'aäci', 'aäd', 'aäel', 'aäe', 'aåss', 'ahnj',
													'aäi', 'aäj', 'aälj', 'aän', 'aär', 'aäs', 'aät', 'aäu', 'aäv', 'aäy', 'aäö', 'aåaana', 'aåad', 'aåan', 'aådt', 'aåe', 'aåg', 'aåia', 'aågo', 'aål', 'aån', 'aåmp', 'aås',
													'aöf', 'aöjr', 'aöm', 'aön', 'aöoia', 'aöpenhamn', 'aöo', 'aöv', 'aöä', 'hing', 'hingfr', 'lihlm', 'lihn', 'prah', 'vtk', 'wib', 'pif', 'puu', 'mtr', 'luutn', 'ahof',
													'aafa', 'aafi', 'aßi', 'cymecxßyexb', 'deß', 'deßz', 'deßzutom', 'dfi', 'dfl', 'dfs', 'dft', 'dgr', 'dha', 'dhe', 'dhm', 'dif', 'dij', 'dtr', 'dto', 'dtm', 'ahorc',
													'dstta', 'dubb', 'dti', 'dth', 'dtt', 'dtn', 'dtd', 'dtii', 'dti', 'dul', 'dur', 'duu', 'duv', 'dvk', 'dux', 'dvi', 'dwi', 'dyg', 'dyl', 'dyy', 'döfn', 'döf', 'ecn',
													'dtä', 'dtö', 'dub', 'duc', 'dud', 'due', 'duf', 'dug', 'dugl', 'duh', 'dufv', 'eaa', 'eag', 'ead', 'eal', 'ean', 'eam', 'eas', 'eau', 'ebb', 'eci', 'ecl', 'eck', 'ect',
													'edd', 'edg', 'edh', 'edlt', 'edlv', 'edm', 'edsv', 'eeh', 'eei', 'eeii', 'eek', 'eel', 'eem', 'eer', 'ees', 'eet', 'eeu', 'eey', 'efa', 'efe', 'eff', 'efi', 
													'eft', 'efu', 'efä', 'efö', 'ega', 'ege', 'egi', 'egl', 'egn', 'egr', 'ehd', 'ehe', 'ehkk', 'ehl', 'ehn', 'ehr', 'eht', 'ehu', 'ehy', 'eia', 'eie', 'eif', 'eig', 'eii',
													'eik', 'eil', 'eim', 'einj', 'eip', 'eir', 'eiy', 'eje', 'ejo', 'ejs', 'ejt', 'eju', 'eka', 'ekr', 'ekt', 'ekv', 'eky', 'elc', 'elel', 'eler', 'elfs', 'elj', 'elm', 'eln',
													'elt', 'elv', 'ely', 'elö', 'emb', 'eme', 'emd', 'emt', 'emu', 'strn', 'sts', 'stsi', 'stst', 'sttj', 'sttm', 'stu', 'nli', 'smf', 'slßo', 'sna', 'snl', 'sni', 'snk',
													'isswi', 'issw', 'ivi', 'vksl', 'osv', 'ley', 'lfa', 'lfi', 'lfl', 'lfs', 'lft', 'lgs', 'lha', 'lho', 'lhs', 'sns', 'snt', 'snu', 'sntl', 'sof', 'sot', 'amm',
													'sotn', 'sowu', 'spee', 'spp', 'spr', 'sps', 'spt', 'sra', 'srbl', 'sri', 'srk', 'sro', 'srs', 'srä', 'ssa', 'ssb', 'sse', 'ssi', 'ssia', 'ssk', 'ssl', 'ssn', 'sso',
													'sson', 'ssr', 'sst', 'ssta', 'sstl', 'ssu', 'ssw', 'stb', 'stc', 'stbk', 'stf', 'stg', 'sti', 'sth', 'stj', 'stl', 'stn', 'ablgt', 'abs', 'ack', 'acc', 'aca',
													'stta', 'stte', 'stti', 'stto', 'sttu', 'stty', 'sttä', 'sttö', 'suc', 'sud', 'sug', 'aamn', 'aar', 'abb', 'abc', 'abd', 'aba', 'abn', 'abi', 'abl', 'ablg', 'abt',
													'abu', 'ach', 'act', 'adj', 'afd', 'aff', 'afe', 'afi', 'afl', 'afm', 'agd', 'afv', 'aftr', 'aft', 'agi', 'agr', 'agn', 'agt', 'ahd', 'aha', 'aht', 'aif', 'aii',
													'aihi', 'aiia', 'aij', 'aik', 'ais', 'aiu', 'aix', 'ajä', 'aja', 'aje', 'aji', 'anj', 'anm', 'anr', 'anp', 'öfv', 'öfr', 'åsg', 'ång', 'åkr', 'äär', 'äug', 'aml',
													'ajo', 'akccp', 'akl', 'akn', 'ako', 'alb', 'ald', 'alg', 'alh', 'allg', 'alm', 'alo', 'alw', 'alv', 'ame', 'amn', 'amä', 'amy', 'ääu', 'åby', 'åbr', 'ärd', 'amk',
													'ängf', 'zsa', 'zta', 'ztg', 'yty', 'yva', 'yvp', 'yys', 'yyt', 'ysy', 'yrn', 'yry', 'srr', 'srm', 'yrj', 'ypy', 'yni', 'ynt', 'ymy', 'ylt', 'ylh', 'yky', 'yii',
													'yjm', 'xbo', 'xii', 'xio', 'xioo', 'xiv', 'xll', 'xlo', 'xoo', 'xsi', 'xso', 'xvi', 'xää', 'wuu', 'wuotcc', 'wuot', 'wuotc', 'wro', 'wri', 'wsa', 'wsta', 'wui',
													'wpk', 'wol', 'wll', 'wlo', 'wmo', 'wmm', 'wno', 'wio', 'wim', 'wij', 'wiin', 'wii', 'wih', 'wid', 'aaw', 'abk', 'abr', 'adl', 'adv', 'aee', 'ael', 'aeg', 'aea',
													'ael', 'aen', 'aer', 'afa', 'aes', 'aet', 'affh', 'afk', 'afs', 'afo', 'afr', 'afsl', 'ahr', 'aiw', 'aju', 'aka', 'aldr', 'alft', 'allg', 'allm', 'alp', 'alr',
													'aoa', 'aol', 'aom', 'aon', 'aoo', 'aor', 'apl', 'apj', 'apn', 'apnl', 'apr', 'arc', 'ard', 'ardr', 'arfd', 'arfw', 'arfv', 'arh', 'ark', 'arj', 'arkl', 'arl',
													'arn', 'ars', 'arth', 'aru', 'arvl', 'arw', 'ary', 'asfa', 'asew', 'asp', 'assr', 'asst', 'asw', 'asz', 'atf', 'atg', 'atk', 'atl', 'ats', 'atn', 'atr', 'atv',
													'auk', 'aui', 'aul', 'aum', 'aun', 'auo', 'aut', 'avd', 'avb', 'avl', 'avo', 'avs', 'awa', 'awi', 'awo', 'awu', 'awsw', 'ayt', 'ar', 'axi', 'ayd', 'bab', 'balf',
													'banl', 'bankpl', 'bba', 'bbi', 'bbl', 'bbo', 'bbä', 'bdn', 'beb', 'bef', 'beh', 'behr', 'behm', 'beij', 'bej', 'bek', 'bekv', 'bel', 'bfi', 'bff', 'bfl', 'bft',
													'bia', 'bex', 'bhrci', 'bib', 'bibi', 'bii', 'iea', 'ёск', 'aablta', 'aabl', 'aabi', 'aabil', 'aabittaisitii', 'aabj', 'aabm', 'aabesti', 'aadtma', 'aadojf',
													'aaer', 'aaes', 'aafarif', 'aaeio', 'adabl', 'aacnti', 'aaby', 'aabaiyyntl', 'aabia', 'aabfos', 'aabio', 'aadut', 'aadla', 'aadtaajsti', 'aaeu', 'aeu', 'aeuu',
													'abg', 'aabg', 'aabclt', 'aabcnra', 'aabu', 'aach', 'aache', 'ache', 'achi', 'aachcn', 'aacce', 'aadbr', 'adbi', 'adb', 'aadbq', 'aabfsa', 'aafj', 'aafl', 
													'aabcd', 'aabc', 'aabcli', 'aabcl', 'lop', 'chr', 'lli', 'lhyww', 'ttll', 'lch', 'lcg', 'lcf', 'lcfv', 'nsö', 'chp', 'pll', 'jvk', 'aacfca', 'aabra', 'aahpo', 'aacli',
													'aabol', 'aabotr', 'aabo', 'aabne', 'abj', 'aabj', 'aabram', 'tipn', 'cfr', 'pltt', 'vpl', 'pustl', 'cbsa', 'alj', 'aall', 'efo', 'aafo', 'aago', 'aagot',
													'qal', 'qam', 'dta', 'itft', 'aabaari', 'aabbcc', 'aabdl', 'aabamiu', 'aabham', 'aabho', 'aabeck', 'aabergi', 'aabelki', 'aababi', 'aabach', 'aaein', 'aaekcta',
													'aabei', 'aabeii', 'aabempai', 'aabenras', 'aaben', 'aabeu', 'aabenra', 'aabelinp', 'aabel', 'aabeli', 'aabe', 'aabelintr', 'aabelint', 'aabehbhahiihiibmbif', 'aabelja',
													'aaefitfi', 'aaeh', 'aaeho', 'aaencaßapt', 'aadl', 'aadlf', 'aadintr', 'aacjjhisnnrv', 'aadw', 'aadwmon', 'aaed', 'aactci', 'aacftrja', 'aace', 'aaceßtkeßy', 'ader',
													'aacftrja', 'aaelgn', 'aael', 'aaeli', 'aaeltpo', 'aaeife', 'aacxflmh', 'efr', 'luu', 'ino', 'aabs', 'aadolrl', 'aaeo', 'affi', 'aaffi', 'aahfi', 'aaha', 
													'aado', 'aabnd', 'aafe', 'aacv', 'aaci', 'aadaminp', 'aacfo', 'aacne', 'aactraejcs', 'aaddodaannddnnandd', 'aacmk', 'aacfuak', 'aaca', 'aacd', 'aacjo', 'aeuo',
													'aadolf', 'aadolfa', 'aadolffi', 'aadolfi', 'aadolfin', 'aadolfink', 'aadiiri', 'aabcpk', 'aabcu', 'aabwl', 'aaccid', 'aabtek', 'acjg', 'aahanno',
													'aacdmo', 'aadgl', 'aadll', 'aadnmukko', 'aem', 'aaem', 'aadu', 'aaduaa', 'aadvlkae', 'aadwmn', 'aadä', 'aaea', 'aaeac', 'aaee', 'aaeie',  'aahc', 'aahcm',
													'aaeka', 'aadol', 'aaeeva', 'aacti', 'aaclo', 'aacnos', 'aaca', 'abm', 'abmi', 'aabmi', 'aabmsp', 'aacla', 'aadalsg', 'aadof', 'aadr', 'aadru', 'ahtcc', 'ahtci', 'ahtb',
													'aahß', 'aaia', 'aaiani', 'aaie', 'aaiia', 'aaifi', 'aaifn', 'aaiel', 'aaiio', 'aaiitj', 'aaija', 'aaikslla', 'aaila', 'aaile', 'aaili', 'aaeb',
													'aair', 'aairp', 'aaiu', 'aaiuatw', 'aaiumwmm', 'aaiuu', 'aaiuub', 'aaiv', 'aaiyy', 'aaizo', 'aaiß', 'aajala', 'aajja', 'aajmf', 'aajne', 'ablqu', 'abmnkd', 'abnn',
													'abnu', 'aboa', 'aboae', 'aboas', 'aboba', 'abobo', 'abobsen', 'abodefgb', 'abodetgh', 'abol', 'abor', 'abot', 'abpe', 'aabp', 'abp', 'abq', 'abv', 'abrh', 'abrg',
													'abta', 'absä', 'abwaf', 'abzu', 'abßbf', 'acb', 'acbe', 'accii', 'accilltl', 'acd', 'ace', 'acdi', 'acfh', 'acfa', 'acfh', 'acgna', 'acgu', 'achr', 'aci', 'acht',
													'acjjaflbioßaro', 'ackm', 'acm', 'acn', 'aco', 'acr', 'acq', 'acu', 'acw', 'acy', 'acwo', 'acls', 'acrv', 'acssr', 'aacs', 'acs', 'actc', 'acts', 'actni', 'acto', 'acva', 'acxe',
													'adc', 'adce', 'adcirc', 'adclc', 'adcle', 'adclf', 'adclf', 'adde' 'addr', 'afj', 'afjr', 'afjm', 'afkw', 'afky', 'afllßbku', 'afll', 'afnn', 
													'aflf', 'aflfa', 'aflfaf', 'aflfc', 'aflfeoßa', 'aflidn', 'aflj', 'aflimr', 'aflk', 'afllfo', 'afllijli', 'aflr', 'afltr', 'aflu', 'aflv', 'afn', 'afna', 'afnd', 'afne',
													'afnftfl', 'afp', 'afowfa', 'afpo', 'afrt', 'afsfvm', 'afsg', 'afsk', 'afst', 'aftbl', 'aftc', 'aftfa', 'aftfld', 'aftfl', 'aftl', 'aftni', 'aftp', 'aftpnf', 'aftti', 'afttcacr',
													'afttm', 'aftx', 'afty', 'afuf', 'afuntci', 'afuwa', 'afz', 'afä', 'agb', 'agbmr', 'agc', 'agcr', 'agct', 'agctll', 'agcutcr', 'agcn', 'agfcl', 'agft', 'agf', 'agj', 'agjl',
													'agjls', 'aglksr', 'aglxvä', 'agrr', 'ags', 'agsb', 'agsjm', 'agsk', 'agsm', 'agtl', 'agtm', 'ahc', 'ahcs', 'aaim', 'aim', 'aaini', 'aain', 'aio','aaioaj', 'aaionk', 'aagmt', 'aabaife',
													'aafl', 'aage', 'aaftc', 'aafsr', 'aaent', 'aaen', 'aacmak', 'aacoßi', 'aade', 'aadho', 'aadltu', 'aadev', 'aaenklo', 'aaesßgo', 'aabh', 'aaberrci', 'aadarr', 'adar',
													'abca', 'abcd', 'abcde', 'abcdefgb', 'abcdefgh', 'aabcdefgh', 'aabcdefghi', 'aabcdefghij', 'abedefgb', 'abedefgh', 'abedetgb', 'abrn', 'abro', 'absa', 'aczi', 'acz',
													'aabbau', 'aabbe', 'aabbergetleatern', 'aabber', 'aabb', 'aabbsaa', 'aabbt',  'aabba', 'aaßaa', 'aabbsk', 'aabbss', 'aabama', 'aabava', 'adamjo', 'aaeto', 'aeto', 'aetr', 
													'aabakkc', 'aabad', 'aaba', 'aabakkc', 'aaballu', 'aabasdigas', 'aabaimv', 'aabaffe', 'aabait', 'aabaja', 'aabauiu', 'aabasfoe', 'aabab', 'aababflaa', 'aaberg', 'aabete', 
													'aabalf', 'aabf', 'aabffl', 'aabfd', 'aabafli', 'aabbd', 'abbd', 'abbi', 'abbm', 'abbo', 'abbor', 'aabend', 'aabara', 'aabbl', 'aabbmbi', 'aabca', 'aabcabiibi', 'aabani', 
													'aabafs', 'aabenta', 'aabakko', 'aabate', 'ablgs', 'abma', 'accls', 'achb', 'achccx', 'achcfjo', 'acrjljl', 'acrma', 'acte', 'acxn', 'acßyn', 'acä', 'adalrnina', 'aetl',
													'aabenjaa', 'aaband', 'aabanv', 'aabattu', 'aabbo', 'aabejbro', 'aabaek', 'aabala', 'aabberg', 'aabebkramlare', 'aabenraag', 'aabenraagi', 'aabaljoki', 'aabbad', 'aett',
													'aabatjec', 'aabcnras', 'aaballan', 'aabhe', 'aabdatonta', 'aatl', 'aatnu', 'aatlnt', 'aabaatl', 'aabcbm', 'aabci', 'aabekatt', 'aabeq', 'aabfo', 'aabika', 'aabaffccii', 
													'aahu', 'ahu', 'ahv', 'aahvi', 'aahv', 'aahve', 'ahw', 'ahö', 'aaiami', 'aaho', 'aahi', 'aahopf', 'aahoagglg', 'aaiaiea', 'aiai', 'aaiamlac', 'aaiant', 'aaica', 'aic',
													'aaisji', 'aaku', 'aalanotf', 'aalbcrse', 'aial', 'aiao', 'aiar', 'aias', 'aaielswa', 'aibli', 'aibtt', 'aicetv', 'aict', 'aicm', 'aicu', 'aicmallc', 'aicßa',
													'aäaau', 'aabaau', 'aaeaau', 'aau', 'aauhunln', 'aaujo', 'aaukk', 'aaul', 'aaun', 'aaua', 'aave', 'aawa', 'aawi', 'aawasa', 'aawo', 'aaws', 'aawu', 'aawwi', 'aclv', 'adcm',
													'aaxo', 'aaxifabni', 'aay', 'aaäri', 'aaé', 'abbado', 'abbar', 'abbé', 'abcnt', 'abdu', 'abef', 'abedsfg', 'abedsffb', 'abjj', 'abjo', 'abko', 'abka', 'abld', 'acme', 'aetv',
													'saadf', 'adf', 'aadfl', 'aadff', 'adfn', 'adi', 'adici', 'adjrmb', 'adjcr', 'adju', 'adlcrcroutz', 'adreß', 'adre', 'aemo', 'aeng', 'aaeng', 'aend', 'aenr', 'aeoo', 'aete',
													'aema', 'aaema', 'aeme', 'aemu', 'aewi', 'aeva', 'aeur', 'afaift', 'afar', 'afat', 'afas', 'afamaf', 'afbcrg', 'afiec', 'aftitn', 'agardh', 'agarc', 'agdn', 'agkr', 'agne',
													'affo', 'affsr', 'aaftro', 'affl', 'afh', 'afflr', 'affä', 'affä', 'affrd', 'afgd', 'afgejrd', 'afgfi', 'afgi', 'afgt', 'afgll', 'afgifw', 'afhä', 'afia', 'aficr', 'afii',
													'afkfs', 'aflfl', 'aflfljt', 'aflfmci', 'aflnva', 'aflp', 'afma', 'afrd', 'afrds', 'afss', 'afssemw', 'afta', 'afts', 'aftu', 'afut', 'afwe', 'aföc', 'agaf', 'agaplt',
													'ahe', 'aher', 'ahel', 'ahft', 'ahfa', 'ahfli', 'adahj', 'ahj', 'adahl', 'stahlhclm', 'ahl', 'ahlha', 'ahlho', 'ahli', 'ahlm', 'ahlqv', 'ahlqvi', 'ahlst', 'aifca', 'aifr',
													'ahm', 'aahmhmnßumrumhaammhmnn', 'aahmtkse', 'ahmff', 'ahme', 'aiig', 'aiif', 'aiik', 'aiie', 'aiit', 'aiirli', 'aiio', 'aiji', 'aijj', 'ailpg', 'ainc',
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