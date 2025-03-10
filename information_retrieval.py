from url_scraping import *
from utils import *

# how to run in background:
# nohup python -u information_retrieval.py --queryLogFile /scratch/project_2004072/Nationalbiblioteket/NLF_Pseudonymized_Logs/nike5.docworks.lib.helsinki.fi_access_log.2021-01-01.log --dataset_dir /scratch/project_2004072/Nationalbiblioteket/NLF_DATASET_XXX > logNEW_q0.out &

# how to run Puhti:
# python information_retrieval.py --queryLogFile /scratch/project_2004072/Nationalbiblioteket/NLF_Pseudonymized_Logs/nike5.docworks.lib.helsinki.fi_access_log.2021-01-01.log --dataset_dir /scratch/project_2004072/Nationalbiblioteket/NLF_DATASET_XXX

def scrape_log_(
		fpath: str,
		dataset_dir: str,
		ts: List[str]=None,
	):
	os.makedirs(dataset_dir, exist_ok=True)

	print(f"Input Query Log File: {fpath}")
	qeury_log_raw_fname = fpath[fpath.rfind("/")+1:] # nike6.docworks.lib.helsinki.fi_access_log.2021-10-13.log
	scraped_query_fname = os.path.join(dataset_dir, f'{qeury_log_raw_fname}.gz')

	if os.path.isfile(scraped_query_fname):
		print(f"{scraped_query_fname} already exist, exiting...")
		return

	st_t = time.time()
	df = get_df_pseudonymized_logs(infile=fpath, TIMESTAMP=ts)
	print(f"DF_init Loaded in: {time.time()-st_t:.3f} sec | {df.shape}".center(100, " "))

	if df.shape[0] == 0:
		print(f"<!> Empty DF_init: {df.shape}, Nothing to retrieve, Stop Executing...")
		return

	print(
		f"{f'Page Analysis'.center(150, ' ')}\n"
		f"search pages: {df.referer.str.count('/search').sum()}, "
		f"collection pages: {df.referer.str.count('/collections').sum()}, "
		f"serial publication pages: {df.referer.str.count('/serial-publications').sum()}, "
		f"paper-for-day pages: {df.referer.str.count('/papers-for-day').sum()}, "
		f"clippings pages: {df.referer.str.count('/clippings').sum()}, "
		f"newspaper content pages: {df.referer.str.count('term=').sum()}, "
		#f"unknown pages: {df.referer.str.count('/collections').sum()}."
	)
	print("*"*150)
	# return
	parsing_t = time.time()
	print(f">> Scraping Newspaper Content Pages...")
	st_nwp_content_t = time.time()
	# df["nwp_content_referer"] = df[df.referer.str.contains('term=')]["referer"]
	df["nwp_content_referer"] = df.referer.map(lambda x: x if re.search(r'\/binding\/(\d+)', x) else np.nan, na_action="ignore",)
	df["nwp_content_results"] = df["nwp_content_referer"].map(scrap_newspaper_content_page, na_action='ignore')
	print(f"{f'Total Elapsed_t [Newspaper Content Pages]: {time.time()-st_nwp_content_t:.2f} s'.center(120, ' ')}")

	print(f">> Scraping Query Search Pages...")
	st_search_t = time.time()
	df["search_referer"] = df[df.referer.str.contains('/search')]["referer"]
	df["search_query_phrase"] = df["search_referer"].map(get_query_phrase, na_action='ignore')
	df["search_results"] = df["search_referer"].map(scrap_search_page, na_action='ignore')
	print(f"{f'Total Elapsed_t [Query Search Pages]: {time.time()-st_search_t:.2f} s'.center(120, ' ')}")

	print(f">> Scraping Collection Pages...")
	st_collection_t = time.time()
	df["collection_referer"] = df[df.referer.str.contains('/collections')]["referer"]
	df["collection_query_phrase"] = df["collection_referer"].map(get_query_phrase, na_action='ignore')
	df["collection_results"] = df["collection_referer"].map(scrap_collection_page, na_action='ignore')
	print(f"{f'Total Elapsed_t [Collection Pages]: {time.time()-st_collection_t:.2f} s'.center(120, ' ')}")

	print(f">> Scraping Clipping Pages...")
	st_clipping_t = time.time()
	df["clipping_referer"] = df[df.referer.str.contains('/clippings')]["referer"]
	df["clipping_query_phrase"] = df["clipping_referer"].map(get_query_phrase, na_action='ignore')
	df["clipping_results"] = df["clipping_referer"].map(scrap_clipping_page, na_action='ignore')
	print(f"{f'Total Elapsed_t [Clipping Pages]: {time.time()-st_clipping_t:.2f} s'.center(120, ' ')}")

	print("#"*100)
	print(f"Total Parsing Elapsed_t: {time.time()-parsing_t:.2f} s | DF: {df.shape}")
	print("<>"*50)

	print(f"Memory usage of each column in bytes (total column(s)={len(list(df.columns))})")
	print(df.memory_usage(deep=True, index=False, ))
	print("-"*100)

	print(df.info(verbose=True, memory_usage="deep", show_counts=True, ))
	print(df.head(10))
	print("*"*100)
	print(df.tail(10))
	print("*"*100)
	save_pickle( pkl=df, fname=scraped_query_fname )

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description='National Library of Finland (NLF) Log Data Extraction')
	parser.add_argument('--queryLogFile', '-f', required=True, type=str, help="Query log file")
	parser.add_argument('--dataset_dir', '-ddir', required=True, type=str, help='Save DataFrame in directory')
	args, unknown = parser.parse_known_args()
	args = parser.parse_args()
	print_args_table(args=args, parser=parser)

	scrape_log_(
		fpath=args.queryLogFile,
		dataset_dir=args.dataset_dir,
		#ts=["14:30:00", "14:56:59"],
	)

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))