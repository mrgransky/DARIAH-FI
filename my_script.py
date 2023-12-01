# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--my_arg', default=None, type=int, required=True)
# parser.add_argument('--second_arg_name', default="Sample text", type=str)
# args = parser.parse_args()

# print(args.my_arg, type(args.my_arg))

# args.my_arg=None if args.my_arg==-1 else int(args.my_arg)

# if args.my_arg:
# 	print('my_arg argument is:', args.my_arg)
# else:
# 	print('my_arg argument was not specified')

# # Open the file in read mode
# with open('my_file.txt', 'r') as file:
# 	# Read lines from the file and remove trailing newline characters
# 	words = [line.strip() for line in file]

# # Now 'words' is a list containing the words from your file
# print(words)
# print(len(words))

def get_recsys_result(qu: str="Tampereen seudun työväenopisto"):
	print(f"Running {__file__} query: {qu}")
	cmd=[	'python', 'concat_dfs.py',
				'--dfsPath', '/scratch/project_2004072/Nationalbiblioteket/dataframes_x30',
				'--lmMethod', 'stanza', 
				'--qphrase', f'{qu}',
			]
	process=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	return_code=process.wait()
	stdout, stderr=process.communicate()
	serialized_result=re.search(r'Serialized Result: (.+)', stdout).group(1)
	recommended_tokens=json.loads(serialized_result)
	print('Captured Result:', type(recommended_tokens), len(recommended_tokens), recommended_tokens)
	# return [f"TK_{i+1}" for i in range(topK)]
	return recommended_tokens

if __name__ == '__main__':
	get_recsys_result()