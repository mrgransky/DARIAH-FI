import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--my_arg', default=None, type=int, required=True)
parser.add_argument('--second_arg_name', default="Sample text", type=str)
args = parser.parse_args()

print(args.my_arg, type(args.my_arg))

args.my_arg=None if args.my_arg==-1 else int(args.my_arg)

if args.my_arg:
	print('my_arg argument is:', args.my_arg)
else:
	print('my_arg argument was not specified')

# Open the file in read mode
with open('my_file.txt', 'r') as file:
	# Read lines from the file and remove trailing newline characters
	words = [line.strip() for line in file]

# Now 'words' is a list containing the words from your file
print(words)
print(len(words))