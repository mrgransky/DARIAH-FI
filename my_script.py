import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--my_arg', default=None, type=int)
parser.add_argument('--second_arg_name', default="Sample text", type=str)
args = parser.parse_args()

print(args.my_arg, type(args.my_arg))

args.my_arg=None if args.my_arg==-1 else int(args.my_arg)

if args.my_arg:
	print('my_arg argument is:', args.my_arg)
else:
	print('my_arg argument was not specified')