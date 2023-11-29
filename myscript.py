import argparse
import json

parser = argparse.ArgumentParser(description='A sample script with argparse')
parser.add_argument('--query', type=str, help='Query Phrase', default="I love coding!")
args = parser.parse_args()

def customized_fcn(inp="This is my sample text!"):
	result=inp.split()
	return result

def main():
	my_res=customized_fcn(inp=args.query)

	# Serialize the list into a string and print it
	serialized_result = json.dumps(my_res)
	print('Serialized Result:', serialized_result)

if __name__ == '__main__':
	main()