import subprocess
import json
import re

def run_my_script():
	command = ['python', 'myscript.py', '--query', 'I want to eat an ice-cream right now!']
	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	return_code = process.wait()
	stdout, stderr = process.communicate()

	# Extract and deserialize the result
	serialized_result = re.search(r'Serialized Result: (.+)', stdout).group(1)
	my_splited_text=json.loads(serialized_result)
	print('Captured Result:', type(my_splited_text), my_splited_text)

run_my_script()