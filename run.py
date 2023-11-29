import subprocess
import json
import re

def run_my_script():
	command = ['python', 'myscript.py', '--query', 'I want an ice-cream right now!']
	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	return_code = process.wait()
	stdout, stderr = process.communicate()

	# Extract and deserialize the result
	serialized_result = re.search(r'Serialized Result: (.+)', stdout).group(1)
	my_res = json.loads(serialized_result)
	print('Captured Result:', type(my_res), my_res)

run_my_script()