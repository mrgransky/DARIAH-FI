import os

os.system('clear')
usr = os.environ['HOME']
dpath = f"{usr}/Datasets/Nationalbiblioteket/no_ip_logs"
fname = "nike6.docworks.lib.helsinki.fi_access_log.2021-02-07.log"

infile = os.path.join(dpath, fname)

print(infile)

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(dpath)  # Get all the files in that directory
#print("Files in %r: %s" % (dpath, files))



important = []
with open(infile) as f:
	for line in f:
		important.append(line)
print( len(important) )
print(important[0])