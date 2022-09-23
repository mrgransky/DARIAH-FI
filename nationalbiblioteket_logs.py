import os
import numpy as np


os.system('clear')
dpath = f"{os.environ['HOME']}/Datasets/Nationalbiblioteket/no_ip_logs"
fname = "nike6.docworks.lib.helsinki.fi_access_log.2021-02-07.log"

infile = os.path.join(dpath, fname)

#print(f">> Loading: {infile}")

# https://note.nkmk.me/en/python-str-replace-translate-re-sub/


important = []
with open(infile) as f:
	for line in f:
		important.append(line)
qidx = int( np.random.randint(0, len(important)+1, size=1) )
#print(f">> Loading complete! no_logs: {len(important)}")
print(f">> q:{qidx} {important[qidx]}")
#print(important[qidx].split(""))