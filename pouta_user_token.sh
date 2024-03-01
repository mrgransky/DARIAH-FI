#!/bin/bash

## run using command:
## $ nohup bash pouta_gan.sh 0 > /dev/null 2>&1 &
## $ nohup bash pouta_user_token.sh 0 > /media/volume/trash/NLF/check_output.out 2>&1 & # with output saved in check_output.out

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began job: `date`"
ch="#"

echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
source $HOME_DIR/miniconda3/bin/activate py39

WDIR="/media/volume"
echo "HOME DIR $HOME_DIR | WDIR: $WDIR"
echo "${stars// /*}"

files=($WDIR/Nationalbiblioteket/datasets/*.dump)
ddir="$WDIR/Nationalbiblioteket/dataframes_x2"
# maxNumFeatures=$(awk -v x="1.9e+6" 'BEGIN {printf("%d\n",x)}') # adjust values 2.2e+6
maxNumFeatures=-1

# Get the input integer argument or set default value to 0
qIDX=${1:-0} # 0 by default!

echo "maxNumFeat: $maxNumFeatures | outDIR $ddir"
echo "Q[$qIDX]: ${files[$qIDX]}"

python -u user_token.py \
	--inputDF ${files[$qIDX]} \
	--outDIR $ddir \
	--lmMethod 'stanza' \
	--qphrase 'Helsingin Pörssi ja Suomen Pankki' \
	--maxNumFeat $maxNumFeatures >>$WDIR/trash/NLF/nikeQ_$qIDX.out 2>&1 &

done_txt="$user finished job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"