#!/bin/bash

## run using command:
## $ nohup bash pouta_user_token.sh 0 > /dev/null 2>&1 &
## $ nohup bash pouta_user_token.sh 0 > /media/volume/trash/NLF/check_output.out 2>&1 & # with output saved in check_output.out
## $ nohup parallel -j 8 "bash pouta_user_token.sh {}" ::: {0..731} > /dev/null 2>&1 &

USR_NAME="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$USR_NAME began job: `date`"
ch="#"

echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
HOME_DIR=$(echo $HOME)
# STORAGE_DIR="/media/volume" # Pouta
STORAGE_DIR="$HOME_DIR/datasets" # local
DATAFRAME_DIR="$STORAGE_DIR/Nationalbiblioteket/dataframes_x2"
files=($STORAGE_DIR/Nationalbiblioteket/datasets/*.dump)
echo "HOME DIR $HOME_DIR"
echo "STORAGE_DIR: $STORAGE_DIR"
echo "DATAFRAME_DIR: $DATAFRAME_DIR"
source $HOME_DIR/miniconda3/bin/activate py39
# maxNumFeatures=$(awk -v x="1.9e+6" 'BEGIN {printf("%d\n",x)}') # adjust values 2.2e+6
maxNumFeatures=-1
# Get the input integer argument or set default value to 0
qIDX=${1:-0} # 0 by default!
echo "maxNumFeat: $maxNumFeatures"
echo "${stars// /*}"

# Check if the input index is within the range of available files
if [ $qIDX -ge 0 ] && [ $qIDX -lt ${#files[@]} ]; then
echo "Processing Q[$qIDX]: ${files[$qIDX]}"
	python -u user_token.py \
		--inputDF ${files[$qIDX]} \
		--outDIR $DATAFRAME_DIR \
		--lmMethod 'stanza' \
		--qphrase 'Helsingin Pörssi ja Suomen Pankki' \
		--maxNumFeat $maxNumFeatures >>$STORAGE_DIR/Nationalbiblioteket/trash/nikeQ_$qIDX.out 2>&1 &
else
	echo "<!> Error: Invalid input query index: $qIDX"
	echo "Please provide a valid query index between 0 and $((${#files[@]} - 1))."
fi

done_txt="$USR_NAME finished job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"