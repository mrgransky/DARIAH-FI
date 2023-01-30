#!/bin/bash

: '
################## 1) Running via Bash Script & hardcoding input arguments ##################
myQUERY="Rusanen"
myORDERBY="DATE_DESC"
myFORMATS='["NEWSPAPER"]'
myFUZZY="false"
myPubPlace='["Iisalmi", "Kuopio"]'
myLANG='["FIN"]'
################## 1) Running via Bash Script & hardcoding input arguments ##################
# Result: OK! a json file with retreived expected information
'

#: '
################## 2) Running from python script with input arguments ##################
for ARGUMENT in "$@"
do
	#echo "$ARGUMENT"
	 KEY=$(echo $ARGUMENT | cut -f1 -d=)
	 KEY_LENGTH=${#KEY}
	 VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	 export "$KEY"="$VALUE"
done
echo $# "ARGS:" $*
################## 2) Running from python script with input arguments ##################
# Result: Error!!
#'

out_file_name="newspaper_info_query_${myQUERY// /_}.json"

echo ">> Running $0 | Searching for QUERY: $myQUERY | Saving in $out_file_name"

curl 'https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset=0&count=10000' \
-H 'Accept: application/json, text/plain, */*' \
-H 'Cache-Control: no-cache' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Pragma: no-cache' \
--compressed \
--output $out_file_name \
-d @- <<EOF
{	"query":"$myQUERY",
	"languages":$myLANG,
	"formats":$myFORMATS,
	"orderBy":"$myORDERBY",
	"fuzzy":$myFUZZY,
	"publicationPlaces": $myPubPlace
}
EOF