
#!/bin/bash

# myQUERY="Rusanen"
# myORDERBY="DATE_DESC"
# myFORMATS='["NEWSPAPER"]'
# myFUZZY="false"
# myPubPlace='["Iisalmi", "Kuopio"]'
# myLANG='["FIN"]'

# out_file_name="newspaper_info_query_${myQUERY// /_}.json"

# echo ">> Running $0 | Searching for QUERY: $myQUERY | Saving in $out_file_name"

# curl 'https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset=0&count=10000' \
# -H 'Accept: application/json, text/plain, */*' \
# -H 'Cache-Control: no-cache' \
# -H 'Connection: keep-alive' \
# -H 'Content-Type: application/json' \
# -H 'Pragma: no-cache' \
# --compressed \
# --output $out_file_name \
# -d @- <<EOF
# {	"authors":[],
# 	"collections":[],
# 	"districts":[],
# 	"endDate":null,
# 	"query":"$myQUERY",
# 	"languages":$myLANG,
# 	"formats":$myFORMATS,
# 	"orderBy":"$myORDERBY",
# 	"fuzzy":$myFUZZY,
# 	"publicationPlaces": $myPubPlace,
# 	"hasIllustrations":false,
# 	"importStartDate":null,
# 	"importTime":"ANY",
# 	"includeUnauthorizedResults":false,
# 	"pages":"",
# 	"publications": [],
# 	"publishers":[],
# 	"queryTargetsMetadata":false,
# 	"queryTargetsOcrText":true,
# 	"requireAllKeywords":true,
# 	"searchForBindings":false,
# 	"showLastPage":false,
# 	"startDate":null,
# 	"tags":[]
# }
# EOF

search_dir=(/home/xenial/WS_Farid/DARIAH-FI/*.py)
# for entry in "$search_dir"/*
# do
#   echo "$entry"
# done
# echo "###"
# ls -d "$PWD"/*
# echo "................."
# ls | cat -n

# my_string="One;Two;Three"
# my_array=($(echo $my_string | tr ";" "\n"))

# #Print the split string
# count=0
# for i in "${my_array[@]}"
# do
#     count=$(($count+1))
#     echo $count - $i
# done

for f in "${search_dir[@]}"; do
   echo "$f"
done

echo "-----------------"
echo ${search_dir[0]}