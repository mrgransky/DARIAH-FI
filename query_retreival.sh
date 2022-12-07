#!/bin/bash
# A simple script

#myQUERY="economic crisis"
myQUERY=$1
#myQUERY=""
#myFORMATS='["JOURNAL","PRINTING","NEWSPAPER"]'
myFORMATS=$2
myFUZZY="false"

out_file_name="results_${myQUERY}.json"
echo ">> Running $0 | Searching for QUERY: $myQUERY"

curl 'https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset=0&count=10000' \
-H 'Accept: application/json, text/plain, */*' \
-H 'Cache-Control: no-cache' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Pragma: no-cache' \
--compressed \
--output $out_file_name \
-d @- <<EOF
{   "authors":[],
    "collections":[],
    "districts":[],
    "endDate":null,
    "formats":$myFORMATS,
    "fuzzy":$myFUZZY,
    "hasIllustrations":false,
    "importStartDate":null,
    "importTime":"ANY",
    "includeUnauthorizedResults":false,
    "languages":[],
    "orderBy":"RELEVANCE",
    "pages":"",
    "publicationPlaces":[],
    "publications":[],
    "publishers":[],
    "query":"$myQUERY", 
    "queryTargetsMetadata":false,
    "queryTargetsOcrText":true,
    "requireAllKeywords":true,
    "searchForBindings":false,
    "showLastPage":false,
    "startDate":null,
    "tags":[]
} 
EOF