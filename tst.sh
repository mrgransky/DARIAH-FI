
#!/bin/bash

myQUERY="Rusanen"
myORDERBY="DATE_DESC"
myFORMATS='["NEWSPAPER"]'
myFUZZY="false"
myPubPlace='["Iisalmi", "Kuopio"]'
myLANG='["FIN"]'

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
{	"authors":[],
	"collections":[],
	"districts":[],
	"endDate":null,
	"query":"$myQUERY",
	"languages":$myLANG,
	"formats":$myFORMATS,
	"orderBy":"$myORDERBY",
	"fuzzy":$myFUZZY,
	"publicationPlaces": $myPubPlace,
	"hasIllustrations":false,
	"importStartDate":null,
	"importTime":"ANY",
	"includeUnauthorizedResults":false,
	"pages":"",
	"publications": [],
	"publishers":[],
	"queryTargetsMetadata":false,
	"queryTargetsOcrText":true,
	"requireAllKeywords":true,
	"searchForBindings":false,
	"showLastPage":false,
	"startDate":null,
	"tags":[]
}
EOF