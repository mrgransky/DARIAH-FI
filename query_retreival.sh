#!/bin/bash

if [[ $# -eq 0 ]] ; then
	echo 'no input args --> EXIT!'
	#echo 'Ex) bash $0 QUERY='kantasonni' DOC_TYPE='["JOURNAL","PRINTING","NEWSPAPER"]''
	exit 1
fi

for ARGUMENT in "$@"
do
	echo "$ARGUMENT"
	 KEY=$(echo $ARGUMENT | cut -f1 -d=)

	 KEY_LENGTH=${#KEY}
	 VALUE="${ARGUMENT:$KEY_LENGTH+1}"

	 export "$KEY"="$VALUE"
done
echo "ARGUMENTS COUNT : " $#
echo "ARGUMENTS LIST  : " $*

myFUZZY="false"

out_file_name="newspaper_info_query_${QUERY// /_}.json"
echo ">> Saving DIR: $PWD/$out_file_name"

if [ -z "$AUTHOR" ]; then AUTHOR="${AUTHOR:-[]}"; fi
if [ -z "$COLLECTION" ]; then COLLECTION="${COLLECTION:-[]}"; fi
if [ -z "$DISTRICT" ]; then DISTRICT="${DISTRICT:-[]}"; fi
if [ -z "$END_DATE" ]; then END_DATE="${END_DATE:-null}"; fi
if [ -z "$DOC_TYPE" ]; then DOC_TYPE="${DOC_TYPE:-[]}"; fi
if [ -z "$FUZZY_SEARCH" ]; then FUZZY_SEARCH="${FUZZY_SEARCH:-false}"; fi
if [ -z "$HAS_ILLUSTRATION" ]; then HAS_ILLUSTRATION="${HAS_ILLUSTRATION:-false}"; fi
if [ -z "$IMPORT_START_DATE" ]; then IMPORT_START_DATE="${IMPORT_START_DATE:-null}"; fi
if [ -z "$IMPORT_TIME" ]; then IMPORT_TIME="${IMPORT_TIME:-ANY}"; fi
if [ -z "$INCLUDE_AUTHORIZED_RESULTS" ]; then INCLUDE_AUTHORIZED_RESULTS="${INCLUDE_AUTHORIZED_RESULTS:-false}"; fi
if [ -z "$LNGs" ]; then LNGs="${LNGs:-[]}"; fi
if [ -z "$ORDER_BY" ]; then ORDER_BY="${ORDER_BY:-}"; fi
if [ -z "$PAGES" ]; then PAGES="${PAGES:-}"; fi
if [ -z "$PUB_PLACE" ]; then PUB_PLACE="${PUB_PLACE:-[]}"; fi
if [ -z "$PUBLICATION" ]; then PUBLICATION="${PUBLICATION:-[]}"; fi
if [ -z "$PUBLISHER" ]; then PUBLISHER="${PUBLISHER:-[]}"; fi
if [ -z "$QUERY_TARGETS_METADATA" ]; then QUERY_TARGETS_METADATA="${QUERY_TARGETS_METADATA:-false}"; fi
if [ -z "$QUERY_TARGETS_OCRTEXT" ]; then QUERY_TARGETS_OCRTEXT="${QUERY_TARGETS_OCRTEXT:-true}"; fi
if [ -z "$REQUIRE_ALL_KEYWORDS" ]; then REQUIRE_ALL_KEYWORDS="${REQUIRE_ALL_KEYWORDS:-true}"; fi
if [ -z "$SEARCH_FOR_BINDINGS" ]; then SEARCH_FOR_BINDINGS="${SEARCH_FOR_BINDINGS:-false}"; fi
if [ -z "$SHOW_LAST_PAGE" ]; then SHOW_LAST_PAGE="${SHOW_LAST_PAGE:-false}"; fi
if [ -z "$START_DATE" ]; then START_DATE="${START_DATE:-null}"; fi
if [ -z "$TAG" ]; then TAG="${TAG:-[]}"; fi

echo "documents: $DOC_TYPE"
echo "collections: $COLLECTION"
echo "lang: $LNGs"
echo "fuzzy: $FUZZY_SEARCH"
echo "AUTHOR: $AUTHOR"
echo "END_DATE: $END_DATE"
echo "IMPORT_START_DATE: $IMPORT_START_DATE"
echo "IMPORT_TIME: $IMPORT_TIME"
echo "INCLUDE_AUTHORIZED_RESULTS: $INCLUDE_AUTHORIZED_RESULTS"
echo "ORDER_BY: $ORDER_BY"
echo "PAGES: $PAGES"
echo "START_DATE: $START_DATE"

#: "
curl \
-v 'https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset=0&count=10000' \
-o $out_file_name \
-H 'Accept: application/json, text/plain, */*' \
-H 'Cache-Control: no-cache' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Pragma: no-cache' \
--compressed \
-d @- <<EOF
{   "authors":$AUTHOR,
		"collections":$COLLECTION,
		"districts":$DISTRICT,
		"endDate":$END_DATE,
		"formats":$DOC_TYPE,
		"fuzzy":$FUZZY_SEARCH,
		"hasIllustrations":$HAS_ILLUSTRATION,
		"importStartDate":$IMPORT_START_DATE,
		"importTime":"$IMPORT_TIME",
		"includeUnauthorizedResults":$INCLUDE_AUTHORIZED_RESULTS,
		"languages":$LNGs,
		"orderBy":"$ORDER_BY",
		"pages":"$PAGES",
		"publicationPlaces":$PUB_PLACE,
		"publications":$PUBLICATION,
		"publishers":$PUBLISHER,
		"query":"$QUERY", 
		"queryTargetsMetadata":$QUERY_TARGETS_METADATA,
		"queryTargetsOcrText":$QUERY_TARGETS_OCRTEXT,
		"requireAllKeywords":$REQUIRE_ALL_KEYWORDS,
		"searchForBindings":$SEARCH_FOR_BINDINGS,
		"showLastPage":$SHOW_LAST_PAGE,
		"startDate":$START_DATE,
		"tags":$TAG
}
EOF
#	"