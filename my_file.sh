#!/bin/bash

for ARGUMENT in "$@"
do
	#echo "$ARGUMENT"
	 KEY=$(echo $ARGUMENT | cut -f1 -d=)

	 KEY_LENGTH=${#KEY}
	 VALUE="${ARGUMENT:$KEY_LENGTH+1}"

	 export "$KEY"="$VALUE"
done
echo $# "ARGS:" $*

stars=$(printf '%*s' 100 '')
out_file_name="newspaper_info_query_${QUERY// /_}.json"
echo ">> Saving DIR: $PWD/$out_file_name"

if [ -z "$ORDER_BY" ]; then ORDER_BY=""; else ORDER_BY="\"orderBy\":\"$ORDER_BY\","; fi
if [ -z "$AUTHOR" ]; then AUTHOR="${AUTHOR:-[]}"; fi
if [ -z "$COLLECTION" ]; then COLLECTION="${COLLECTION:-[]}"; fi
if [ -z "$DISTRICT" ]; then DISTRICT="${DISTRICT:-[]}"; fi
if [ -z "$DOC_TYPE" ]; then DOC_TYPE="${DOC_TYPE:-[]}"; fi
if [ -z "$FUZZY_SEARCH" ]; then FUZZY_SEARCH="${FUZZY_SEARCH:-false}"; fi
if [ -z "$HAS_ILLUSTRATION" ]; then HAS_ILLUSTRATION="${HAS_ILLUSTRATION:-false}"; fi
if [ -z "$IMPORT_START_DATE" ]; then IMPORT_START_DATE="${IMPORT_START_DATE:-null}"; fi
if [ -z "$IMPORT_TIME" ]; then IMPORT_TIME="${IMPORT_TIME:-ANY}"; fi
if [ -z "$INCLUDE_AUTHORIZED_RESULTS" ]; then INCLUDE_AUTHORIZED_RESULTS="${INCLUDE_AUTHORIZED_RESULTS:-false}"; fi
if [ -z "$LANGUAGES" ]; then LANGUAGES="${LANGUAGES:-[]}"; fi
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
if [ -z "$END_DATE" ]; then END_DATE="${END_DATE:-null}"; fi
if [ -z "$TAG" ]; then TAG="${TAG:-[]}"; fi

echo "${stars// /*}"
echo "AUTHOR: $AUTHOR"
echo "collections: $COLLECTION"
echo "documents: $DOC_TYPE"
echo "ORDER_BY: $ORDER_BY"
echo "lang: $LANGUAGES"
echo "fuzzy: $FUZZY_SEARCH"
echo "has_illus: $HAS_ILLUSTRATION"
echo "IMPORT_START_DATE: $IMPORT_START_DATE"
echo "IMPORT_TIME: $IMPORT_TIME"
echo "include_authorized: $INCLUDE_AUTHORIZED_RESULTS"
echo "Q_tgt_meta: $QUERY_TARGETS_METADATA"
echo "Q_tgt_ocr: $QUERY_TARGETS_OCRTEXT"
echo "all keywords: $REQUIRE_ALL_KEYWORDS"
echo "binding search: $SEARCH_FOR_BINDINGS"
echo "show last pg? $SHOW_LAST_PAGE"
echo "pub place: $PUB_PLACE"
echo "publisher: $PUBLISHER"
echo "publication: $PUBLICATION"
echo "PAGES: $PAGES"
echo "start_date: $START_DATE | end_date: $END_DATE"
echo "tag: $TAG"
echo "${stars// /*}"

curl 'https://digi.kansalliskirjasto.fi/rest/binding-search/search/binding?offset=0&count=10000' \
-o $out_file_name \
-H 'Accept: application/json, text/plain, */*' \
-H 'Cache-Control: no-cache' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Pragma: no-cache' \
--compressed \
-d @- <<EOF
{   ${ORDER_BY}
		"query":"$QUERY",
		"authors":$AUTHOR,
		"collections":$COLLECTION,
		"districts":$DISTRICT,
		"endDate":$END_DATE,
		"formats":$DOC_TYPE,
		"fuzzy":$FUZZY_SEARCH,
		"hasIllustrations":$HAS_ILLUSTRATION,
		"importStartDate":$IMPORT_START_DATE,
		"importTime":"$IMPORT_TIME",
		"includeUnauthorizedResults":$INCLUDE_AUTHORIZED_RESULTS,
		"languages":$LANGUAGES,
		"pages":"$PAGES",
		"publications":$PUBLICATION,
		"publishers":$PUBLISHER,
		"queryTargetsMetadata":$QUERY_TARGETS_METADATA,
		"queryTargetsOcrText":$QUERY_TARGETS_OCRTEXT,
		"requireAllKeywords":$REQUIRE_ALL_KEYWORDS,
		"searchForBindings":$SEARCH_FOR_BINDINGS,
		"showLastPage":$SHOW_LAST_PAGE,
		"startDate":$START_DATE,
		"publicationPlaces":$PUB_PLACE,
		"tags":$TAG
}
EOF