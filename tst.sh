#!/bin/bash
# A simple script

query=bash
curl 'https://mywiki.wooledge.org'
-F action=fullsearch \
-F fullsearch=text \
--form-string value="$query" \
${context+--form-string context="$context"}