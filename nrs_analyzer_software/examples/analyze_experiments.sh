#!/usr/bin/env bash
set -e
nrs-analyze experiments.zip -o narrativeforge_output --input-order auto,brief,rag,full --prompt-order short,standard,detailed
