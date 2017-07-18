#!/usr/bin/env bash
python retrofit.py -l lexicons/framenet.txt -o framenet.pkl
python retrofit.py -l lexicons/ppdb-xl.txt -o ppdb.pkl
python retrofit.py -l lexicons/wordnet-synonyms+.txt -o wordnet.pkl