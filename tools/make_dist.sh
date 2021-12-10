
#! /usr/bin/env bash

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"
cd ../

find easyctr/ -name '*.pyc' | xargs -i rm -rf {}
find common/ -name '*.pyc' | xargs -i rm -rf {}

zip -r easyctr.zip easyctr/ common/ main.py --exclude=*.pyc*
tar cvzf easyctr.tar.gz easyctr/ common/ main.py --exclude=*.pyc*
