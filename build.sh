#! /usr/bin/env bash

set -e

cd "$( dirname "${BASH_SOURCE[0]}" )"

mkdir -p build
cd build
cmake ..
make
cd ..
