#! /usr/bin/env bash

set -e

python conf.py > conf.json
python example.py ../build/ops/libassembler_ops.so conf.json
