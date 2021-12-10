#! /usr/bin/env bash

cpplint/cpplint.py \
    --exclude=assembler/jsoncpp_helper.hpp \
    --filter=-legal/copyright \
    --recursive assembler/
cpplint/cpplint.py  --filter=-legal/copyright --recursive ops/
#cpplint/cpplint.py  --filter=-legal/copyright --recursive test/
