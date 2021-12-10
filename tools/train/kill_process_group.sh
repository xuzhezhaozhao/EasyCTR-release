#! /usr/bin/env bash

if [[ $# != 1 ]]; then
    echo "Usage: <pid>"
    exit -1
fi

pid=$1

echo "kill training process group [${pid}] ..."
kill $(pstree ${pid} -p -a -l | cut -d, -f2 | cut -d' ' -f1)
