#!/bin/bash
# Pack dataset/<name>/split into data_tars/<name>.tar for NVMe staging (run on a login/dtn node).
#   frontier/make_tar.sh dota
set -e
P=/lustre/orion/proj-shared/cli138/z7a/Project
mkdir -p $P/data_tars
tar -cf $P/data_tars/$1.tar -C $P/dataset/$1/split .
ls -lh $P/data_tars/$1.tar
