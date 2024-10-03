#!/bin/bash

rm -rf build

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

make -j32

cd ../

python3 test.py 2>&1 | tee run.log
