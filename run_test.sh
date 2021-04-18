#!/bin/bash


echo "START"
echo "Comienza GCC"
./optimization_flags_wrapper_gcc
echo "Comienza CLANG"
./optimization_flags_wrapper_clang
echo "Comienza INTEL"
./optimization_flags_wrapper_intel
echo "END"
