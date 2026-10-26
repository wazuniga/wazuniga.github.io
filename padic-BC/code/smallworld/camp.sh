#!/bin/bash
# coverage time T50: tree (several sigma) and ring, b=6, s0=0.25
run(){ # mode sigma L nruns tag
  ./cover $1 $2 6 0.25 $4 1e7 7 $3 cov_$5_$3.txt 1500000 2>>cov.err; }
for L in 6 8 10 12 14 16; do
  for sg in 0.25 0.5 0.75 1.0; do run 0 $sg $L 40 tree$sg; done
done
for L in 6 8 10 12; do run 1 0 $L 30 ring; done
for L in 6 8 10 12 14; do run 0 1.5 $L 24 tree1.5; done
run 1 0 14 8 ring
run 0 1.5 16 8 tree1.5
