#!/bin/bash
for sg in 0.5 1.5; do for L in 6 8 10 12; do ./levy $sg 6 0.25 40 7 $L lv_${sg}_$L.txt; done; done
./levy 0.5 6 0.25 16 7 14 lv_0.5_14.txt
./levy 1.5 6 0.25 10 7 14 lv_1.5_14.txt
