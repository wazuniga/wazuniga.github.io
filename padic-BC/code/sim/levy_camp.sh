#!/bin/bash
# critical point and exponents on the Euclidean chain with power-law coupling (mode 3)
for sg in 0.5 1.0 0.75 1.5 0.25; do python3 extra.py levy 3 $sg 1.0 4.0 200000 > extra_levy_$sg.out 2>&1; done
echo CAMPAIGN DONE > levy_camp.done
