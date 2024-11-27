#!/bin/bash

for i in 23 28 58 32 34 93 46 13 138 96 65 2 68 7 212 157 8 101 173 209; do
    python3 gpt2_interactive_figure.py --cluster $i
done