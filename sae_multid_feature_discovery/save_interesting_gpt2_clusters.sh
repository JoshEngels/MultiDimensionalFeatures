#!/bin/bash

for i in 30 80 53 69 183 66 78 251 125 83 39 97 45 94 126 12 6 4 18 10; do
    python3 gpt2_interactive_figure.py --cluster $i
done