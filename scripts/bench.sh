#!/bin/bash

sudo ./scripts/down.sh
python ./scripts/get_perf_plot.py
sudo ./scripts/up.sh
