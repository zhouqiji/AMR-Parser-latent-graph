#!/usr/bin/env bash

#!/bin/bash

set -e

pred=$1
gold=$2



cp $pred $gold evaluation_tools/amr-evaluation-tool-enhanced
cd evaluation_tools/amr-evaluation-tool-enhanced && ./evaluation.sh test.pred.txt test.txt