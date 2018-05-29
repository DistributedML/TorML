#!/bin/bash

#This script generates TOR config files for TorMentor clients.

#starting index is 0, all entries in bridges.txt contain three bridges
#followed by a dashed line delimeter
function ExtractGateWays {
    echo "extracting the $1'th set of gateways"
    let "start=(($1*4)+1)"
    echo $start
    let "end=(($1*4)+3)"
    echo $end
    cat bridges.txt | sed -n "${start},${end}p"
}

ExtractGateWays 0
echo cut
ExtractGateWays 1

