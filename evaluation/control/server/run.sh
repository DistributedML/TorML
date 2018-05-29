#!/bin/bash

#Parameters
#Threshold: The server will continue to learn untill this threshold is reached
#SampleRate: How often the server writes out it's state to a csv

threshold=$1
samplerate=$2

cd ~/go/src/github.com/wantonsolutions/TorML/DistSys/

go run torserver.go $threshold $samplerate &
