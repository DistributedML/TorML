#!/bin/bash

#$1 The name of the curator
#$2 The model

curatorName=$1
model=$2
clients=$3

#move to the distributed system directory
cd ~/go/src/github.com/wantonsolutions/TorML/DistSys/

#if [ -e $model ];then
#    echo starting curator: $curatorName with model: $model
#else
#    echo $model does not exist unable to start curator
#    exit
#fi

go run torcurator.go $curatorName $model $clients


