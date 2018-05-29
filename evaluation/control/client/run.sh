#!/bin/bash -x

#This script controls the execution of a single client machine

#Parameters:
#modelname: the name of the model the client will request from the server and train on
#dataset: the name of the dataset the client starts with
#clientnumber: each client is issued a unique number, this number is appended to it's id, and is appended to the name of it's dataset to determine the specific dataset that the client will user
#epsilon: the ammount of differential privacy the client wants
#tor: True/False, if true the client will connect through tor, if not a regular tcp connection is opened
#serverip: The ip of the server, only used in the non tor case
#onionservice: name of the servers hidden onion service, used in the tor case
#adversary: True/False, if true the client will act as an adversary
#latency: artifical latenct to inject on each request

modelname=$1
dataset=$2
clientnumber=$3
epsilon=$4
tor=$5
servername=$6
onionservice=$7
adversary=$8
latency=$9

#TODO check that each of these parameters are sane


#calculagte adversaries
let "datanumber = $clientnumber % 100 + 1 "
truedatasetname=$dataset$datanumber
echo "$truedatasetname"

if [ "$adversary" = true ];then
    echo "$truedatasetname"
    truedatasetname="${truedatasetname}_b"
else
    echo "$truedatasetname"
    truedatasetname="${truedatasetname}_g"
fi

#if [ -e $truedatasetname ];then
#    echo starting client with the $truedatasetname dataset
#else
#    echo $truedatasetname does not exist
#    exit
#fi

#export go paths
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
export PATH=$PATH:/usr/local/go/bin

echo Resetting Tor
killall tor;
tor & sleep 10;
cd go/src/github.com/wantonsolutions/TorML/DistSys/

#run the client
go run torclient.go $HOSTNAME-$clientnumber $modelname $truedatasetname $epsilon $tor $servername $onionservice $adversary $latency
