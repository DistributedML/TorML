#!/bin/bash

#This is an example script which launches a cluster, moniters the
#execution of the cluster, and collects resources, and cleans up
#afterwards.

cluster=../control/master/ClusterControl.sh

#server parameters
threshold=0.001
samplerate=1000

#curator parameters
curatorname="CuratorCerf"
model="study"
minClients=6

#client parameters
dataset="credit"
clientcount=24
latency=0
goodpercent=6
tor=true
servername=198.162.52.147
onionservice="33bwoexeu3sjrxoe.onion"
diffpriv=1000

#TODO launch TOR first 

$cluster "kill"
$cluster "server" $threshold $samplerate
#give the server time to warm up
sleep 5
$cluster "curator" $curatorname $model $minClients
#give the curator time to commit its model
sleep 5

$cluster "clients" $model $dataset $clientcount $latency $goodpercent $tor $servername $onionservice $diffpriv

#TODO collect the csv of the first run and graph it




