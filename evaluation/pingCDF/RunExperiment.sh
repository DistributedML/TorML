#!/bin/bash

#This is an example script which launches a cluster, moniters the
#execution of the cluster, and collects resources, and cleans up
#afterwards.

cluster=../control/master/ClusterControl.sh

#server parameters
threshold=0.01
samplerate=1

#curator parameters
curatorname="CuratorCerf"
model="models"
minClients=25

#client parameters
dataset="credit"
clientcount=25
latency=0
goodpercent=100
tor=true
servername=198.162.52.147
onionservice="33bwoexeu3sjrxoe.onion"
diffpriv=0.5

#TODO launch TOR first 

$cluster "kill"
#$cluster "pull"
$cluster "server" $threshold $samplerate
#give the server time to warm up
sleep 5
$cluster "curator" $curatorname $model $minClients 
#give the curator time to commit its model
sleep 5

$cluster "clients" $model $dataset $clientcount $latency $goodpercent $tor $servername $onionservice $diffpriv
sleep 120

$cluster "kill"

$cluster "getpings"

#TODO graph a dule cdf with the two ping sets




