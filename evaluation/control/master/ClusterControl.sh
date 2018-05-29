#!/bin/bash 

declare -A vms


username='stewbertgrant'

bootstrap='rm TorMentorAzureInstall.sh;
           wget https://raw.githubusercontent.com/wantonsolutions/TorML/master/evaluation/TorMentorAzureInstall.sh;
           chmod 755 TorMentorAzureInstall.sh;
           sudo sh -c "yes | ./TorMentorAzureInstall.sh"'

firstcommand='echo hello $HOSTNAME'
pingcommand='ping -c 1 198.162.52.147 | tail -1 | cut -d / -f 5 > $HOSTNAME.ping'
pinglocation="/home/stewbertgrant/*ping"
torpinglocation="/home/stewbertgrant/go/src/github.com/wantonsolutions/TorML/DistSys/*torping"
permission="sudo chown -R stewbertgrant go"


sysdir="/home/stewartgrant/go/src/github.com/wantonsolutions/TorML"
vmlist="$sysdir/evaluation/control/master/nameip.txt"

#uses remote access
pull='cd go/src/github.com/wantonsolutions/TorML/DistSys/; git pull'
clientscript="go/src/github.com/wantonsolutions/TorML/evaluation/control/client/run.sh"

#local
serverscript="$sysdir/evaluation/control/server/run.sh"
curatorscript="$sysdir/evaluation/control/curator/run.sh"

killeverything='killall torserver; killall torclient; killall torcurator'

## $1 is the filename to read vms from
function readVMs {
IFS=$'\n'
#set -f
    for line in $(cat $1);do
        vmname=`echo $line | cut -d, -f1`
        vmpubip=`echo $line | cut -d, -f2`
        vms["$vmname"]="$vmpubip"
    done
echo ${vms[@]}
IFS=$' \t\n'
}

function yeshello {
    for vm in ${vms[@]}
    do
        ssh $username@$vm -oStrictHostKeyChecking=no -x 'echo $HOSTNAME'
    done
}

function onall {
    echo running $1
    for vm in ${vms[@]}
    do
        ssh $username@$vm -x $1 
        break
    done
}

function onallasync {
    echo running $1
    for vm in ${vms[@]}
    do
        ssh $username@$vm -x $1 &
    done
}

function getall {
    echo grabbing $1
    for vm in ${vms[@]}
    do
        scp $username@$vm:$1 $2
    done
}
function getallasync {
    echo grabbing $1
    for vm in ${vms[@]}
    do
        scp $username@$vm:$1 $2 &
    done
}

function getPings {
    onallasync "$pingcommand"
    sleep 10
    getallasync "$pinglocation" ./
    getallasync "$torpinglocation" ./
    sleep 20
    cat *.ping > regular.ping
    for torfile in *.torping
    do
        tail -1 $torfile >> tor.ping
    done
    mkdir -p ping
    rm ping/*
    mv tor.ping regular.ping ping
    rm *.ping
    rm *.torping
IFS=$' \t\n'
}

function installAll {
    onallasync "$bootstrap"
    sleep 500
}

function killAll {
    $killeverything
    onallasync "$killeverything"
    sleep 10
}

function runclients {
    modelname=$1
    dataset=$2
    clientcount=$3
    latency=$4
    goodpercent=$5
    tor=$6
    servername=$7
    onionservice=$8
    diffpriv=$9

    #main loop starts all individual clients async
    for (( i=1; i<=$clientcount; ))
    do
        #cycle through vms continuously for even distribution
        for vm in ${vms[@]}
        do
            clientnumber=$i
            if [ $i -gt $clientcount ];then
                break
            fi
            #calculagte adversaries
            let "gb = i % 100"
            adversary=false
            if [ $gb -gt $goodpercent ];then
                adversary=true
                echo "launching an adversary client"
            fi

            #TODO allow for distributions to be calculated
            clientlatency=0
            case $latency in
            *)
                clientlatency=0
                ;;
            esac

            ssh $username@$vm -x "$clientscript $modelname $dataset $clientnumber $diffpriv $tor $servername $onionservice $adversary $clientlatency" &
            let "i=$i + 1"
        done
        sleep 30
    done
}

function runcurator {
    curatorname=$1
    model=$2
    clients=$3
    #The expectation is that the currator is run locally
    killall torcurator
    $curatorscript $curatorname $model $clients 
}

function runserver {
    threshold=$1
    samplerate=$2
    killall torserver
    $serverscript $threshold $samplerate
}
    
readVMs $vmlist 
command=$1

case $command in
    
"kill")
    killAll
    ;;
"getpings")
    getPings
    ;;
"pull")
    onallasync "$pull"
    ;;
"permission")
    onallasync "$permission"
    ;;
"clients")
    #The global parameters to the clients command:
    modelname=$2
    dataset=$3
    clientcount=$4
    latency=$5
    goodpercent=$6
    tor=$7
    servername=$8
    onionservice=$9
    diffpriv=${10}
    runclients $modelname $dataset $clientcount $latency $goodpercent $tor $servername $onionservice $diffpriv
    ;;
"curator")
    curatorname=$2
    model=$3
    clients=$4
    runcurator $curatorname $model $clients
    ;;
"server")
    threshold=$2
    samplerate=$3
    runserver $threshold $samplerate
    ;;

*)
    echo "Error unknown cluster command: $command"
    ;;
esac

#killAll
#onallasync "$runclient"
#yeshello
#installAll
#getPings
#onall "$bootstrap"

#onallasync "$pull"
#onallasync "$permission"
#onallasync "$pull"
#onall "$firstcommand"
#onall "$pingcommand"
#sleep 10
#onall "$runclient"
#getall "$pinglocation" ./
