#!/bin/bash

vms[0]=40.86.185.245

function onall {
    for vm in ${vms[*]}
    do
        ssh stew@$vm -x $1 &
    done
}
