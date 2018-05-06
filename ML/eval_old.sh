#!/bin/bash



ARG='{"models": ["mnist0", "mnist1","mnist2","mnist3", "mnist4", "mnist5", "mnist6", "mnist7", "mnist8", "mnist9"'
ATTACKS=',"attacks": [[1,7]] }'

for i in `seq 0 2`;
do #append 17 i times to ARG
  ii=0
  NEWARG=$ARG
  while [ $ii -lt $i ]
  do
    NEWARG=$NEWARG', "mnist_bad_17"'
    (( ii++ ))
  done

  for j in `seq 0 2`;
  do
    jj=0
    NEWARG2=$NEWARG
    while [ $jj -lt $j ]
    do
      NEWARG2=$NEWARG2', "mnist_bad_49"'
      (( jj++ ))
    done

    for k in `seq 0 2`;
    do
      kk=0
      NEWARG3=$NEWARG2
      while [ $kk -lt $k ]
      do
        NEWARG3=$NEWARG3', "mnist_bad_08"'
        (( kk++ ))
      done

      NEWARG3=$NEWARG3"]"$ATTACKS
      echo $NEWARG3
    done
  done
done



# NEWARG=$ARG'", mnist_bad_17", "mnist_bad_17"]'$ATTACKS
# echo $NEWARG

#python code/ML_main.py "$ARG" > $FILE


# SEED="SEED"
# for i in `seq 0 5`;
# do
#   ii=0
#   NEWSEED=SEED
#   while [ $ii -lt $i ]
#   do
#     NEWSEED=$NEWSEED"mnist_bad_17"
#     ((ii++))
#   done
#   echo $NEWSEED
# done

# for i in `seq 0 5`;
# do for j in `seq 0 5`;
#   do for k in `seq 0 5`;
#     do
#
#     done
#   done
# done
