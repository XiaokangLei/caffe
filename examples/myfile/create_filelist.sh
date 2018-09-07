#!/usr/bin/env sh
    DATA=data/re/
    MY=examples/myfile

    t=0
    echo "Create train.txt..."
    rm -rf $MY/train.txt
    for i in 3 4 5 6 7
    do
    t=$(($i-3))
    find $DATA/train -name $i*.jpg | cut -d '/' -f4-5 | sed "s/$/ $t/">>$MY/train.txt
    done
    echo "Create test.txt..."
    rm -rf $MY/test.txt
    for i in 3 4 5 6 7
    do
    t=$(($i-3))
    find $DATA/test -name $i*.jpg | cut -d '/' -f4-5 | sed "s/$/ $t/">>$MY/test.txt
    done
    echo "All done"


