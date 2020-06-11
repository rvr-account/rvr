#!/bin/bash


base="/n/home01/rhong/rvr/sweeps/"
#base="/n/home06/fding/rvr/sweeps/"
#file="/Users/Frances/Documents/seas-fellowship/rvr/sweeps/runp1_2_sweep/commands.sh"
sweep="mnist_simple_sweep"
file=${base}${sweep}"/commands.sh"
out=${base}${sweep}"/array_commands/"

count=0
subcount=0
numcommands=1

while IFS= read -r line
do
	echo $line >> $out$count.sh
	let "subcount++"
	if [ $subcount = $numcommands ]; then
	    let "count++"
	    subcount=0
	fi
done < $file
