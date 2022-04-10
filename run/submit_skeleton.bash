#!/bin/bash
# Please submit this file from the shape_al conda environment, which contains 
# scipy mpi4py h5py cython pytest matplotlib flake8 pandas scikit-learn, nbodykit, pynverse

dm_type_list="fdm" 
snap_list="024 026 028 030 032 034"

# Part 0
cd ../output
bash create_dir.bash
cd ../src/shape_al/disperse

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vault/td448/DisPerSE/external/gmp/gmp-4.2.4/install/usr/local/lib # only for HTCondor
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vault/td448/DisPerSE/external/libc6/glibc-2.14/install/lib # not for HTCondor
    
for dm_type in ${dm_type_list}; do
    for snap in ${snap_list}; do
    
        # Part 1: Generating density grid
        python3 make_fits.py -l "${dm_type} ${snap}"
        
        # Part 2: Calculating Morse-Smale complex with cut 0, i.e. retaining even least significant fils
        ./mse DM_"${dm_type}"_"${snap}".fits -cut 0 -periodicity 111 -outName DM_"${dm_type}"_"${snap}".NDnet -noTags -upSkl -forceLoops
        ./skelconv DM_"${dm_type}"_"${snap}".NDnet.up.NDskl -outName DM_"${dm_type}"_"${snap}".NDnet.up.NDskl -noTags -smooth 10 -to crits_ascii
        
        # Part 3: Finding 5 sigma, i.e. the difference of the density value of each pair, 5 * RMS
        LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's/:\/data\/vault\/td448\/DisPerSE\/external\/libc6\/glibc-2.14\/install\/lib\(:\|$\)//')
        python3 five_sigma.py -l "${dm_type} ${snap}"
        key=$(awk -F'five_sig = ' '{print $2}' five_sig_"${dm_type}"_"${snap}".txt)
        temp="${key%\"}" # Remove suffix quote
        temp="${temp//[[:space:]]}" # Remove all whitespaces
        five_sig="${temp#\"}" # Remove prefix quote. Note that five_sig is a string. Apply "$(($five_sig))" to make it into integer.
        ./skelconv DM_"${dm_type}"_"${snap}".NDnet.up.NDskl -outName DM_"${dm_type}"_"${snap}".NDskl -noTags -smooth 10 -to vtp
        rm DM_"${dm_type}"_"${snap}".NDskl.vtp
        rm DM_"${dm_type}"_"${snap}".NDnet.up.NDskl
        rm DM_"${dm_type}"_"${snap}".NDnet.MSC
        rm DM_"${dm_type}"_"${snap}".NDnet.up.NDskl.a.crits
        
        # Part 4: Calculating Morse-Smale complex with cut just found
        #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/vault/td448/DisPerSE/external/libc6/glibc-2.14/install/lib # not for HTCondor
        ./mse DM_"${dm_type}"_"${snap}".fits -cut "$five_sig" -periodicity 111 -outName DM_"${dm_type}"_"${snap}".NDnet -noTags -upSkl -forceLoops
        ./skelconv DM_"${dm_type}"_"${snap}".NDnet.up.NDskl -outName DM_"${dm_type}"_"${snap}".NDnet.up.NDskl -noTags -smooth 10 -to NDskl_ascii
        ./skelconv DM_"${dm_type}"_"${snap}".NDnet.up.NDskl -outName DM_"${dm_type}"_"${snap}".NDskl -noTags -smooth 10 -to vtp
        
        # Part 5: Extract the filaments' spine
        LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's/:\/data\/vault\/td448\/DisPerSE\/external\/libc6\/glibc-2.14\/install\/lib\(:\|$\)//')
        mpirun -n 8 python3 extract_spine.py -l "${dm_type} ${snap}"
        
        # Part 6: Cleanup
        rm DM_"${dm_type}"_"${snap}".NDnet.up.NDskl
        rm DM_"${dm_type}"_"${snap}".NDnet.MSC
        rm DM_"${dm_type}"_"${snap}".NDnet.up.NDskl.a.NDskl
        mv DM_"${dm_type}"_"${snap}".NDskl.vtp ../../../output/skeleton
        mv samples_"${dm_type}"_"${snap}".txt ../../../output/skeleton
        mv sampling_pos_"${dm_type}"_"${snap}".txt ../../../output/skeleton
    done
done