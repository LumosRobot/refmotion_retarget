#!/usr/bin/bash
#config: --utf-8
##!/bin/pyenv python

set -e 

robot_id="1"
while getopts "r:m:fsv" arg
do 
case $arg in 
r)
robot_id=$OPTARG
;;
m)
motion_name=$OPTARG
;;
f)
fit=1
;;
s)
shape=1
;;
v)
vis=1
;;
?)
echo "Unknow args"
;;
esac
done

echo "Robot id is ${robot_id} and Motion name is $motion_name in the retarget."


if [ -n "$shape" ]; then

    echo "fit smpl shape ..."
python -u fit_smpl_shape.py robot=${robot_id}_fitting
fi


if [ -n "$fit" ]; then

    echo "fit smpl motion ..."
python -u fit_smpl_motion.py robot=${robot_id}_fitting +motion_name=$motion_name
fi

if [ -n "$vis" ]; then
    echo "convert and vis retarget motion in mujoco ..."
    python -u ./convert_in_mj.py robot=${robot_id}_fitting +motion_name=$motion_name
fi


