#! /bin/bash

current_path=$(pwd)
BRANCH=main
if [ ! -z "$1" ]; then
    echo "pull branch: $1"
    BRANCH=$1
fi

echo "-----------------------------------------------------------------------"
echo "-------------------------pull Grasping-with-SSDRL ---------------------"
echo "-----------------------------------------------------------------------"
git pull

CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in Grasping-with-SSDRL. Aborting"
   return 1
fi

BRANCH=indigo-devel

echo "-----------------------------------------------------------------------"
echo "-------------------------pull apriltags -------------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/catkin_ws/src/sensor/apriltags
git checkout $BRANCH
git pull



CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in apriltags. Aborting"
   return 1
fi

BRANCH=melodic

echo "-----------------------------------------------------------------------"
echo "-------------------------pull vision_opencv ---------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/catkin_ws/src/sensor/vision_opencv
git checkout $BRANCH
git pull



CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in vision_opencv. Aborting"
   return 1
fi

BRANCH=2.2.15

echo "-----------------------------------------------------------------------"
echo "-------------------------pull realsense-ros ---------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/catkin_ws/src/sensor/realsense-ros
git checkout $BRANCH
git pull

CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in realsense-ros. Aborting"
   return 1
fi

BRANCH=kinetic-devel

echo "-----------------------------------------------------------------------"
echo "-------------------------pull robotiq -------------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/catkin_ws/src/ur5/robotiq
git checkout $BRANCH
git pull

CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in robotiq. Aborting"
   return 1
fi

BRANCH=melodic-devel

echo "-----------------------------------------------------------------------"
echo "-------------------------pull universal_robot -------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/catkin_ws/src/ur5/universal_robot
git checkout $BRANCH
git pull

CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in universal_robot. Aborting"
   return 1
fi

BRANCH=kinetic-devel

echo "-----------------------------------------------------------------------"
echo "-------------------------pull ur_modern_driver -------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/catkin_ws/src/ur5/ur_modern_driver
git checkout $BRANCH
git pull

CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in ur_modern_driver. Aborting"
   return 1
fi


BRANCH=master

echo "-----------------------------------------------------------------------"
echo "-------------------------pull arm_operation ---------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/catkin_ws/src/ur5/arm_operation
git checkout $BRANCH
git pull

CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in arm_operation. Aborting"
   return 1
fi

cd $current_path
return 0