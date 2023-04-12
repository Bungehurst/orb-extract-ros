# orb-extract-ros
This repo realizes two ORB extractors, one depends on OpenCV; the other is made by hand from scratch.
# Prerequisites
- ROS Noetic
- OpenCV4.1.0 or above
# How to run?
After you have built the project, you can run the following command to see the matching results.
```bash
rosrun orb_extract orb_self src/orb_extract/img/1.png src/orb_extract/img/2.png
rosrun orb_extract orb_opencv src/orb_extract/img/1.png src/orb_extract/img/2.png
```