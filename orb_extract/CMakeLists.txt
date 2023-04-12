cmake_minimum_required(VERSION 3.0.2)

project(orb_extract)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -mfma -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
    cmake_modules
    rosconsole 
    roscpp
)

find_package(OpenCV REQUIRED)
# find_package(Eigen3 REQUIRED)
# find_package(Ceres REQUIRED)
# find_package(Sophus REQUIRED)
# add_subdirectory(3rdparty/matplotplusplus)
# add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/g2o)

include_directories(
    include/${PROJECT_NAME}
    ${catkin_INCLUDE_DIRS} 
    # ${EIGEN_INCLUDE_DIRS}
    # ${CERES_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
    # ${G2O_INCLUDE_DIR}
    # ${Sophus_INCLUDE_DIRS}
)

add_definitions(${EIGEN_DEFINITIONS})

catkin_package()

add_executable(orb_opencv orb_opencv.cpp)
target_link_libraries(orb_opencv 
                        ${catkin_LIBRARIES} 
                        ${OpenCV_LIBRARIES} 
                        # ${CERES_LIBRARIES}
                        )

add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self 
                        ${catkin_LIBRARIES} 
                        ${OpenCV_LIBRARIES} 
                        # ${CERES_LIBRARIES}
                        )
