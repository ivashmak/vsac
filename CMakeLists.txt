cmake_minimum_required(VERSION 3.0)

project (vsac)

# cache for faster compilation
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# set C++ version
SET (CMAKE_CXX_STANDARD 11)

# add optimization plus warnings
IF(CMAKE_COMPILER_IS_GNUCXX)
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
    # turn off unused function warning
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers -Wno-unused-parameter")
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# not critical if LAPACK is missing
find_package(LAPACK QUIET)
if (LAPACK_FOUND)
    message("LAPACK found.")
    set(lapack_libs ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    add_definitions(-DHAVE_LAPACK)
else (LAPACK_FOUND)
    message("LAPACK not found.")
endif(LAPACK_FOUND)

# not critical if Eigen is missing
find_package(Eigen3 QUIET)
if (EIGEN3_FOUND)
     message("Eigen found.")
     include_directories(${EIGEN3_INCLUDE_DIR})
#     add_definitions(-DHAVE_EIGEN)
else (EIGEN3_FOUND)
    message("Eigen not found.")
endif (EIGEN3_FOUND)

# Check opencv libraries
find_package (OpenCV REQUIRED)
if (OPENCV_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
    include_directories(${OpenCV_INDLUDE_DIRS})
else (OPENCV_FOUND)
    message(FATAL_ERROR "OpenCV not found.")
endif (OPENCV_FOUND)

# Add source files
file(GLOB_RECURSE SRCS_VSAC "src/*.cpp")
file(GLOB_RECURSE SRCS_SAMPLES "samples/*.cpp")

# create libraries
add_library(src ${SRCS_VSAC})
target_link_libraries(src ${OpenCV_LIBS} ${lapack_libs})

add_library(samples ${SRCS_SAMPLES})
target_link_libraries(samples ${OpenCV_LIBS} src)

############# add python bindings ##############
add_definitions(-fPIC)
add_subdirectory(lib/pybind11)
pybind11_add_module(pvsac "python/bindings.cpp")
target_link_libraries(pvsac PRIVATE src)
################################################

#add_executable(${PROJECT_NAME} main.cpp)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} src samples ${lapack_libs})

###################### for test #######################
# in case of problems with link errors on linux with filesystem, uncomment the following line
link_libraries(stdc++fs)

file(GLOB_RECURSE SRCS_TEST "test/*.cpp")
add_library(tests ${SRCS_TEST})
target_link_libraries(tests ${OpenCV_LIBS} src)
add_executable(${PROJECT_NAME} main.cpp ${SRCS_VSAC} ${SRCS_SAMPLES} ${SRCS_EXPS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} src samples tests ${lapack_libs})
#####################################################