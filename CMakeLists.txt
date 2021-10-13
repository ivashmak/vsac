cmake_minimum_required(VERSION 3.0)

project (vsac)

# cache for faster compilation
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

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
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS} -O3")
    include_directories(${OpenCV_INDLUDE_DIRS})
else (OPENCV_FOUND)
    message(FATAL_ERROR "OpenCV not found.")
endif (OPENCV_FOUND)

# Add source files
file(GLOB_RECURSE SRCS_VSAC "src/*.cpp")
file(GLOB_RECURSE SRCS_SAMPLES "samples/*.cpp")

# create libraries
add_library(lvsac SHARED ${SRCS_VSAC})
target_link_libraries(lvsac ${OpenCV_LIBS} ${lapack_libs})

add_library(samples ${SRCS_SAMPLES})
target_link_libraries(samples ${OpenCV_LIBS} lvsac)

############# add python bindings ##############
add_definitions(-fPIC)
add_subdirectory(lib/pybind11)
pybind11_add_module(pvsac "python/bindings.cpp")
target_link_libraries(pvsac PRIVATE lvsac)
################################################

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} lvsac samples ${lapack_libs})

#################### install #####################
if (INSTALL)
    message("Install VSAC")
    # if (NOT DEFINED ${CMAKE_INSTALL_LIBDIR})
    #    message("Set CMAKE_INSTALL_PREFIX to current directory")
    #    SET(CMAKE_INSTALL_LIBDIR ".")
    # endif()

    #https://stackoverflow.com/questions/17511496/how-to-create-a-shared-library-with-cmake
    include(GNUInstallDirs)
    set_target_properties(lvsac PROPERTIES VERSION 1.0 SOVERSION 1 PUBLIC_HEADER include/vsac.hpp)
    configure_file(vsac.pc.in vsac.pc @ONLY)
    target_include_directories(lvsac PRIVATE .)
    install(TARGETS lvsac LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${CMAKE_BINARY_DIR}/vsac.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
endif(INSTALL)
###################################################
