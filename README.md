See samples/samples.cpp

Estimation problems:
- Affine matrix (3 points)
- Homography matrix (4 points)
- Fundamental matrix (7 and 8 points)
- Essential matrix (5 points)
- Perspective projection matrix (3 points P3P, linear 6 points)

Install and run:
1) ```mkdir build```
2) ```cmake ..```
3) ```make -j $nproc```
4) ```./vsac```

Install Python bindings:
```bash
python ./setup.py install
```

Libraries:
- OpenCV (Required).
- Eigen (Optional, recommended).
- LAPACK (optional)

Note, to run essential matrix estimation either Eigen or LAPACK has to be installed. \
If your own CMakeLists.txt is used then include `add_definitions(-DHAVE_LAPACK)` and `add_definitions(-DHAVE_EIGEN)` in case any of these libraries is available.

Older VSAC version is intergrated into OpenCV could be run with the following flags:
- USAC_PROSAC 
- USAC_ACCURATE
- USAC_PARALLEL
- USAC_DEFAULT
- USAC_FAST
- USAC_MAGSAC

