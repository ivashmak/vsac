Solvers for:
- Affine matrix (3 points)
- Homography matrix (4 points)
- Fundamental matrix (7 and 8 points)
- Essential matrix (5 points)
- Perspective projection matrix (3 points P3P, linear 6 points)

Install and run: 
1) ```git submodule update --init```
2) ```mkdir build && cd build```
3) ```cmake ..```
4) ```make -j $(nproc)```
5) ```./vsac```

Install Python bindings:
```bash
python3 ./setup.py install
```
Run Python example:
```bash
python3 python/example.py 
```

Libraries:
- OpenCV (required)
- Eigen (optional, recommended)
- LAPACK (optional)

Note, to run essential matrix estimation either Eigen or LAPACK has to be installed. \
If your own CMakeLists.txt is used then include `add_definitions(-DHAVE_LAPACK)` and `add_definitions(-DHAVE_EIGEN)` in case any of these libraries is available.
Under `lib` directory there are `Eigen` headers that are included in case Eigen library is not installed, otherwise this directory can be deleted.

File with example code is `samples/samples.cpp`. For Python it is `python/example.py`.
The VSAC framework enables to add new methods and solvers. To do it you need:
1) Implement your new method as a derived class.
2) Add a name of your method in the `inlude/vsac.hpp` enum. 
3) Expand if condition in the `src/init.cpp` to choose your method.

Older VSAC version is integrated into OpenCV could be run with the following flags:
- USAC_PROSAC 
- USAC_ACCURATE
- USAC_PARALLEL
- USAC_DEFAULT
- USAC_FAST
- USAC_MAGSAC

