# SCNN_GPU

### Compilation:
Command line compilation. First we need to configure the project:
    
    cmake -H. -Bcmake-build-release -DCMAKE_BUILD_TYPE=Release

Then, we can proceed to build the project

    cmake --build cmake-build-release/ --target all
