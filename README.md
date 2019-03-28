# SCNN_GPU

### CPU code compilation:
Command line compilation. First we need to configure the project:
    
    cmake -H. -Bcmake-build-release -DCMAKE_BUILD_TYPE=Release

Then, we can proceed to build the project

    cmake --build cmake-build-release/ --target all

Execute

	./cmake-build-release/bin/SCNN_GPU

### GPU code compilation:
Run script:

	./compile.sh

Execute

	./SCNN-GPU

### Reference papers:
https://arxiv.org/pdf/1801.02108.pdf

