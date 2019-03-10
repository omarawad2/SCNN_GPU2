/usr/bin/nvcc -ccbin clang++-3.8 -arch=sm_52  main.cu Layer.cpp cnpy.cpp -o SCNN_GPU
