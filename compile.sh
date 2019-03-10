/usr/bin/nvcc -ccbin clang++-3.8 -arch=sm_52  main.cu gpu_Layer.cu cnpy.cpp -o SCNN_GPU
