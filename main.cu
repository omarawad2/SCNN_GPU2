#include "tensor.h"
#include "layer.h"
#include <cmath>

// Constants

/* Number of PE per column */
const int Wt = 8;

/* Number of PE per row */
const int Ht = 8;

/* Column multipliers per PE */
const int I = 4;

/* Row multipliers per PE */
const int F = 4;

std::vector<Layer> read_bvlc_alexnet() {
    std::vector<Layer> network;
    network.emplace_back(Layer("bvlc_alexnet","conv1","conv",true,4,0));
    network.emplace_back(Layer("bvlc_alexnet","conv2","conv",true,1,2));
    network.emplace_back(Layer("bvlc_alexnet","conv3","conv",true,1,1));
    network.emplace_back(Layer("bvlc_alexnet","conv4","conv",true,1,1));
    network.emplace_back(Layer("bvlc_alexnet","conv5","conv",true,1,1));
    network.emplace_back(Layer("bvlc_alexnet","fc6","fc",true,1,0));
    network.emplace_back(Layer("bvlc_alexnet","fc7","fc",true,1,0));
    network.emplace_back(Layer("bvlc_alexnet","fc8","fc",false,1,0));
    return network;
};

inline void check_error(cudaError_t err){
    if(err != cudaSuccess){
        printf("Error: device, %s\n",cudaGetErrorString(err));
        exit(1);
    }
}

template <typename T>
T* host2Dev(uint64_t size, T *h_data){
    T* d_data;
    check_error(cudaMalloc((void**) &d_data, size*sizeof(T)));
    cudaMemcpy(d_data, h_data, size*sizeof(T), cudaMemcpyHostToDevice);
}

//SCNN functions
//naive implmentation
__global__ void kComputePE(int n, int W, int H, int K, int stride, const float* d_act_queue, const int* d_act_queue_x, 
    const int* d_act_queue_y, uint64_t act_queue_size, const float* d_wgt_queue, const int* d_wgt_queue_k,
    const int* d_wgt_queue_r, const int* d_wgt_queue_s, uint64_t wgt_queue_size, float* d_output_activations){
    //TODO: use shared mem.
    //TODO: try different configurations

    int ii = threadIdx.x + blockIdx.x*blockDim.x;
    //int ff = threadIdx.y + blockIdx.y*blockDim.y;

    if(ii < act_queue_size){
        float act = d_act_queue[ii];
        int x = d_act_queue_x[ii];
        int y = d_act_queue_y[ii];
        for(int ff = 0; ff < wgt_queue_size; ff++){
            float wgt = d_wgt_queue[ff];
            float k = d_wgt_queue_k[ff];
            float r = d_wgt_queue_r[ff];
            float s = d_wgt_queue_s[ff];

            //TODO: try to remove div. (takes a lot on GPU)
            int w = (x-r)/stride;
            int h = (y-s)/stride;

             if(w >= 0 && w < W && h >= 0 && h < H) {
                auto pos = n * W * H * K + k * W * H + w * H + h;
                //TODO: memory access not coalesced
                d_output_activations[pos] += act * wgt;
            }
        }
    }
}

void computePE(int n, int W, int K, int stride, const float* h_act_queue,, const int* h_act_queue_x,
        const int* h_act_queue_y, uint64_t act_queue_size, const float* h_wgt_queue, const int* h_wgt_queue_k,
        const int* h_wgt_queue_r, const int* h_wgt_queue_s, uint64_t wgt_queue_size, float* h_output_activations) {

    //TODO: overlap mem. transfer
    float* d_act_queue = host2Dev(act_queue_size, h_act_queue);
    int* d_act_queue_x = host2Dev(act_queue_size, h_act_queue_x);
    int* d_act_queue_y = host2Dev(act_queue_size, h_act_queue_y);
    float* d_wgt_queue = host2Dev(wgt_queue_size, h_wgt_queue);
    int* d_wgt_queue_k = host2Dev(wgt_queue_size, h_wgt_queue_k);
    int* d_wgt_queue_r = host2Dev(wgt_queue_size, h_wgt_queue_r);
    int* d_wgt_queue_s = host2Dev(wgt_queue_size, h_wgt_queue_s);
   
    float* d_output_activations;
    check_error(cudaMalloc((void**) &d_output_activations, size*sizeof(float)));

    dim3 block(1024, 1);
    dim3 grid((act_queue_size+block.x-1)/block.x,1);

    //TODO: add streams
    kComputePE<<<grid,block>>>(n,W,K,stride,d_act_queue,d_act_queue_x,d_act_queue_y,act_queue_size,d_wgt_queue,d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,wgt_queue_size,d_output_activations);

    cudaDeviceSynchronize(); 
}

// MAIN

int main(int argc, char *argv[]) {

    auto network = read_bvlc_alexnet();

    for(auto layer : network) {

        read_layer(layer);

        //TODO: resize for FC layer


        }

return 0;
}