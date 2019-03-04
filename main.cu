#include "Layer.h"
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
void print(int size, const T a){
    for(int i=0; i<size;i++)
        printf("%4.4f;",a[i]);
    printf("\n");
}


// Check function
void check_values(const Layer &layer, const float* output_activations, float min_error = 0.01) {

    printf("Checking values for layer: %s of type %s\n",layer.name.c_str(),layer.type == "conv" ? "convolution" :
            "fully connected");
    uint32_t count = 0;
    for(uint32_t i = 0; i < layer.getMaxIndex("output_activations"); i++) {
        if(fabsf(output_activations[i] - layer.output_activations[i]) > min_error) count++;
    }
    printf("ERRORS: %u out of %lu with absolute error tolerance of %.2f\n\n",count,
            layer.getMaxIndex("output_activations"), min_error);
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
    check_error(cudaMalloc((void**) &d_output_activations, N * K * W * H * sizeof(float)));

    dim3 block(1024, 1);
    dim3 grid((act_queue_size+block.x-1)/block.x,1);

    //TODO: add streams
    kComputePE<<<grid,block>>>(n,W,K,stride,d_act_queue,d_act_queue_x,d_act_queue_y,act_queue_size,d_wgt_queue,d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,wgt_queue_size,d_output_activations);

    cudaDeviceSynchronize();

    //copy output activations back to host
    //FIX: no need to copy the whole output activations each time
    cudaMemcpy(h_output_activations, d_output_activations, N * K * W * H * sizeof(float), cudaMemcpyDeviceToHost);

    //free GPU resources
    cudaFree(d_act_queue); 
    cudaFree(d_act_queue_x); 
    cudaFree(d_act_queue_y); 
    cudaFree(d_wgt_queue); 
    cudaFree(d_wgt_queue_k); 
    cudaFree(d_wgt_queue_r); 
    cudaFree(d_wgt_queue_s);
    cudaFree(d_output_activations);
}

//naive implementation
__global__ void kAddBias(int N, int K, int W, int H, float* d_output_activations, const Layer &d_layer){

    int n = threadIdx.x + blockIdx.x*blockDim.x;
    int k = threadIdx.y + blockIdx.y*blockDim.y;

    if(n < N && k < K){
        for(int w =0; w < W; w++){
            for(int h=0; h<H; h++){
                int pos = n * W * H * K + k * W * H + w * H + h;
                d_output_activations[pos] = d_layer.bias[k];
            }
        }
    }
}

void addBias(int N, int K, int W, int H, float* d_output_activations, Layer &d_layer){

    dim3 block(32, 32);
    dim3 grid((N+block.x-1)/block.x,(K+block.y-1)/block.y);
    kAddBias<<<grid, block>>>(N,K,W,H,d_output_activations,d_layer);
    cudaDeviceSynchronize();
}

__global__ void kRelu(int N, int K, int W, const int H, float* d_output_activations){
    
    int x = threadIdx.x + blockIdx.x*blockDim.x;

    if(x < N*K*W*H){
        d_output_activations[x] = (d_output_activations[x] > 0)? d_output_activations[x] : 0;
    }
}

void relu(int N, int K, int W, int H, float* d_output_activations, const Layer &d_layer){
    
    if(d_layer.ReLU){
        dim3 block(1024, 1);
        dim3 grid((N*K*W*H+block.x-1)/block.x,1);
        kRelu<<<grid,block>>>(N,K,W,H,d_output_activations);
        cudaDeviceSynchronize();
    }
}

//naive implementation
__global__ void kCount_effectual_activations(int n, int channels, int sx, int sy, int X, int Y, int stride, const Layer &d_layer, uint64_t &d_queue_count){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < X){
        int tmp_sx = x % stride;
        if(y < Y){
            int tmp_sy = y % stride;
            float act_bits = d_layer.act_get(n,channels,x,y);
            if(act_bits !=0 && sx == tmp_sx && sy == tmp_sy) d_queue_count++;
        }
    }
}

void count_effectual_activations(int n, int channels, int sx, int sy, int X, int Y, int stride, const Layer &d_layer, uint64_t &d_queue_count){
     
     dim3 block(32, 32);
     dim3 grid((X+block.x-1)/block.x,(Y+block.y-1)/block.y);
     //TODO:add streams 
     kCount_effectual<<<grid, block>>>(n,channels,sx,sy,X,Y,stide,d_layer,d_queue_count);
     cudaDeviceSynchronize();
}

//naive implementation
__global__ void kCount_effectual_weights(int ck, int sx, int sy, int R, int S, int k_end, int ck, int stride, int padding, const Layer &d_layer, uint64_t &d_queue_count){
    
    int r = threadIdx.x + blockIdx.x*blockDim.x;
    int s = threadIdx.y + blockIdx.y*blockDim.y;
    int k = threadIdx.z + blockIdx.z*blockDim.z;

    if(r < R){
        int tmp_sx = (r + padding) % stride;
        if(s < S){
            int tmp_sy = (s + padding) % stride;
            if(k < k_end){
                float wgt_bits = d_layer.wgt_get(k,ck,r,s);
                if(wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy) d_queue_count++;
            }
        }
    }
}


void count_effectual_weights(int ck, int sx, int sy, int R, int S, int k_end, int ck, int stride, int padding, const Layer &d_layer, uint64_t &d_queue_count){

    dim3 block(16, 16, 4);
    dim3 grid((R+block.x-1)/block.x,(S+block.y-1)/block.y,(k_end+block.z-1)/block.z);
    //TODO:add streams 
    kCount_effectual_weights<<<grid,block>>>(ck,sx,sy,R,S,k_end,ck,stride,padding,d_layer,d_queue_count);
    cudaDeviceSynchronize();
}

/*
n: nth activations window  
ct: activations channel
ck: weights channel
kc: 
tw, th: not needed
X: activations window x-dimension
Y: activations window y-dimension
K: #weights windows 
W: no of kernel window strides in x-dimension
H: no of kernel window strides in y-dimension
R: weights window x-dimension
S: weights window y-dimension
*/
void computeTile(int n, int ct, int ck, int kc, int Kc, int X, int Y, int K, int W, int H, int R, int S, const Layer &layer, float* h_output_activations){
    int padding = layer.padding;
    int stride = layer.stride;

    //TODO: try different configurations
    //iterate strides
    for(int sx = 0; sx < stride; sx++){
        for(int sy = 0; sy < stride; sy++){
    
            // Count number of effectual activations
            uint64_t act_queue_count = 0;
            count_effectual_activations(n,ct+ck,sx,sy,X,Y,stride,d_layer,act_queue_count);
           
            // Count number of effectual weights
            uint64_t wgt_queue_count = 0;
            count_effectual_weights(ck,sx,sy,R,S,kc+Kc,ck,stride,padding,d_layer,wgt_queue_count);

            // Allocate space for the queues
            


        }
    }



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