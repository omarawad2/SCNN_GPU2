
#include "Layer.h"
#include <cmath>
#include <algorithm>
#include <chrono>

// Constants

/* Number of PE per column */
const int Wt = 8;

/* Number of PE per row */
const int Ht = 8;

/* Column multipliers per PE */
const int I = 4;

/* Row multipliers per PE */
const int F = 4;

//############################################### Read networks ########################################################

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
}

//############################################### Auxiliary functions ##################################################

inline
cudaError_t check_error(cudaError_t err, std::string task) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: Failed to %s (error code: %s)!\n", task.c_str(), cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}


template <typename T>
void print(int size, const T a){
    for(int i=0; i<size;i++)
        printf("%4.4f;",a[i]);
    printf("\n");
}

template <typename T>
T* host2Dev(uint64_t size, const T *h_data, std::string task){
    T* d_data;
    check_error(cudaMalloc((void**) &d_data, size*sizeof(T)),task);
    cudaMemcpy(d_data, h_data, size*sizeof(T), cudaMemcpyHostToDevice);
    return d_data;
}

// Checking function
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

//############################################### CUDA SCNN ############################################################

//naive implementation
__global__ void kAddBias(int N, int K, int W, int H, const float* d_bias, float* d_output_activations){

    int n = threadIdx.x + blockIdx.x*blockDim.x;
    int k = threadIdx.y + blockIdx.y*blockDim.y;

    //TODO: try different configurations
    if(n < N && k < K){
        for(int w =0; w < W; w++){
            for(int h=0; h<H; h++){
                int pos = n * W * H * K + k * W * H + w * H + h;
                d_output_activations[pos] = d_bias[k];
            }
        }
    }
}

__global__ void kRelu(int N, int K, int W, int H, float* d_output_activations){
    
    int x = threadIdx.x + blockIdx.x*blockDim.x;

    if(x < N*K*W*H){
    	// Maybe a max with 0 would be faster?
        //d_output_activations[x] = (d_output_activations[x] > 0) ? d_output_activations[x] : 0;
        d_output_activations[x] = std::max(d_output_activations[x],0); 
    }
}

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
                uint32_t pos = n * W * H * K + k * W * H + w * H + h;
                //TODO: memory access not coalesced
                d_output_activations[pos] += act * wgt;
            }
        }
    }
}

//naive implementation
__global__ void kCount_effectual_activations(int n, int channels, int sx, int sy, int X, int Y, int stride, const Layer &d_layer, uint64_t &d_act_queue_count, 
    float* d_act_queue, int* d_act_queue_x, int* d_act_queue_y, bool populate){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < X){
        int tmp_sx = x % stride;
        if(y < Y){
            int tmp_sy = y % stride;
            float act_bits = d_layer.act_get(n,channels,x,y);
            if(act_bits !=0 && sx == tmp_sx && sy == tmp_sy){
                if(populate){
                    int index = x+y*X;
                    d_act_queue[index] = act_bits;
                    d_act_queue_x[index] = x;
                    d_act_queue_y[index] = y;
                }    
                 else {
                    d_act_queue_count++;
                }
            } 
        }
    }
}

//naive implementation
__global__ void kCount_effectual_weights(int ck, int sx, int sy, int R, int S, int k_end, int ck, int stride, int padding, const Layer &d_layer, uint64_t &d_wgt_queue_count, 
    float* d_wgt_queue, int* d_wgt_queue_k, int* d_wgt_queue_r, int* d_wgt_queue_s, bool populate){
    
    int r = threadIdx.x + blockIdx.x*blockDim.x;
    int s = threadIdx.y + blockIdx.y*blockDim.y;
    int k = threadIdx.z + blockIdx.z*blockDim.z;

    if(r < R){
        int tmp_sx = (r + padding) % stride;
        if(s < S){
            int tmp_sy = (s + padding) % stride;
            if(k < k_end){
                float wgt_bits = d_layer.wgt_get(k,ck,r,s);
                if(wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy){
                    if(populate){
                        int index = r+s*R+k*R*S;
                        d_wgt_queue[index] = wgt_bits;
                        d_wgt_queue_k[index] = k;
                        d_wgt_queue_r[index] = r;
                        d_wgt_queue_s[index] = s;
                    } else {
                        d_queue_count++; 
                    }
                }
            }
        }
    }
}

//############################################### CPU SCNN #############################################################

void addBias(int N, int K, int W, int H, float* d_output_activations, const Layer* d_layer) {

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	auto bytes = d_layer.getMaxIndex("Bias") * sizeof(float);
	float* d_bias = host2Dev(bytes, d_layer.bias,"allocate device bias");

    dim3 block(32, 32);
    dim3 grid((N+block.x-1)/block.x,(K+block.y-1)/block.y);
    printf("kAddBias block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    kAddBias<<<grid, block>>>(N,K,W,H,d_bias,d_output_activations);
    cudaDeviceSynchronize();

    check_error(cudaFree(d_bias),"free device bias");

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("kAddBias time %.6f\n",time_span.count());
}

void relu(int N, int K, int W, int H, float* h_output_activations, const Layer &d_layer) {

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    
    if(d_layer.ReLU){
        dim3 block(1024, 1);
        dim3 grid((N*K*W*H+block.x-1)/block.x,1);
        printf("kRelu block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
        kRelu<<<grid,block>>>(N,K,W,H,d_output_activations);
        cudaDeviceSynchronize();
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("kRelu time %.6f\n",time_span.count());
}

void computePE(int n, int W, int H, int K, int stride, const float* d_act_queue, const int* d_act_queue_x,
        const int* d_act_queue_y, uint64_t act_queue_size, const float* d_wgt_queue, const int* d_wgt_queue_k,
        const int* d_wgt_queue_r, const int* d_wgt_queue_s, uint64_t wgt_queue_size, float* d_output_activations) {
/*
    //TODO: overlap mem. transfer
    float* d_act_queue = host2Dev(act_queue_size, h_act_queue,"allocate device activations queue");
    int* d_act_queue_x = host2Dev(act_queue_size, h_act_queue_x,"allocate device activations queue X dim");
    int* d_act_queue_y = host2Dev(act_queue_size, h_act_queue_y,"allocate device activations queue Y dim");
    float* d_wgt_queue = host2Dev(wgt_queue_size, h_wgt_queue,"allocate device weights queue");
    int* d_wgt_queue_k = host2Dev(wgt_queue_size, h_wgt_queue_k,"allocate device weights queue K filter");
    int* d_wgt_queue_r = host2Dev(wgt_queue_size, h_wgt_queue_r,"allocate device weights queue R dim");
    int* d_wgt_queue_s = host2Dev(wgt_queue_size, h_wgt_queue_s,"allocate device weights queue S dim");
*/
   	// I think that for now we can allocate just once at the begining, since the networks are smaller than 300MB
   	// We can change that in the future though
    //float* d_output_activations;
    //check_error(cudaMalloc((void**) &d_output_activations, N * K * W * H * sizeof(float)),"allocate device output
    //	activations");

    dim3 block(1024, 1);
    dim3 grid((act_queue_size+block.x-1)/block.x,1);

    //TODO: add streams
    kComputePE<<<grid,block>>>(n,W,H,K,stride/2,d_act_queue,d_act_queue_x,d_act_queue_y,act_queue_size,d_wgt_queue,
    	d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,wgt_queue_size,d_output_activations);

    cudaDeviceSynchronize();

    //copy output activations back to host
    //FIX: no need to copy the whole output activations each time
    //cudaMemcpy(h_output_activations, d_output_activations, N * K * W * H * sizeof(float), cudaMemcpyDeviceToHost);
/*
    //free GPU resources
    check_error(cudaFree(d_act_queue),"free device activations queue"); 
    check_error(cudaFree(d_act_queue_x),"free device activations queue X dim"); 
    check_error(cudaFree(d_act_queue_y),"free device activations queue Y dim"); 
    check_error(cudaFree(d_wgt_queue),"free device weights queue"); 
    check_error(cudaFree(d_wgt_queue_k),"free device weights queue K dim"); 
    check_error(cudaFree(d_wgt_queue_r),"free device weights queue R dim"); 
    check_error(cudaFree(d_wgt_queue_s),"free device weights queue S dim");
    //check_error(cudaFree(d_output_activations),"free device output activations");
*/
}


void count_effectual_weights(int ck, int sx, int sy, int R, int S, int k_end, int ck, int stride, int padding, const Layer &d_layer, uint64_t &d_wgt_queue_count, 
    float* d_wgt_queue, int* d_wgt_queue_k, int* d_wgt_queue_r, int* d_wgt_queue_s, bool populate){

    dim3 block(16, 16, 4);
    dim3 grid((R+block.x-1)/block.x,(S+block.y-1)/block.y,(k_end+block.z-1)/block.z);
    //TODO:add streams 
    kCount_effectual_weights<<<grid,block>>>(ck,sx,sy,R,S,k_end,ck,stride,padding,d_layer,d_wgt_queue_count,d_wgt_queue_k,d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,populate);
    cudaDeviceSynchronize();
}

void count_effectual_activations(int n, int channels, int sx, int sy, int X, int Y, int stride, const Layer &d_layer, uint64_t &d_act_queue_count, 
    float* d_act_queue, int* d_act_queue_x, int* d_act_queue_y, bool populate){
     
     dim3 block(32, 32);
     dim3 grid((X+block.x-1)/block.x,(Y+block.y-1)/block.y);
     //TODO:add streams 
     kCount_effectual<<<grid, block>>>(n,channels,sx,sy,X,Y,stide,d_layer,d_act_queue_count, d_act_queue, d_act_queue_x, d_act_queue_y, populate);
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
void computeTile(int n, int ct, int ck, int kc, int Kc, int X, int Y, int K, int W, int H, int R, int S, const Layer &layer, float* d_output_activations){

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    int padding = layer.padding;
    int stride = layer.stride;

    //TODO: try different configurations
    //iterate strides
    for(int sx = 0; sx < stride; sx++){
        for(int sy = 0; sy < stride; sy++){
    
            float* d_act_queue, d_wgt_queue;
            int* d_act_queue_x, d_act_queue_y, d_wgt_queue_k, d_wgt_queue_r, d_wgt_queue_s;

            // Count number of effectual activations
            uint64_t act_queue_count = 0;
            count_effectual_activations(n,ct+ck,sx,sy,X,Y,stride,d_layer,act_queue_count,d_act_queue,d_act_queue_x,d_act_queue_y,false);
           
            // Count number of effectual weights
            uint64_t wgt_queue_count = 0;
            count_effectual_weights(ck,sx,sy,R,S,kc+Kc,ck,stride,padding,d_layer,wgt_queue_count,d_wgt_queue,d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,false);

            // Allocate space for the queues on device
            check_error(cudaMalloc((void**) &d_act_queue, act_queue_count*sizeof(float)),"allocate device activations queue");
            check_error(cudaMalloc((void**) &d_act_queue_x, act_queue_count*sizeof(int)),"allocate device activations queue X dim");
            check_error(cudaMalloc((void**) &d_act_queue_y, act_queue_count*sizeof(int)),"allocate device activations queue Y dim");
            check_error(cudaMalloc((void**) &d_wgt_queue, wgt_queue_count*sizeof(float)),"allocate device weights queue");
            check_error(cudaMalloc((void**) &d_wgt_queue_k, wgt_queue_count*sizeof(int)),"allocate device weights queue K filter");
            check_error(cudaMalloc((void**) &d_wgt_queue_r, wgt_queue_count*sizeof(int)),"allocate device weights queue R dim");
            check_error(cudaMalloc((void**) &d_wgt_queue_s, wgt_queue_count*sizeof(int)),"allocate device weights queue S dim");
            
            // Populate activations queue
            count_effectual_activations(n,ct+ck,sx,sy,X,Y,stride,d_layer,act_queue_count,d_act_queue,d_act_queue_x,d_act_queue_y,true);

            // Populate weights queue
            count_effectual_weights(ck,sx,sy,R,S,kc+Kc,ck,stride,padding,d_layer,wgt_queue_count,d_wgt_queue,d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,true);

            //TODO: add streams
            computePE(n,W,H,K,stride,d_act_queue,d_act_queue_x,d_act_queue_y,act_queue_count,d_wgt_queue, d_wgt_queue_k,
                            d_wgt_queue_r,d_wgt_queue_s, wgt_queue_count, d_output_activations);

            //free GPU resources
            check_error(cudaFree(d_act_queue),"free device activations queue"); 
            check_error(cudaFree(d_act_queue_x),"free device activations queue X dim"); 
            check_error(cudaFree(d_act_queue_y),"free device activations queue Y dim"); 
            check_error(cudaFree(d_wgt_queue),"free device weights queue"); 
            check_error(cudaFree(d_wgt_queue_k),"free device weights queue K dim"); 
            check_error(cudaFree(d_wgt_queue_r),"free device weights queue R dim"); 
            check_error(cudaFree(d_wgt_queue_s),"free device weights queue S dim");
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    printf("computeTile time %.6f\n",time_span.count());
}


//############################################### Main #################################################################

int main(int argc, char *argv[]) {

    auto network = read_bvlc_alexnet();

    for(auto layer : network) {

        layer.read_layer();

        if(layer.type == "fc") {
            layer.reshape_to_2D();
            auto C = layer.act_shape[1];
            layer.act_split_4D((unsigned)(C / 256), 16, 16);

            auto Ck = layer.wgt_shape[1];
            layer.wgt_split_4D((unsigned)(Ck / 256), 16, 16);
        }

  layer.zero_pad();
        int N = 1; // Force one image, (int) layer.act_shape[0];
        int C = (int) layer.act_shape[1];
        int X = (int) layer.act_shape[2];
        int Y = (int) layer.act_shape[3];

        int K = (int) layer.wgt_shape[0];
        int Ck = (int) layer.wgt_shape[1];
        int R = (int) layer.wgt_shape[2];
        int S = (int) layer.wgt_shape[3];

        int stride = layer.stride;

        int W = (X - R)/stride + 1;
        int H = (Y - S)/stride + 1;

        int groups = C / Ck;
        int Kc = K / groups;
        int kc = 0;

        X = (int)(ceil(X/(double)Wt))*Wt;
        Y = (int)(ceil(Y/(double)Ht))*Ht;
        auto tw = X/Wt;
        auto th = Y/Wt;

        layer.grid_zero_pad(X ,Y);

        uint32_t bytes = N*K*W*H * sizeof(float);

        float* d_output_activations;
        check_error(cudaMalloc((void **) &d_output_activations, bytes),"allocate device output activations");

        addBias(N, K, W, H, layer, d_output_activations);

        //core compute
        for(int n = 0; n < N; n++) {
            for(int ct = 0; ct < C; ct+=Ck) {
                for(int ck = 0; ck < Ck; ck++) {
                    computeTile(n,ct,ck,kc,Kc,X,Y,K,W,H,R,S,layer,d_output_activations);
                }
                kc += Kc;
            }
        }

        relu(N, K, W, H, layer, d_output_activations);

        auto h_output_activations = (float *) malloc(bytes);
        if (h_output_activations == nullptr) {
            fprintf(stderr, "Error: Failed to allocate output activations!\n");
            exit(EXIT_FAILURE);
        }

        check_error(cudaMemcpy(h_output_activations, d_output_activations, bytes, cudaMemcpyDeviceToHost),
        		"copy output activations from device to host");

        check_values(layer,h_output_activations);
        free(h_output_activations);

        }

		return 0;
}