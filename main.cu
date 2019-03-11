
#include "Layer.h"
#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define GLOBAL_TIME

//############################################### Read networks ########################################################

std::vector<Layer> read_bvlc_alexnet() {
    std::vector<Layer> network;
    network.push_back(Layer("bvlc_alexnet","conv1","conv",true,4,0));
    network.push_back(Layer("bvlc_alexnet","conv2","conv",true,1,2));
    network.push_back(Layer("bvlc_alexnet","conv3","conv",true,1,1));
    network.push_back(Layer("bvlc_alexnet","conv4","conv",true,1,1));
    network.push_back(Layer("bvlc_alexnet","conv5","conv",true,1,1));
    network.push_back(Layer("bvlc_alexnet","fc6","fc",true,1,0));
    network.push_back(Layer("bvlc_alexnet","fc7","fc",true,1,0));
    network.push_back(Layer("bvlc_alexnet","fc8","fc",false,1,0));
    return network;
}

std::vector<Layer> read_vgg_cnn_s() {
    std::vector<Layer> network;
    network.emplace_back(Layer("vgg_cnn_s","conv1","conv",true,2,0));
    network.emplace_back(Layer("vgg_cnn_s","conv2","conv",true,1,0));
    network.emplace_back(Layer("vgg_cnn_s","conv3","conv",true,1,1));
    network.emplace_back(Layer("vgg_cnn_s","conv4","conv",true,1,1));
    network.emplace_back(Layer("vgg_cnn_s","conv5","conv",true,1,1));
    network.emplace_back(Layer("vgg_cnn_s","fc6","fc",true,1,0));
    network.emplace_back(Layer("vgg_cnn_s","fc7","fc",true,1,0));
    network.emplace_back(Layer("vgg_cnn_s","fc8","fc",false,1,0));
    return network;
}

//############################################### Auxiliary functions ##################################################
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

inline
cudaError_t check_error(cudaError_t err, std::string task) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: Failed to %s (error code: %s)!\n", task.c_str(), cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}

void check_grid(dim3 grid, std::string kernel){
    if(grid.x >65535 || grid.y >65535 ||grid.z >65535){
        printf("Kernel:%s...Wrong grid assignment\n",kernel.c_str());
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void print(const T a, int size = 20){
    for(int i=0; i<size;i++)
        printf("%4.4f;",a[i]);
    printf("\n");
}

template <typename T>
T* host2Dev(uint64_t size, const T *h_data, std::string task){
    T* d_data;
    check_error(cudaMalloc((void**) &d_data, size*sizeof(T)),task);
    check_error(cudaMemcpy(d_data, h_data, size*sizeof(T), cudaMemcpyHostToDevice),task);

    return d_data;
}

// Checking function
void check_values(const Layer &layer, const float *output_activations, float min_error = 0.01) {

    #ifdef VERBOSE
    printf("Checking values for layer: %s of type %s\n",layer.name.c_str(),layer.type == "conv" ? "convolution" :
            "fully connected");
    uint32_t count = 0;
    #endif
    for(uint32_t i = 0; i < layer.getMaxIndex("output_activations"); i++) {
        #ifdef VERBOSE
        if(fabsf(output_activations[i] - layer.output_activations[i]) > min_error) count++;
		#else
		assert(fabsf(output_activations[i] - layer.output_activations[i]) <= min_error);
		#endif
    }
	#ifdef VERBOSE
    printf("ERRORS: %u out of %lu with absolute error tolerance of %.2f\n\n",count,
            layer.getMaxIndex("output_activations"), min_error);
	#endif
}

//############################################### CUDA SCNN ############################################################

//naive implementation
__global__ void kAddBias(int n, int K, int W, int H, const float *d_bias, float *d_output_activations){

    int h = threadIdx.x + blockIdx.x*blockDim.x;
    int w = threadIdx.y + blockIdx.y*blockDim.y;
    int k = threadIdx.z + blockIdx.z*blockDim.z;

    //TODO: try different configurations
    if(k < K && w < W && h < H){
        int pos = n*K*W*H + k * W * H + w * H + h;
        d_output_activations[pos] = d_bias[k];
    }
}

__global__ void kRelu(int N, int K, int W, int H, float *d_output_activations){
    
    int x = threadIdx.x + blockIdx.x*blockDim.x;

    if(x < N*K*W*H){
        // Maybe a max with 0 would be faster?
        d_output_activations[x] = (d_output_activations[x] > 0) ? d_output_activations[x] : 0;
        //d_output_activations[x] = std::max(d_output_activations[x],0);
    }
}

//naive implementation
__global__ void kPopulate_effectual_activations(int n, int channel, int sx, int sy, int C, int X, int Y, int stride,
        const float *d_act,float *d_act_queue, int *d_act_queue_x, int *d_act_queue_y, int *act_queue_count) {

    int y = threadIdx.x + blockIdx.x*blockDim.x;
    int x = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < X){
        int tmp_sx = x % stride;
        if(y < Y){
            int tmp_sy = y % stride;
            int pos = C*X*Y*n + X*Y*channel + x*Y + y;
            float act_bits = d_act[pos];
            if(act_bits !=0 && sx == tmp_sx && sy == tmp_sy){
                int index = atomicAdd(act_queue_count,1);
                d_act_queue[index] = act_bits;
                d_act_queue_x[index] = x;
                d_act_queue_y[index] = y;
            }
        }
    }
}

//naive implementation
__global__ void kPopulate_effectual_weights(int Kc, int ck, int sx, int sy,int k_begin, int k_end, int Ck, int R, int S,
        int stride, int padding, const float *d_wgt, float *d_wgt_queue, int *d_wgt_queue_k, int *d_wgt_queue_r,
        int *d_wgt_queue_s, int *wgt_queue_count) {
 
    int k = threadIdx.x + blockIdx.x*blockDim.x + k_begin;
    int s = threadIdx.y + blockIdx.y*blockDim.y;
    int r = threadIdx.z + blockIdx.z*blockDim.z;

    if(r < R){
        int tmp_sx = (r + padding) % stride;
        if(s < S){
            int tmp_sy = (s + padding) % stride;
            if(k < k_end){
                int pos = Ck*R*S*k + R*S*ck + r*S + s;
                float wgt_bits = d_wgt[pos];
                if(wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy){ 
                    int index = atomicAdd(wgt_queue_count,1);          
                    d_wgt_queue[index] = wgt_bits;
                    d_wgt_queue_k[index] = k;
                    d_wgt_queue_r[index] = r;
                    d_wgt_queue_s[index] = s;
                }
            }
        }
    }
}

//naive implmentation
__global__ void kComputePE(int n, int W, int H, int K, int stride, int *act_queue_size, const float *d_act_queue, 
		const int *d_act_queue_x, const int *d_act_queue_y, int *wgt_queue_size, const float *d_wgt_queue, 
		const int *d_wgt_queue_k, const int *d_wgt_queue_r, const int *d_wgt_queue_s, float *d_output_activations) {
    //TODO: use shared mem.
    //TODO: try different configurations

    int ff = threadIdx.x + blockIdx.x*blockDim.x;
    int ii = threadIdx.y + blockIdx.y*blockDim.y;

    if(ii < *act_queue_size && ff < *wgt_queue_size){
        float act = d_act_queue[ii];
        int x = d_act_queue_x[ii];
        int y = d_act_queue_y[ii];

        float wgt = d_wgt_queue[ff];
        int k = d_wgt_queue_k[ff];
        int r = d_wgt_queue_r[ff];
        int s = d_wgt_queue_s[ff];

        //TODO: try to remove div. (takes a lot on GPU)
        int w = (x-r)/stride;
        int h = (y-s)/stride;

        if(w >= 0 && w < W && h >= 0 && h < H) {
            int pos = n * W * H * K + k * W * H + w * H + h;
            //TODO: memory access not coalesced
            //TODO: try to remove atomicAdd
            atomicAdd(d_output_activations + pos, act * wgt);
        }
    }
}

//############################################### CPU SCNN #############################################################

void addBias(int N, int K, int W, int H, const Layer &layer, float *d_output_activations) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    float *d_bias = host2Dev(layer.getMaxIndex("bias"), layer.bias,"allocate device bias");

    dim3 block(16, 16, 4);
    dim3 grid((H+block.x-1)/block.x,(W+block.y-1)/block.y,(K+block.z-1)/block.z);
    check_grid(grid,"addBias");
    for(int n=0; n< N; n++){
        //TODO: add streams
        kAddBias<<<grid, block>>>(n,K,W,H,d_bias,d_output_activations);
    }
    cudaDeviceSynchronize();

    check_error(cudaFree(d_bias),"free device bias");

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kAddBias block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kAddBias time %.6f\n",(timeStampB-timeStampA));
    #endif
}

void relu(int N, int K, int W, int H, const Layer &layer, float *d_output_activations) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    dim3 block(1024, 1);
    dim3 grid((N*K*W*H+block.x-1)/block.x,1);
    
    if(layer.ReLU){
        check_grid(grid,"relu");
        kRelu<<<grid,block>>>(N,K,W,H,d_output_activations);
        cudaDeviceSynchronize();
    }

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kRelu block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kRelu time %.6f\n",(timeStampB-timeStampA));
    #endif
}

void populate_effectual_activations(int n, int channel, int sx, int sy, int stride, const Layer &layer, float *d_act,
        float *d_act_queue, int *d_act_queue_x, int *d_act_queue_y, int *act_queue_count) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    //int N = 1; // Force one image, (int) layer.act_shape[0];
    int C = (int) layer.act_shape[1];
    int X = (int) layer.act_shape[2];
    int Y = (int) layer.act_shape[3];

    dim3 block(32, 32);
    dim3 grid((Y+block.x-1)/block.x,(X+block.y-1)/block.y);
    check_grid(grid,"populate_effectual_activations");

    //TODO:add streams
    kPopulate_effectual_activations<<<grid,block>>>(n,channel,sx,sy,C,X,Y,stride,d_act,d_act_queue,d_act_queue_x,
            d_act_queue_y,act_queue_count);
    cudaDeviceSynchronize();

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kPopulate_effectual_activations block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kPopulate_effectual_activations time %.6f\n",(timeStampB-timeStampA));
    #endif
}

void populate_effectual_weights(int ck, int sx, int sy, int Kc, int k_begin, int k_end, int stride, int padding,
        const Layer &layer, float *d_wgt, float *d_wgt_queue, int *d_wgt_queue_k, int *d_wgt_queue_r, 
		int *d_wgt_queue_s, int *wgt_queue_count) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    //int K = (int) layer.wgt_shape[0];
    int Ck = (int) layer.wgt_shape[1];
    int R = (int) layer.wgt_shape[2];
	int S = (int) layer.wgt_shape[3];

    dim3 block(16, 16, 4);
    dim3 grid((Kc+block.x-1)/block.x,(S+block.y-1)/block.y,(R+block.z-1)/block.z);
    check_grid(grid,"populate_effectual_weights");

    //TODO:add streams
    kPopulate_effectual_weights<<<grid,block>>>(Kc,ck,sx,sy,k_begin,k_end,Ck,R,S,stride,padding,d_wgt,d_wgt_queue,
            d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,wgt_queue_count);
    cudaDeviceSynchronize();

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kPopulate_effectual_weights block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kPopulate_effectual_weights time %.6f\n",(timeStampB-timeStampA));
    #endif
}

void computePE(int n, int W, int H, int K, int stride, int act_queue_size, int wgt_queue_size, const float *d_act_queue,
        const int *d_act_queue_x, const int *d_act_queue_y, int *d_act_queue_size, const float *d_wgt_queue, 
        const int *d_wgt_queue_k, const int *d_wgt_queue_r, const int *d_wgt_queue_s, int *d_wgt_queue_size, 
        float *d_output_activations) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    dim3 block(32, 32);
    dim3 grid((wgt_queue_size+block.x-1)/block.x,(act_queue_size+block.y-1)/block.y);
    check_grid(grid,"computePE");

    kComputePE<<<grid,block>>>(n,W,H,K,stride,d_act_queue_size,d_act_queue,d_act_queue_x,d_act_queue_y,d_wgt_queue_size,
            d_wgt_queue,d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,d_output_activations);
    cudaDeviceSynchronize();

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kComputePE block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kComputePE time %.6f\n",(timeStampB-timeStampA));
    #endif

}

void computeTile(int n, int ct, int ck, int kc, int Kc, int X, int Y, int K, int W, int H, int R, int S,
        const Layer &layer, float *d_output_activations, float *d_act, float *d_act_queue, float *d_wgt_queue, 
		int *d_act_queue_x, int *d_act_queue_y, float *d_wgt, int *d_wgt_queue_k, int *d_wgt_queue_r, 
		int *d_wgt_queue_s) {

    int padding = layer.padding;
    int stride = layer.stride;

    // Iterate PEs

    int k_begin = kc;
    int k_end = k_begin + Kc;

    // Iterate strides
    for(int sx = 0; sx < stride; sx++) {
        for(int sy = 0; sy < stride; sy++) {

            // Populate activations queue
            int act_queue_count = 0;
            int *d_act_queue_count = host2Dev(1,&act_queue_count,"allocate activations queue count");
            populate_effectual_activations(n,ct+ck,sx,sy,stride,layer,d_act,d_act_queue,d_act_queue_x,d_act_queue_y,
                    d_act_queue_count);

            // Populate weights queue
            int wgt_queue_count = 0;
            int *d_wgt_queue_count = host2Dev(1,&act_queue_count,"allocate weights queue count");
            populate_effectual_weights(ck,sx,sy,Kc,k_begin,k_end,stride,padding,layer,d_wgt,d_wgt_queue,d_wgt_queue_k,
                   d_wgt_queue_r,d_wgt_queue_s,d_wgt_queue_count);

            //TODO optimize count usage (computePE needs to read it from mem, and we need to read it from host
            // in order to assign the block size
            check_error(cudaMemcpy(&act_queue_count, d_act_queue_count, sizeof(int), cudaMemcpyDeviceToHost),
                "copy activation queue count from device to host");
            check_error(cudaMemcpy(&wgt_queue_count, d_wgt_queue_count, sizeof(int), cudaMemcpyDeviceToHost),
                "copy weight queue count from device to host");

            //do actual convolution
            computePE(n,W,H,K,stride,act_queue_count,wgt_queue_count,d_act_queue,d_act_queue_x,d_act_queue_y,
                    d_act_queue_count,d_wgt_queue,d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s,d_wgt_queue_count,
                    d_output_activations);

            //free GPU resources
            check_error(cudaFree(d_act_queue_count),"free device activations counter");
            check_error(cudaFree(d_wgt_queue_count),"free device weights counter");
        }
    }
}

//############################################### Main #################################################################

int main(int argc, char *argv[]) {

    double total_time = 0.0;

    //std::vector<Layer> network = read_bvlc_alexnet();
    std::vector<Layer> network = read_vgg_cnn_s();

    for(int i = 0; i < network.size(); i++) {

    	Layer layer = network[i];
    
        layer.read_layer();

        if(layer.type == "fc") {
            layer.reshape_to_2D();
            int C = layer.act_shape[1];
            layer.act_split_4D((unsigned)(C / 256), 16, 16);

            int Ck = layer.wgt_shape[1];
            layer.wgt_split_4D((unsigned)(Ck / 256), 16, 16);
        }

        layer.zero_pad();
        #ifdef FORCE_ONE_IMAGE
        auto N = 1;
        #else
        auto N = (int) layer.act_shape[0];
        #endif
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

        uint32_t bytes = N*K*W*H * sizeof(float);

        float *d_output_activations;
        check_error(cudaMalloc((void **) &d_output_activations, bytes),"allocate device output activations");

    	double timeStampA = getTimeStamp();

        //tested
        addBias(N, K, W, H, layer, d_output_activations);

        ////////core compute/////////////
        // Allocate space for the queues on device (allocate once and reuse)
        float *d_act_queue, *d_wgt_queue;
        int *d_act_queue_x, *d_act_queue_y;
        int *d_wgt_queue_k, *d_wgt_queue_r, *d_wgt_queue_s;

    	float *d_act = host2Dev(layer.getMaxIndex("activations"), layer.activations,"copy device activations");
    	float *d_wgt = host2Dev(layer.getMaxIndex("weights"), layer.weights,"copy device weights channel");

        //max. size is one activation channel
        check_error(cudaMalloc((void**) &d_act_queue, X*Y*sizeof(float)),"allocate device activations queue");
        check_error(cudaMalloc((void**) &d_act_queue_x, X*Y*sizeof(int)),"allocate device activations queue X dim");
        check_error(cudaMalloc((void**) &d_act_queue_y, X*Y*sizeof(int)),"allocate device activations queue Y dim");

        //max. size is the numebr of kernel channels processed in parallel with each activation channel
        check_error(cudaMalloc((void**) &d_wgt_queue, Kc*R*S*sizeof(float)),"allocate device weights queue");
        check_error(cudaMalloc((void**) &d_wgt_queue_k, Kc*R*S*sizeof(int)),"allocate device weights queue K filter");
        check_error(cudaMalloc((void**) &d_wgt_queue_r, Kc*R*S*sizeof(int)),"allocate device weights queue R dim");
        check_error(cudaMalloc((void**) &d_wgt_queue_s, Kc*R*S*sizeof(int)),"allocate device weights queue S dim");

        for(int n = 0; n < N; n++) {
            kc = n;
            for(int ct = 0; ct < C; ct+=Ck) {
                for(int ck = 0; ck < Ck; ck++) {
                    computeTile(n,ct,ck,kc,Kc,X,Y,K,W,H,R,S,layer,d_output_activations,d_act,d_act_queue,d_wgt_queue,
						d_act_queue_x,d_act_queue_y,d_wgt,d_wgt_queue_k,d_wgt_queue_r,d_wgt_queue_s);
                }
                kc += Kc;
            }
        }

        //free GPU resources
    	check_error(cudaFree(d_act),"free device activations");
    	check_error(cudaFree(d_wgt),"free device weights");
        
		check_error(cudaFree(d_act_queue),"free device activations queue");
        check_error(cudaFree(d_act_queue_x),"free device activations queue X dim");
        check_error(cudaFree(d_act_queue_y),"free device activations queue Y dim");
        
		check_error(cudaFree(d_wgt_queue),"free device weights queue");
        check_error(cudaFree(d_wgt_queue_k),"free device weights queue K dim");
        check_error(cudaFree(d_wgt_queue_r),"free device weights queue R dim");
        check_error(cudaFree(d_wgt_queue_s),"free device weights queue S dim");
        ///////////////////////////

        relu(N, K, W, H, layer, d_output_activations);

        float *h_output_activations;
        check_error(cudaMallocHost((void **) &h_output_activations, bytes),"allocate output activations");

        check_error(cudaMemcpy(h_output_activations, d_output_activations, bytes, cudaMemcpyDeviceToHost),
                "copy output activations from device to host");

		double timeStampB = getTimeStamp();
		printf("Layer %s time: %.6f\n",layer.name.c_str(),timeStampB-timeStampA);
		total_time += timeStampB-timeStampA;

        check_values(layer,h_output_activations);
        cudaFreeHost(h_output_activations);

    }

	printf("Total time: %.6f\n",total_time);

    return 0;
}
