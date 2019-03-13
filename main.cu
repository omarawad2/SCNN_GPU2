
#include "Layer.h"
#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#define GLOBAL_TIME
//#define VERBOSE

struct host_data {
	std::vector<float*> wgt_queue;
	std::vector<int*> wgt_queue_k;
    std::vector<int*> wgt_queue_r;
    std::vector<int*> wgt_queue_s;
    std::vector<int> wgt_queue_size;
};

struct device_data {
	float *act;

	float *act_queue; 
    int *act_queue_x;
	int *act_queue_y;
    
	float *wgt_queue;
    int *wgt_queue_k;
	int *wgt_queue_r;
	int *wgt_queue_s;
};

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
    network.push_back(Layer("vgg_cnn_s","conv1","conv",true,2,0));
    network.push_back(Layer("vgg_cnn_s","conv2","conv",true,1,0));
    network.push_back(Layer("vgg_cnn_s","conv3","conv",true,1,1));
    network.push_back(Layer("vgg_cnn_s","conv4","conv",true,1,1));
    network.push_back(Layer("vgg_cnn_s","conv5","conv",true,1,1));
    network.push_back(Layer("vgg_cnn_s","fc6","fc",true,1,0));
    network.push_back(Layer("vgg_cnn_s","fc7","fc",true,1,0));
    network.push_back(Layer("vgg_cnn_s","fc8","fc",false,1,0));
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
        d_output_activations[x] = fmaxf(d_output_activations[x],0);
    }
}

//naive implementation
__global__ void kPopulate_effectual_activations(int n, int channel, int sx, int sy, int C, int X, int Y, int stride,
        device_data dev, int *act_queue_size) {

    int y = threadIdx.x + blockIdx.x*blockDim.x;
    int x = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < X){
        int tmp_sx = x & (stride-1);
        if(y < Y){
            int pos = C*X*Y*n + X*Y*channel + x*Y + y;
            int tmp_sy = y & (stride-1);
            float act_bits = dev.act[pos];
            if(act_bits !=0 && sx == tmp_sx && sy == tmp_sy){
                int index = atomicAdd(act_queue_size,1);
                dev.act_queue[index] = act_bits;
                dev.act_queue_x[index] = x;
                dev.act_queue_y[index] = y;
            }
        }
    }
}

static __device__ __forceinline__ unsigned int log2(unsigned int a){
    return (a) ? (__float_as_int(__uint2float_rz(a)) >> 23) - 127 : 0;
}

//naive implmentation
__global__ void kComputePE(int n, int W, int H, int K, unsigned int stride, int *act_queue_size, int wgt_queue_size, 
		float *d_wgt_queue, int *d_wgt_queue_k, int *d_wgt_queue_r, int *d_wgt_queue_s, int size_eff, device_data dev, float *d_output_activations) {
    //TODO: use shared mem.
    //TODO: try different configurations
    /*__shared__ float s_wgt_queue[10];
    __shared__ int s_wgt_queue_k[10];
    __shared__ int s_wgt_queue_r[10];
    __shared__ int s_wgt_queue_s[10];*/

    int ff = threadIdx.x + blockIdx.x*blockDim.x;// + offset;
    int ii = threadIdx.y + blockIdx.y*blockDim.y;
/*
    if(ff < size_eff && ff+offset < wgt_queue_size){
        s_wgt_queue[ff] = dev.wgt_queue[ff+offset];
        s_wgt_queue_k[ff] = dev.wgt_queue_k[ff+offset];
        s_wgt_queue_r[ff] = dev.wgt_queue_r[ff+offset];
        s_wgt_queue_s[ff] = dev.wgt_queue_s[ff+offset];
    }   
    __syncthreads();
*/
    if(ii < *act_queue_size && ff < size_eff){
        float act = dev.act_queue[ii];
        int x = dev.act_queue_x[ii];
        int y = dev.act_queue_y[ii];

        float wgt = d_wgt_queue[ff];//dev.wgt_queue[ff];
        int k = d_wgt_queue_k[ff];//dev.wgt_queue_k[ff];
        int r = d_wgt_queue_r[ff];//dev.wgt_queue_r[ff];
        int s = d_wgt_queue_s[ff];//dev.wgt_queue_s[ff];

        //works for power of 2 strides
        int w = (x-r) >> log2(stride);
        int h = (y-s) >> log2(stride);

        if(w >= 0 && w < W && h >= 0 && h < H) {
            int pos = n * W * H * K + k * W * H + w * H + h;
            //TODO: memory access not coalesced
            //TODO: try to remove atomicAdd
            //d_output_activations[pos] += act * wgt;
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

    cudaStream_t streams[N+1];

    for(int n=0; n< N; n++){
        cudaStreamCreate(&streams[n+1]);
        kAddBias<<<grid, block,0,streams[n+1]>>>(n,K,W,H,d_bias,d_output_activations);
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
    dim3 grid((K*W*H+block.x-1)/block.x,1);
    
    if(layer.ReLU){
        check_grid(grid,"relu");
        cudaStream_t streams[N+1];
        for(int n = 0; n < N; n++){
            cudaStreamCreate(&streams[n+1]);
            kRelu<<<grid,block,0,streams[n+1]>>>(N,K,W,H,d_output_activations);
        }
        cudaDeviceSynchronize();
    }

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kRelu block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kRelu time %.6f\n",(timeStampB-timeStampA));
    #endif
}

void populate_effectual_activations(int n, int channel, int sx, int sy, int stride, const Layer &layer, 
		device_data dev, int *act_queue_size) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    int C = (int) layer.act_shape[1];
    int X = (int) layer.act_shape[2];
    int Y = (int) layer.act_shape[3];

    dim3 block(32, 32);
    dim3 grid((Y+block.x-1)/block.x,(X+block.y-1)/block.y);
    check_grid(grid,"populate_effectual_activations");

    //TODO:add streams
    kPopulate_effectual_activations<<<grid,block>>>(n,channel,sx,sy,C,X,Y,stride,dev,act_queue_size);
    cudaDeviceSynchronize();

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kPopulate_effectual_activations block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kPopulate_effectual_activations time %.6f\n",(timeStampB-timeStampA));
    #endif
}

void computePE(int n, int W, int H, int K, int stride, int act_queue_size, int wgt_queue_size, int *d_act_queue_size, 
		device_data dev, int size_eff, int offset, float *d_output_activations, cudaStream_t stream) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    //block size might be different for conv and fc
    dim3 block(128, 8);
    dim3 grid((size_eff+block.x-1)/block.x,(act_queue_size+block.y-1)/block.y);
    check_grid(grid,"computePE");

    kComputePE<<<grid,block,0,stream>>>(n,W,H,K,stride,d_act_queue_size,wgt_queue_size,dev.wgt_queue+offset,dev.wgt_queue_k+offset,
        dev.wgt_queue_r+offset,dev.wgt_queue_s+offset,size_eff,dev,d_output_activations);
    //cudaDeviceSynchronize();

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kComputePE block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kComputePE time %.6f\n",(timeStampB-timeStampA));
    #endif

}

void computeTile(int n, int ct, int ck, int kc, int Kc, int X, int Y, int K, int W, int H, int R, int S,
        const Layer &layer, const host_data &hst, device_data dev, float *d_output_activations) {

    int stride = layer.stride;

    // Iterate strides
    for(int sx = 0; sx < stride; sx++) {
        for(int sy = 0; sy < stride; sy++) {

			// Transfer working weights to GPU
        	int pos = (ct+ck)*stride*stride + sx*stride + sy;
        	/*check_error(cudaMemcpy(dev.wgt_queue, hst.wgt_queue[pos], hst.wgt_queue_size[pos]*sizeof(float), cudaMemcpyHostToDevice),
           		"copy weights queue from host to device");
        	check_error(cudaMemcpy(dev.wgt_queue_k, hst.wgt_queue_k[pos], hst.wgt_queue_size[pos]*sizeof(int), cudaMemcpyHostToDevice),
           		"copy weights queue from host to device");
			check_error(cudaMemcpy(dev.wgt_queue_r, hst.wgt_queue_r[pos], hst.wgt_queue_size[pos]*sizeof(int), cudaMemcpyHostToDevice),
           		"copy weights queue from host to device");
        	check_error(cudaMemcpy(dev.wgt_queue_s, hst.wgt_queue_s[pos], hst.wgt_queue_size[pos]*sizeof(int), cudaMemcpyHostToDevice),
           		"copy weights queue from host to device");*/

            int *d_act_queue_size;
    		check_error(cudaMalloc((void**) &d_act_queue_size, sizeof(int)),"allocate activations queue size");
    		check_error(cudaMemset(d_act_queue_size,0, sizeof(int)),"set activations queue size to zero");

            // Populate activations queue
            populate_effectual_activations(n,ct+ck,sx,sy,stride,layer,dev,d_act_queue_size);

            //TODO optimize size usage (computePE needs to read it from mem, and we need to read it from host
            // in order to assign the block size
            int act_queue_size;
            check_error(cudaMemcpy(&act_queue_size, d_act_queue_size, sizeof(int), cudaMemcpyDeviceToHost),
                "copy activation queue size from device to host");
            
            int streamSize = 20000;
            int nStreams = (hst.wgt_queue_size[pos]+streamSize-1)/streamSize;
            cudaStream_t streams[nStreams+1];
            int offset = 0, size_eff = 0;

            for(int i = 0; i< nStreams;i++){
                offset = i*streamSize;
                size_eff = (offset+streamSize > hst.wgt_queue_size[pos])? hst.wgt_queue_size[pos]-offset : streamSize;
                cudaStreamCreate(&streams[i+1]);

                check_error(cudaMemcpyAsync(dev.wgt_queue+offset, hst.wgt_queue[pos]+offset, size_eff*sizeof(float), cudaMemcpyHostToDevice, streams[i+1]),
                    "copy weights queue from host to device");
                check_error(cudaMemcpyAsync(dev.wgt_queue_k+offset, hst.wgt_queue_k[pos]+offset, size_eff*sizeof(int), cudaMemcpyHostToDevice, streams[i+1]),
                    "copy weights queue from host to device");
                check_error(cudaMemcpyAsync(dev.wgt_queue_r+offset, hst.wgt_queue_r[pos]+offset, size_eff*sizeof(int), cudaMemcpyHostToDevice, streams[i+1]),
                    "copy weights queue from host to device");
                check_error(cudaMemcpyAsync(dev.wgt_queue_s+offset, hst.wgt_queue_s[pos]+offset, size_eff*sizeof(int), cudaMemcpyHostToDevice, streams[i+1]),
                    "copy weights queue from host to device");
                //do actual convolution
                //can be done better (remove dev and pass the wgt tiles only ?)
                //computePE(n,W,H,K,stride,act_queue_size,hst.wgt_queue_size[pos],d_act_queue_size,dev,size_eff,offset,d_output_activations,streams[i+1]);
                computePE(n,W,H,K,stride,act_queue_size,hst.wgt_queue_size[pos],d_act_queue_size,dev,size_eff,offset,d_output_activations,streams[i+1]);
            }

            //cudaDeviceSynchronize();

            //do actual convolution
            //computePE(n,W,H,K,stride,act_queue_size,hst.wgt_queue_size[pos],d_act_queue_size,dev,d_output_activations);

            //free GPU resources
            check_error(cudaFree(d_act_queue_size),"free device activations size");

        }
    }
}

//############################################### Main #################################################################

int main(int argc, char *argv[]) {

    double total_time = 0.0;

    std::vector<Layer> network = read_bvlc_alexnet();
    //std::vector<Layer> network = read_vgg_cnn_s();

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
        int N = 1;
        #else
        int N = (int) layer.act_shape[0];
        #endif
        int C = (int) layer.act_shape[1];
        int X = (int) layer.act_shape[2];
        int Y = (int) layer.act_shape[3];

        int K = (int) layer.wgt_shape[0];
        int Ck = (int) layer.wgt_shape[1];
        int R = (int) layer.wgt_shape[2];
        int S = (int) layer.wgt_shape[3];

		int padding = layer.padding;
        int stride = layer.stride;

        int W = (X - R)/stride + 1;
        int H = (Y - S)/stride + 1;

        int groups = C / Ck;
        int Kc = K / groups;
        int kc = 0;

        // Allocate compressed weights off-line
		host_data hst;

        for(int ct = 0; ct < C; ct+=Ck) {
            for(int ck = 0; ck < Ck; ck++) {
            	for(int sx = 0; sx < stride; sx++) {
        			for(int sy = 0; sy < stride; sy++) {   

        				int wgt_queue_max_size = R*S*Kc;

        				int k_begin = kc;
    					int k_end = k_begin + Kc;

        			    int wgt_queue_size_ch = 0;
    					float *wgt_queue_ch;
						int *wgt_queue_k_ch, *wgt_queue_r_ch, *wgt_queue_s_ch;

    					cudaMallocHost((void **) &wgt_queue_ch, wgt_queue_max_size * sizeof(float));
			            if (wgt_queue_ch == NULL) {
			                fprintf(stderr, "Error: Failed to allocate weights queue!\n");
			                exit(EXIT_FAILURE);
			            }
    					cudaMallocHost((void **) &wgt_queue_k_ch, wgt_queue_max_size * sizeof(int));
			            if (wgt_queue_k_ch == NULL) {
			                fprintf(stderr, "Error: Failed to allocate weights queue k!\n");
			                exit(EXIT_FAILURE);
			            }
    					cudaMallocHost((void **) &wgt_queue_r_ch, wgt_queue_max_size * sizeof(int));
			            if (wgt_queue_r_ch == NULL) {
			                fprintf(stderr, "Error: Failed to allocate weights queue r!\n");
			                exit(EXIT_FAILURE);
			            }
    					cudaMallocHost((void **) &wgt_queue_s_ch, wgt_queue_max_size * sizeof(int));
			            if (wgt_queue_s_ch == NULL) {
			                fprintf(stderr, "Error: Failed to allocate weights queue s!\n");
			                exit(EXIT_FAILURE);
			            }

			            for(int r = 0; r < R; r++) {
			                int tmp_sx = (r + padding) % stride;
			                for(int s = 0; s < S; s++) {
			                    int tmp_sy = (s + padding) % stride;
			                    for(int k = k_begin; k < k_end; k++) {
			                        float wgt_bits = layer.wgt_get(k,ck,r,s);
			                        if (wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy) {
			                            wgt_queue_ch[wgt_queue_size_ch] = wgt_bits;
			                            wgt_queue_k_ch[wgt_queue_size_ch] = k;
			                            wgt_queue_r_ch[wgt_queue_size_ch] = r;
			                            wgt_queue_s_ch[wgt_queue_size_ch] = s;
			                            wgt_queue_size_ch++;
			                        }
			                    }
			                }
			            }

			            hst.wgt_queue.push_back(wgt_queue_ch);
			            hst.wgt_queue_k.push_back(wgt_queue_k_ch);
			            hst.wgt_queue_r.push_back(wgt_queue_r_ch);
			            hst.wgt_queue_s.push_back(wgt_queue_s_ch);
			            hst.wgt_queue_size.push_back(wgt_queue_size_ch);

        			}
    			}
            }
            kc += Kc;
        }

        uint32_t bytes = N*K*W*H * sizeof(float);

        float *d_output_activations;
        check_error(cudaMalloc((void **) &d_output_activations, bytes),"allocate device output activations");

        float *h_output_activations;
        check_error(cudaMallocHost((void **) &h_output_activations, bytes),"allocate output activations");

    	double timeStampA = getTimeStamp();

        addBias(N, K, W, H, layer, d_output_activations);

        ////////core compute/////////////
        // Allocate space for the queues on device (allocate once and reuse)
        float *d_act_queue, *d_wgt_queue;
        int *d_act_queue_x, *d_act_queue_y;
        int *d_wgt_queue_k, *d_wgt_queue_r, *d_wgt_queue_s;

    	float *d_act = host2Dev(layer.getMaxIndex("activations"), layer.activations,"copy device activations");

        //max. size is one activation channel
        check_error(cudaMalloc((void**) &d_act_queue, X*Y*sizeof(float)),"allocate device activations queue");
        check_error(cudaMalloc((void**) &d_act_queue_x, X*Y*sizeof(int)),"allocate device activations queue X dim");
        check_error(cudaMalloc((void**) &d_act_queue_y, X*Y*sizeof(int)),"allocate device activations queue Y dim");

        //max. size is the numebr of kernel channels processed in parallel with each activation channel
        check_error(cudaMalloc((void**) &d_wgt_queue, Kc*R*S*sizeof(float)),"allocate device weights queue");
        check_error(cudaMalloc((void**) &d_wgt_queue_k, Kc*R*S*sizeof(int)),"allocate device weights queue K filter");
        check_error(cudaMalloc((void**) &d_wgt_queue_r, Kc*R*S*sizeof(int)),"allocate device weights queue R dim");
        check_error(cudaMalloc((void**) &d_wgt_queue_s, Kc*R*S*sizeof(int)),"allocate device weights queue S dim");

		//copy to struct
		device_data dev;
		dev.act = d_act;
		dev.act_queue = d_act_queue;
		dev.act_queue_x = d_act_queue_x;
		dev.act_queue_y = d_act_queue_y;
		dev.wgt_queue = d_wgt_queue;
		dev.wgt_queue_k = d_wgt_queue_k;
		dev.wgt_queue_r = d_wgt_queue_r;
		dev.wgt_queue_s = d_wgt_queue_s;		

        for(int n = 0; n < N; n++) {
            kc = n;
            for(int ct = 0; ct < C; ct+=Ck) {
                for(int ck = 0; ck < Ck; ck++) {
                    computeTile(n,ct,ck,kc,Kc,X,Y,K,W,H,R,S,layer,hst,dev,d_output_activations);
                }
                kc += Kc;
            }
        }

        relu(N, K, W, H, layer, d_output_activations);

        check_error(cudaMemcpy(h_output_activations, d_output_activations, bytes, cudaMemcpyDeviceToHost),
                "copy output activations from device to host");

		double timeStampB = getTimeStamp();
		printf("Layer %s time: %.6f\n",layer.name.c_str(),timeStampB-timeStampA);
		total_time += timeStampB-timeStampA;

        //free GPU resources
        check_error(cudaFree(d_act),"free device activations");
        
        check_error(cudaFree(d_act_queue),"free device activations queue");
        check_error(cudaFree(d_act_queue_x),"free device activations queue X dim");
        check_error(cudaFree(d_act_queue_y),"free device activations queue Y dim");
        
        check_error(cudaFree(d_wgt_queue),"free device weights queue");
        check_error(cudaFree(d_wgt_queue_k),"free device weights queue K dim");
        check_error(cudaFree(d_wgt_queue_r),"free device weights queue R dim");
        check_error(cudaFree(d_wgt_queue_s),"free device weights queue S dim");
        ///////////////////////////

		for(int ck = 0; ck < C; ck++) {
           	for(int sx = 0; sx < stride; sx++) {
        		for(int sy = 0; sy < stride; sy++) {  

       				int pos = ck*stride*stride + sx*stride + sy;

                    cudaFreeHost(hst.wgt_queue[pos]);
		            cudaFreeHost(hst.wgt_queue_k[pos]);
		            cudaFreeHost(hst.wgt_queue_r[pos]);
		            cudaFreeHost(hst.wgt_queue_s[pos]);
        		}
        	}
        }

        check_values(layer,h_output_activations);
        cudaFreeHost(h_output_activations);

    }

	printf("Total time: %.6f\n",total_time);

    return 0;
}
