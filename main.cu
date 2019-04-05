
#include "Layer.h"
#include <cuda.h>
#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <string> 
#include<iostream>

#define GLOBAL_TIME
//#define VERBOSE

//sweep params
#define bias_xDim 16
#define bias_yDim 16
#define bias_zDim 4
#define relu_xDim 1024
#define populate_xyDim 4
#define compute_fc_batches 64
#define compute_conv_batches 4
#define compute_xDim 16
#define compute_yDim 64  // compute_yDim < compute_fc_batches
//#define compute_streams_flag
#define compute_streams 8 

struct wgt {
	float value;
	int k, r, s;
};

struct host_data {
	std::vector<wgt*> wgt_queue;
    std::vector<int> wgt_queue_size;
};

struct device_data {
	float *act;

	float *act_queue; 
    int *act_queue_x;
	int *act_queue_y;
    int *act_queue_size;
    
	wgt *wgt_queue;
};

//############################################### Read networks ########################################################

void check_path(const std::string &path) {
	std::ifstream file(path.c_str());
    if(!file.good()) {
    	throw std::runtime_error("The path " + path + " does not exist.");
    }
}

std::vector<Layer> read_trace_params(const std::string network_name) {
    std::vector<Layer> network;

	check_path("/nfs/ug/homes-4/e/edoisak/SCNN_GPU/net_traces/" + network_name);
    std::string path = "/nfs/ug/homes-4/e/edoisak/SCNN_GPU/net_traces/" + network_name + "/trace_params.csv";
    check_path(path);

    std::ifstream myfile (path.c_str());
    if (myfile.is_open()) {

    	std::string line;
        while (getline(myfile,line)) {

        	std::vector<std::string> words;
           	std::string word;
           	std::stringstream ss_line(line);
            while (getline(ss_line,word,','))
            	words.push_back(word);

			// Format: Layer_name, Type, ReLU?, stride, padding
			network.push_back(Layer(network_name,words[0],words[1],(words[2] == "true" ? true : false), 
					atoi(words[3].c_str()),	atoi(words[4].c_str())));

        }
        myfile.close();
    }

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
T* host2Dev(uint64_t size, const T *h_data, std::string task, cudaStream_t stream){
    T* d_data;
    check_error(cudaMalloc((void**) &d_data, size*sizeof(T)),task);
    check_error(cudaMemcpyAsync(d_data, h_data, size*sizeof(T), cudaMemcpyHostToDevice, stream),task);

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
            int offset = X*Y*channel;
            int pos = C*X*Y*n + offset + x*Y + y;
            int tmp_sy = y & (stride-1);
            float act_bits = dev.act[pos];
            if(act_bits !=0 && sx == tmp_sx && sy == tmp_sy){
                int index = (atomicAdd(act_queue_size,1) + offset);
                dev.act_queue[index] = act_bits;
                dev.act_queue_x[index] = x;
                dev.act_queue_y[index] = y;
            }
        }
    }
}

//naive implementation
__global__ void kPopulate_effectual_activations(int n, int channel, int C, int X, int Y, device_data dev, 
		int *act_queue_size) {

    int y = threadIdx.x + blockIdx.x*blockDim.x;
    int x = threadIdx.y + blockIdx.y*blockDim.y;

    if(x < X && y < Y) {
        int offset = X*Y*channel;
    	int pos = C*X*Y*n + offset + x*Y + y;
        float act_bits = dev.act[pos];
        if(act_bits !=0) {
        	int index = (atomicAdd(act_queue_size,1) + offset);
        	dev.act_queue[index] = act_bits;
            dev.act_queue_x[index] = x;
            dev.act_queue_y[index] = y;
        }
    }
}

struct pixel{
    float value;
    int k, r, s, padding;
};


__global__ void kComputePE(int act_ch_offset, int batches, int n_offset, int k_offset, int W, int H, int stride, int *d_act_queue_size, 
        int wgt_queue_size, device_data dev, float *d_output_activations, int offset) {
    //TODO: try different configurations

    __shared__ pixel sh_wgt_queue[1024];

    int x_idx = threadIdx.x + blockIdx.x*blockDim.x*batches;
    int y_idx = threadIdx.y + blockIdx.y*blockDim.y;
    int log_x = (int)__log2f(blockDim.x);
    int g_idx = x_idx + threadIdx.y*blockDim.x;
    int s_idx = threadIdx.x + threadIdx.y*blockDim.x;

    //part before syncthreads can be optimized
        if(g_idx < wgt_queue_size && s_idx < blockDim.x*batches){
            wgt *tmp = (dev.wgt_queue +offset);
            sh_wgt_queue[s_idx].value = tmp[g_idx].value;
            sh_wgt_queue[s_idx].k = tmp[g_idx].k;
            sh_wgt_queue[s_idx].r = tmp[g_idx].r;
            sh_wgt_queue[s_idx].s = tmp[g_idx].s;
        }
    __syncthreads();

    if(y_idx < *d_act_queue_size){  
        float act = (dev.act_queue + act_ch_offset)[y_idx];
        int x =(dev.act_queue_x + act_ch_offset)[y_idx];
        int y = (dev.act_queue_y + act_ch_offset)[y_idx];

        for(int b = 0; b < batches; b++) {
            int hop = b << log_x;
            int sh_idx = threadIdx.x + hop;

            if(x_idx + hop >= wgt_queue_size)
                continue;
            
            float wgt  = sh_wgt_queue[sh_idx].value;
            int k = sh_wgt_queue[sh_idx].k;
            int r = sh_wgt_queue[sh_idx].r;
            int s = sh_wgt_queue[sh_idx].s;

            float mult = act*wgt;
        
            int w = (x-r) >> (int)__log2f(stride);
            int h = (y-s) >> (int)__log2f(stride);

            if(w >= 0 && w < W && h >= 0 && h < H) {
                int pos = n_offset + k * k_offset + w * H + h;
                //TODO: memory access not coalesced
                //TODO: try to remove atomicAdd
                atomicAdd(d_output_activations + pos, mult);
            }
        }
    }
}

__global__ void kComputePE(int act_ch_offset, int batches, int n_offset, int k_offset, int W, int H,  int *d_act_queue_size, 
		int wgt_queue_size, device_data dev, float *d_output_activations, int offset) {
    //TODO: try different configurations

    __shared__ pixel sh_wgt_queue[1024];

    int x_idx = threadIdx.x + blockIdx.x*blockDim.x*batches;
    int y_idx = threadIdx.y + blockIdx.y*blockDim.y;
    int log_x = (int)__log2f(blockDim.x);
    int g_idx = x_idx + threadIdx.y*blockDim.x;
    int s_idx = threadIdx.x + threadIdx.y*blockDim.x;

    //part before syncthreads can be optimized
   		if(g_idx < wgt_queue_size && s_idx < blockDim.x*batches){
            wgt *tmp = (dev.wgt_queue +offset);
	    	sh_wgt_queue[s_idx].value = tmp[g_idx].value;
		    sh_wgt_queue[s_idx].k = tmp[g_idx].k;
		    sh_wgt_queue[s_idx].r = tmp[g_idx].r;
		    sh_wgt_queue[s_idx].s = tmp[g_idx].s;
   		}
    __syncthreads();

    if(y_idx < *d_act_queue_size){	
        float act = (dev.act_queue + act_ch_offset)[y_idx];
        int x =(dev.act_queue_x + act_ch_offset)[y_idx];
        int y = (dev.act_queue_y + act_ch_offset)[y_idx];

		for(int b = 0; b < batches; b++) {
			int hop = b << log_x;
			int sh_idx = threadIdx.x + hop;

			if(x_idx + hop >= wgt_queue_size)
				continue;
	 		
            float wgt  = sh_wgt_queue[sh_idx].value;
            int k = sh_wgt_queue[sh_idx].k;
            int r = sh_wgt_queue[sh_idx].r;
            int s = sh_wgt_queue[sh_idx].s;

            float mult = act*wgt;
        
		    int w = x-r;
		    int h = y-s;

		    if(w >= 0 && w < W && h >= 0 && h < H) {
		        int pos = n_offset + k * k_offset + w * H + h;
		        //TODO: memory access not coalesced
		        //TODO: try to remove atomicAdd
		        atomicAdd(d_output_activations + pos, mult);
		    }
		}
    }
}

//############################################### CPU SCNN #############################################################

void addBias(int N, int K, int W, int H, const Layer &layer, float *d_output_activations, const float *d_bias) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    dim3 block(bias_xDim, bias_yDim, bias_zDim);
    dim3 grid((H+block.x-1)/block.x,(W+block.y-1)/block.y,(K+block.z-1)/block.z);
    //check_grid(grid,"addBias");

    //TODO: we can stream on the channels instead
    for(int n=0; n< N; n++){
        kAddBias<<<grid, block>>>(n,K,W,H,d_bias,d_output_activations);
    }
    //cudaDeviceSynchronize();

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

    dim3 block(relu_xDim, 1);
    dim3 grid((K*W*H+block.x-1)/block.x,1);
    
    //TODO: we can stream on the channels instead
    if(layer.ReLU){
        //check_grid(grid,"relu");
        cudaStream_t streams[N+1];
        for(int n = 0; n < N; n++){
            cudaStreamCreate(&streams[n+1]);
            kRelu<<<grid,block,0,streams[n+1]>>>(N,K,W,H,d_output_activations);
        }
        //cudaDeviceSynchronize();
    }

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kRelu block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kRelu time %.6f\n",(timeStampB-timeStampA));
    #endif
}

void populate_effectual_activations(int n, int channel, int sx, int sy, int stride, const Layer &layer, 
		device_data dev, cudaStream_t stream) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    int C = (int) layer.act_shape[1];
    int X = (int) layer.act_shape[2];
    int Y = (int) layer.act_shape[3];

    int *act_queue_size = dev.act_queue_size+channel;

    dim3 block(populate_xyDim, populate_xyDim);
    dim3 grid((Y+block.x-1)/block.x,(X+block.y-1)/block.y);
    //check_grid(grid,"populate_effectual_activations");

	if(stride != 1)
    	kPopulate_effectual_activations<<<grid,block,0,stream>>>(n,channel,sx,sy,C,X,Y,stride,dev,act_queue_size);
	else
		kPopulate_effectual_activations<<<grid,block,0,stream>>>(n,channel,C,X,Y,dev,act_queue_size);
    //cudaDeviceSynchronize();

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kPopulate_effectual_activations block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kPopulate_effectual_activations time %.6f\n",(timeStampB-timeStampA));
    #endif
}

void computePE(int act_ch_offset, int n, int W, int H, const Layer &layer, int act_queue_size, int wgt_queue_size, int *d_act_queue_size, 
		device_data dev, float *d_output_activations, cudaStream_t stream, int offset) {

    #ifndef GLOBAL_TIME
    double timeStampA = getTimeStamp();
    #endif

    int K = (int) layer.wgt_shape[0];
    int stride = layer.stride;

    int k_offset = W*H;
    int n_offset = n*K*k_offset;
    //TODO can be improved
    int batches = (layer.type == "fc") ? compute_fc_batches : compute_conv_batches;

    //block size might be different for conv and fc
    dim3 block(compute_xDim, compute_yDim);
    int batch_size = block.x*batches;
    dim3 grid((wgt_queue_size+batch_size-1)/batch_size,(act_queue_size+block.y-1)/block.y);
    //check_grid(grid,"computePE");
	
	if(stride != 1)
    	kComputePE<<<grid,block,0,stream>>>(act_ch_offset,batches,n_offset,k_offset,W,H,stride,d_act_queue_size,wgt_queue_size,dev,d_output_activations,offset);
	else
		kComputePE<<<grid,block,0,stream>>>(act_ch_offset,batches,n_offset,k_offset,W,H,d_act_queue_size,wgt_queue_size,dev,d_output_activations,offset);
    //cudaDeviceSynchronize();

    #ifndef GLOBAL_TIME
    double timeStampB = getTimeStamp();
    printf("kComputePE block: (%d,%d,1), grid: (%d,%d,1)\n",block.x,block.y,grid.x,grid.y);
    printf("kComputePE time %.6f\n",(timeStampB-timeStampA));
    #endif

}

void computeTile(int n, int C, int Kc, int X, int Y, int K, int W, int H, int R, int S,
        const Layer &layer, const host_data &hst, device_data dev, float *d_output_activations, int *act_queue_size, cudaStream_t *streams, int nStreams) {

    int stride = layer.stride;

    // Iterate strides
    for(int sx = 0; sx < stride; sx++) {
        for(int sy = 0; sy < stride; sy++) {

            check_error(cudaMemset(dev.act_queue_size,0, C*sizeof(int)),"set activations queue size to zero");
            
            //parallelize across activation channels
            int stream_num = 0; 
            for(int ch = 0; ch < C; ch++) {
                stream_num = ch % nStreams;
                populate_effectual_activations(n,ch,sx,sy,stride,layer,dev,streams[stream_num]);
            }
            cudaDeviceSynchronize();

            check_error(cudaMemcpyAsync(act_queue_size, dev.act_queue_size, C*sizeof(int), cudaMemcpyDeviceToHost,streams[0]),
                "copy activation queue size from device to host");

            
            for(int ch = 0; ch < C; ch++) {
                int pos = ch*stride*stride + sx*stride + sy;
                int wgt_ch_offset = ch*Kc*R*S;
                int act_ch_offset = ch*X*Y;

                stream_num = ch % nStreams;

                check_error(cudaMemcpyAsync(dev.wgt_queue+wgt_ch_offset, hst.wgt_queue[pos], hst.wgt_queue_size[pos]*sizeof(wgt),
                    cudaMemcpyHostToDevice, streams[stream_num]), "copy weights queue from host to device");
                //cudaDeviceSynchronize();

                computePE(act_ch_offset,n,W,H,layer,*(act_queue_size+ch),hst.wgt_queue_size[pos],dev.act_queue_size+ch,dev,
                    d_output_activations,streams[stream_num],wgt_ch_offset);
            }
            cudaDeviceSynchronize();
        }
    }
}

//############################################### Main #################################################################

int main(int argc, char *argv[]) {


	if(argc != 2) {
		printf("Error in number of parameters, usage: %s <network_name>\n",argv[0]);
		return -1;
	}

    double total_time = 0.0;

    int n_streams;

    std::vector<Layer> network = read_trace_params(argv[1]);

/*
    //depending on the network allocate different no. of streams
    std::string net = argv[1];
    int fc_streams = 0, conv_streams = 0;
    if(net == "bvlc_alexnet"){
        fc_streams = 14;
        conv_streams = 1;
    } else if(net == "vgg_cnn_s"){
        fc_streams = 16;
        conv_streams = 2;
    }
*/
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
    					wgt *wgt_queue_ch;

    					cudaMallocHost((void **) &wgt_queue_ch, wgt_queue_max_size * sizeof(wgt));
			            if (wgt_queue_ch == NULL) {
			                fprintf(stderr, "Error: Failed to allocate weights queue!\n");
			                exit(EXIT_FAILURE);
			            }

			            for(int r = 0; r < R; r++) {
			                int tmp_sx = (r + padding) % stride;
			                for(int s = 0; s < S; s++) {
			                    int tmp_sy = (s + padding) % stride;
			                    for(int k = k_begin; k < k_end; k++) {
			                        float wgt_bits = layer.wgt_get(k,ck,r,s);
			                        if (wgt_bits != 0 && sx == tmp_sx && sy == tmp_sy) {
			                            wgt_queue_ch[wgt_queue_size_ch].value = wgt_bits;
			                            wgt_queue_ch[wgt_queue_size_ch].k = k;
			                            wgt_queue_ch[wgt_queue_size_ch].r = r;
			                            wgt_queue_ch[wgt_queue_size_ch].s = s;
			                            wgt_queue_size_ch++;
			                        }
			                    }
			                }
			            }

			            hst.wgt_queue.push_back(wgt_queue_ch);
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

         int *act_queue_size ;
        cudaMallocHost((void **) &act_queue_size, C * sizeof(int));
        if (act_queue_size == NULL) {
            fprintf(stderr, "Error: Failed to allocate activation queue size!\n");
            exit(EXIT_FAILURE);
        }


        #ifdef compute_streams_flag
            //assert(compute_streams < C); 
            n_streams = compute_streams;
        #else
            n_streams = C;
        #endif

    	//double timeStampA = getTimeStamp();

        cudaStream_t streams[3];
        cudaStreamCreate(&streams[0]);
        float *d_bias = host2Dev(layer.getMaxIndex("bias"), layer.bias,"allocate device bias", streams[0]);

        addBias(N, K, W, H, layer, d_output_activations, d_bias);

        ////////core compute/////////////
        // Allocate space for the queues on device (allocate once and reuse)
        float *d_act_queue;
        int *d_act_queue_x, *d_act_queue_y;
        wgt *d_wgt_queue;

        cudaStreamCreate(&streams[1]);
    	float *d_act = host2Dev(layer.getMaxIndex("activations"), layer.activations,"copy device activations",streams[1]);

        //max. size is one activation channel
        check_error(cudaMalloc((void**) &d_act_queue, C*X*Y*sizeof(float)),"allocate device activations queue");
        check_error(cudaMalloc((void**) &d_act_queue_x, C*X*Y*sizeof(int)),"allocate device activations queue X dim");
        check_error(cudaMalloc((void**) &d_act_queue_y, C*X*Y*sizeof(int)),"allocate device activations queue Y dim");

        //max. size is the numebr of kernel channels processed in parallel with each activation channel
        check_error(cudaMalloc((void**) &d_wgt_queue, K*Ck*R*S*sizeof(wgt)),"allocate device weights queue");

        int *d_act_queue_size;
        check_error(cudaMalloc((void**) &d_act_queue_size, C*sizeof(int)),"allocate activations queue size");
        double timeStampA = getTimeStamp();

		//copy to struct
		device_data dev;
		dev.act = d_act;
		dev.act_queue = d_act_queue;
		dev.act_queue_x = d_act_queue_x;
		dev.act_queue_y = d_act_queue_y;
        dev.act_queue_size = d_act_queue_size;
		dev.wgt_queue = d_wgt_queue;

        int nStreams = n_streams;//C;//(layer.type == "fc") ? fc_streams : conv_streams;
        cudaStream_t *computeTile_streams = (cudaStream_t *) malloc(nStreams * sizeof(cudaStream_t));
        for(int i = 0; i < nStreams; i++)
            cudaStreamCreate(&computeTile_streams[i]);		

        for(int n = 0; n < N; n++) {
            //parallelize across different activation channels
            computeTile(n,C,Kc,X,Y,K,W,H,R,S,layer,hst,dev,d_output_activations,act_queue_size,computeTile_streams,nStreams);
        }

        //cudaDeviceSynchronize();

        relu(N, K, W, H, layer, d_output_activations);

        cudaStreamCreate(&streams[2]);
        check_error(cudaMemcpyAsync(h_output_activations, d_output_activations, bytes, cudaMemcpyDeviceToHost, streams[2]),
                "copy output activations from device to host");

		double timeStampB = getTimeStamp();
		printf("Layer %s time: %.6f\n",layer.name.c_str(),timeStampB-timeStampA);
		total_time += timeStampB-timeStampA;


        for(int i = 0; i< nStreams;i++){
        cudaStreamDestroy(computeTile_streams[i]);
        }

        //free GPU resources
        check_error(cudaFree(d_bias),"free device bias");
        check_error(cudaFree(d_act),"free device activations");
        
        check_error(cudaFree(d_act_queue),"free device activations queue");
        check_error(cudaFree(d_act_queue_x),"free device activations queue X dim");
        check_error(cudaFree(d_act_queue_y),"free device activations queue Y dim");
        
        check_error(cudaFree(d_wgt_queue),"free device weights queue");

        check_error(cudaFree(d_act_queue_size),"free device activations size");
        ///////////////////////////

		for(int ck = 0; ck < C; ck++) {
           	for(int sx = 0; sx < stride; sx++) {
        		for(int sy = 0; sy < stride; sy++) {  

       				int pos = ck*stride*stride + sx*stride + sy;

                    cudaFreeHost(hst.wgt_queue[pos]);
        		}
        	}
        }

        check_values(layer,h_output_activations);
        cudaFreeHost(h_output_activations);
        cudaFreeHost(act_queue_size);
    }

	printf("Total time: %.6f\n",total_time);
    cudaDeviceReset();
    return 0;
}
