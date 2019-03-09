################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
/home/omar/UofT_files/courseWork/GPU/project/SCNN_GPU/main.cu 

CPP_SRCS += \
/home/omar/UofT_files/courseWork/GPU/project/SCNN_GPU/Layer.cpp \
/home/omar/UofT_files/courseWork/GPU/project/SCNN_GPU/cnpy.cpp 

OBJS += \
./Layer.o \
./cnpy.o \
./main.o 

CU_DEPS += \
./main.d 

CPP_DEPS += \
./Layer.d \
./cnpy.d 


# Each subdirectory must supply rules for building sources it contributes
Layer.o: /home/omar/UofT_files/courseWork/GPU/project/SCNN_GPU/Layer.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

cnpy.o: /home/omar/UofT_files/courseWork/GPU/project/SCNN_GPU/cnpy.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

main.o: /home/omar/UofT_files/courseWork/GPU/project/SCNN_GPU/main.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


