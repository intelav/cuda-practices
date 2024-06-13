/*
 * This example shows how to use the clock function to measure the performance
 * of block of threads of a kernel accurately. Blocks are executed in parallel
 * and out of order. Since there's no synchronization mechanism between blocks,
 * we measure the clock once for each block. The clock samples are written to
 * device memory.
 */

// System includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <nvrtc_helper.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>

#define NUM_BLOCKS 64

#define NUM_THREADS 256

// It's interesting to change the number of blocks and the number of threads to
// understand how to keep the hardware busy.
//

// Here are some numbers I get on my G80:
//    blocks - clocks
//    1 - 3096
//    8 - 3232
//    16 - 3364
//    32 - 4615
//    64 - 9981clear

//
// With less than 16 blocks some of the multiprocessors of the device are idle.
// With
// more than 16 you are using all the multiprocessors, but there's only one
// block per
// multiprocessor and that doesn't allow you to hide the latency of the memory.
// With
// more than 32 the speed scales linearly.

// Start the main CUDA Sample here

int main(int argc, char **argv) {
  printf("CUDA Clock sample\n");

  typedef long clock_t;

  clock_t timer[NUM_BLOCKS * 2];

  float input[NUM_THREADS * 2];

  for (int i = 0; i < NUM_THREADS * 2; i++) {
    input[i] = (float)i;
  }

  char *cubin, *kernel_file;
  size_t cubinSize;
   if (argc < 2){
    printf("number of argument are invalid\n");
    return 0;
   } 
  if(strcmp(argv[1],"x")){
    kernel_file = sdkFindFilePath("clock_kernel_x.cu", argv[0]);  
  } else if (strcmp(argv[1],"y")){
    kernel_file = sdkFindFilePath("clock_kernel_y.cu", argv[0]);
  } 
  kernel_file = sdkFindFilePath("clock_kernel_y.cu", argv[0]);
  compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubinSize, 0);
   printf("size of cubin =%ld,cubinaddr=%p\n",cubinSize,cubin) ;
  CUmodule module = loadCUBIN(cubin, argc, argv);
  CUfunction kernel_addr;

  checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "timedReduction"));
  printf("kernel addr=%p\n",kernel_addr);
  
  dim3 cudaBlockSize_x(NUM_THREADS,1, 1);
  dim3 cudaGridSize_x(NUM_BLOCKS,1, 1);

  dim3 cudaBlockSize_y(1,NUM_THREADS, 1);
  dim3 cudaGridSize_y(1,NUM_BLOCKS, 1);
  CUdeviceptr dinput, doutput, dtimer;
  checkCudaErrors(cuMemAlloc(&dinput, sizeof(float) * NUM_THREADS * 2));
  checkCudaErrors(cuMemAlloc(&doutput, sizeof(float) * NUM_BLOCKS));
  checkCudaErrors(cuMemAlloc(&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
  checkCudaErrors(cuMemcpyHtoD(dinput, input, sizeof(float) * NUM_THREADS * 2));

  void *arr[] = {(void *)&dinput, (void *)&doutput, (void *)&dtimer};

  
  if(strcmp(argv[1],"x")){
    checkCudaErrors(cuLaunchKernel(
      kernel_addr, cudaGridSize_x.x, cudaGridSize_x.y,
      cudaGridSize_x.z,                                    /* grid dim */
      cudaBlockSize_x.x, cudaBlockSize_x.y, cudaBlockSize_x.z, /* block dim */
      sizeof(float) * 2 * NUM_THREADS, 0, /* shared mem, stream */
      &arr[0],                            /* arguments */
      0));  
  } else if (strcmp(argv[1],"y")){
    checkCudaErrors(cuLaunchKernel(
      kernel_addr, cudaGridSize_y.x, cudaGridSize_y.y,
      cudaGridSize_y.z,                                    /* grid dim */
      cudaBlockSize_y.x, cudaBlockSize_y.y, cudaBlockSize_y.z, /* block dim */
      sizeof(float) * 2 * NUM_THREADS, 0, /* shared mem, stream */
      &arr[0],                            /* arguments */
      0));
  }  
  checkCudaErrors(cuCtxSynchronize());
  checkCudaErrors(
      cuMemcpyDtoH(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));
  checkCudaErrors(cuMemFree(dinput));
  checkCudaErrors(cuMemFree(doutput));
  checkCudaErrors(cuMemFree(dtimer));

  long double avgElapsedClocks = 0;

  for (int i = 0; i < NUM_BLOCKS; i++) {
    avgElapsedClocks += (long double)(timer[i + NUM_BLOCKS] - timer[i]);
  }

  avgElapsedClocks = avgElapsedClocks / NUM_BLOCKS;
  printf("Average clocks/block = %Lf\n", avgElapsedClocks);

  return EXIT_SUCCESS;
}
