// System includes
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// This kernel computes a standard parallel reduction and evaluates the
// time it takes to do that for each block. The timing results are stored
// in device memory.
__global__ static void timedReduction( const float *input, float *output,
                                      clock_t *timer) {
  // __shared__ float shared[2 * blockDim.x];
  extern __shared__ float shared[];

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  printf("value of threadix.x=%d,blockidx.x=%d,blockdim.x=%d,gridDim.x=%d\n",threadIdx.x,blockIdx.x,blockDim.x,gridDim.x);

  if (tid == 0){ 
  timer[bid] = clock(); //return the clock value in HZ
  printf("Start time for Block=%d is %f\n",bid,(long double)timer[bid]);
  }

  

  // Copy input.
  shared[tid] = input[tid];
  shared[tid + blockDim.x] = input[tid + blockDim.x];
  
  //printf("value of blockDim.x=%d\n",blockDim.x);
  
 
  // Perform reduction to find minimum.
  for (int d = blockDim.x; d > 0; d /= 2) {
    __syncthreads();

    if (tid < d) {
      float f0 = shared[tid];
      float f1 = shared[tid + d];

      
      if (f1 < f0) {
        shared[tid] = f1;
        printf("Here f1 is less than f0.\n");
      }
      printf("shared[tid]=%f, f0=%f,f1=%f,d=%d,tid=%d\n",shared[tid],f0,f1,d,tid);
    }
  }
  

  // Write result.
  if (tid == 0){
    output[bid] = shared[0];
    printf("Minimum value for Block %d is output[bid]=%f\n",bid,output[bid]);
  
  } 
  

  __syncthreads();

  if (tid == 0) {
    timer[bid + gridDim.x] = clock();
    printf("Completion Time for Block %d is %f\n",bid,(long double)timer[bid+gridDim.x]);
  }
}

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
//    64 - 9981
//
// With less than 16 blocks some of the multiprocessors of the device are idle.
// With more than 16 you are using all the multiprocessors, but there's only one
// block per multiprocessor and that doesn't allow you to hide the latency of
// the memory. With more than 32 the speed scales linearly.

// Start the main CUDA Sample here
int main(int argc, char **argv) {
  printf("CUDA Clock sample\n");

  // This will pick the best possible CUDA capable device
  int dev = findCudaDevice(argc, (const char **)argv);

  float *dinput = NULL;
  float *doutput = NULL;
  clock_t *dtimer = NULL;

  clock_t timer[NUM_BLOCKS * 2];
  float input[NUM_THREADS * 2];

  for (int i = 0; i < NUM_THREADS * 2; i++) {
    //srand(time(0));
    input[i] = rand() ;
    printf("Assigned values for input[%d]=%f\n",i,input[i]);
  }


  checkCudaErrors(
      cudaMalloc((void **)&dinput, sizeof(float) * NUM_THREADS * 2));
  checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS));
  checkCudaErrors(
      cudaMalloc((void **)&dtimer, sizeof(clock_t) * NUM_BLOCKS * 2));

  checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float) * NUM_THREADS * 2,
                             cudaMemcpyHostToDevice));

 timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(
     dinput, doutput, dtimer);

  checkCudaErrors(cudaMemcpy(timer, dtimer, sizeof(clock_t) * NUM_BLOCKS * 2,
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(dinput));
  checkCudaErrors(cudaFree(doutput));
  checkCudaErrors(cudaFree(dtimer));

  long double avgElapsedClocks[NUM_BLOCKS]={0};
  long double total=0; 
   
  for (int i = 0; i < NUM_BLOCKS; i++) {
    avgElapsedClocks[i] = (long double)(timer[i + NUM_BLOCKS] - timer[i]);
    printf("Host recevied clocks consumed for Block %d Completed=%Lf,Entered=%Lf,avgElapsedClocks=%Lf\n",i,(long double)timer[i+NUM_BLOCKS],(long double)timer[i],avgElapsedClocks[i]);
  }
  for (int i = 0; i < NUM_BLOCKS; i++) {
    total += avgElapsedClocks[i];
  }
  printf("Average clocks/block = %Lf\n", (total / NUM_BLOCKS));

  return EXIT_SUCCESS;
}