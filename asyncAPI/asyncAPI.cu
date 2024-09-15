// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // helper utility functions

#define N (16 * 1024 * 1024)
#define VAL 26

__global__ void increment_kernel(int *g_data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + VAL;
}


bool correct_output(int *data) {
  for (int i = 0; i < N; i++)
    if (data[i] != VAL) {
      printf("Error! data[%d] = %d, ref = %d\n", i, data[i], VAL);
      return false;
    }

  return true;
}

int cpucheck(){
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    int *a = 0;
    int nbytes = N * sizeof(int);

    checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    memset(a, 0, nbytes);

    for (auto i=0;i < N;i++){
            a[i] = VAL;
    }
    sdkStopTimer(&timer);
    printf("time spent by CPU in incrementing operations: %.2f ms\n", sdkGetTimerValue(&timer));
    
    bool bFinalResults = correct_output(a);
    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);  

 }

int main(int argc, char *argv[]) {

  int devID;
  cudaDeviceProp deviceProps;

  printf("[%s] - Starting...\n", argv[0]);

  // This will pick the best possible CUDA capable device
  devID = findCudaDevice(argc, (const char **)argv);

    // get device name
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s]\n", deviceProps.name);

  int nbytes = N * sizeof(int);
 
  // allocate host memory  
  int *a = 0;
  checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
  memset(a, 0, nbytes);

  // allocate device memory
  int *d_a = 0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks = dim3(N / threads.x, 1);
  printf("Processing %d integers with below threads and blocks!!\n",N);
  printf("Threads Dimension (X,Y,Z) : %d,%d,%d\n",threads.x,threads.y,threads.z);
  printf("Blocks Dimension (X,Y,Z) : %d,%d,%d\n",blocks.x,blocks.y,blocks.z);

   cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);

  checkCudaErrors(cudaDeviceSynchronize());
  float gpu_time = 0.0f;

  // asynchronously issue work to the GPU (all to stream 0)
  checkCudaErrors(cudaProfilerStart());
  sdkStartTimer(&timer);
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
  increment_kernel<<<blocks, threads, 0, 0>>>(d_a);
  cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);
  sdkStopTimer(&timer);
  checkCudaErrors(cudaProfilerStop());

    // have CPU do some work while waiting for stage 1 to finish
  unsigned long int counter = 0;

  while (cudaEventQuery(stop) == cudaErrorNotReady) {
    counter++;
  }

  checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // print the cpu and gpu times
  printf("time spent executing by the GPU: %.2f ms\n", gpu_time);
  printf("time spent by CPU in CUDA calls: %.2f ms\n", sdkGetTimerValue(&timer));
  printf("CPU executed %lu iterations while waiting for GPU to finish\n",
         counter);

  // check the output for correctness
  bool bFinalResults = correct_output(a);

  // release resources
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFreeHost(a));
  checkCudaErrors(cudaFree(d_a));
 
  cpucheck();
  exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);  
  //return 0;

}