# CPP Integration

## Description
### Refer to [CUDA GITHUB CPPIntegration Sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/cppIntegration) for original code.

The modification has been done to demonstrate cuda sample execution for quick and better understanding.

## Key Concepts

Demonstrating the cpp integration of CUDA Kernel by writing a simple array of "Hello World"  using scalar and vector data types. For Scalar type, the 16 byte data has been grouped in 4 bytes each  which will be executed by one thread . There would be total 4 threads for scalar type.  The vector  data type is being executed using 16 threads where each thread is executing a single byte . 

## Supported  Architectures Where this code has been executed successfuly.
Device 0: "NVIDIA GeForce RTX 3060"
  CUDA Driver Version / Runtime Version          12.2 / 12.4
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 12044 MBytes (12629508096 bytes)
  (028) Multiprocessors, (128) CUDA Cores/MP:    3584 CUDA Cores
  GPU Max Clock rate:                            1792 MHz (1.79 GHz)

## Profiling using Nsight 
nsys nvprof --print-gpu-trace ./cppintegration

Start (ns)   Duration (ns)  CorrId  GrdX  GrdY  GrdZ  BlkX  BlkY  BlkZ  Reg/Trd  StcSMem (MB)  DymSMem (MB)  Bytes (MB)  Throughput (MB/s)  SrcMemKd  DstMemKd            Device             Ctx  GreenCtx  Strm              Name            
 -----------  -------------  ------  ----  ----  ----  ----  ----  ----  -------  ------------  ------------  ----------  -----------------  --------  --------  ---------------------------  ---  --------  ----  ----------------------------
 288,824,971            672     128                                                                                0.000             23.810  Pageable  Device    NVIDIA GeForce RTX 3060 (0)    1               7  [CUDA memcpy Host-to-Device]
 288,835,691            352     130                                                                                0.000            363.636  Pageable  Device    NVIDIA GeForce RTX 3060 (0)    1               7  [CUDA memcpy Host-to-Device]
 291,473,497         65,025     131     1     1     1     4     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1               7  kernel(int *)               
 291,545,786        137,633     132     1     1     1    16     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1               7  kernel2(int2 *)             
 291,685,531          1,440     134                                                                                0.000             11.111  Device    Pageable  NVIDIA GeForce RTX 3060 (0)    1               7  [CUDA memcpy Device-to-Host]
 291,741,467          1,280     135                                                                                0.000            100.000  Device    Pageable  NVIDIA GeForce RTX 3060 (0)    1               7  [CUDA memcpy Device-to-Host]

