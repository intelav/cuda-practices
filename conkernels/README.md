# concurrentKernels - Concurrent Kernels

## Description
### Refer to [CUDA GITHUB ConCurrent Kernel Sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/concurrentKernels) for original code.

The modification has been done to demonstrate cuda sample execution for quick and better understanding.

## Key Concepts

Performance Strategies
Serial Executon of Multiple Kernels and Paraellel Execution of same number of Kernels 

## Supported  Architectures Where this code has been executed successfuly.
Device 0: "NVIDIA GeForce RTX 3060"
  CUDA Driver Version / Runtime Version          12.2 / 12.4
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 12044 MBytes (12629508096 bytes)
  (028) Multiprocessors, (128) CUDA Cores/MP:    3584 CUDA Cores
  GPU Max Clock rate:                            1792 MHz (1.79 GHz)

## Profiling using Nsight 
nsys nvprof --print-gpu-trace ./conkern

Start (ns)   Duration (ns)  CorrId  GrdX  GrdY  GrdZ  BlkX  BlkY  BlkZ  Reg/Trd  StcSMem (MB)  DymSMem (MB)  Bytes (MB)  Throughput (MB/s)  SrcMemKd  DstMemKd            Device             Ctx  GreenCtx  Strm              Name            
 -----------  -------------  ------  ----  ----  ----  ----  ----  ----  -------  ------------  ------------  ----------  -----------------  --------  --------  ---------------------------  ---  --------  ----  ----------------------------
 262,634,911     10,047,686     151     1     1     1     1     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              13  clock_block(long *, long)   
 262,648,063     10,049,190     154     1     1     1     1     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              14  clock_block(long *, long)   
 262,653,246     10,054,311     157     1     1     1     1     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              15  clock_block(long *, long)   
 262,658,047     10,055,558     160     1     1     1     1     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              16  clock_block(long *, long)   
 262,662,750     10,054,631     163     1     1     1     1     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              17  clock_block(long *, long)   
 262,667,454     10,051,079     166     1     1     1     1     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              18  clock_block(long *, long)   
 262,671,838     10,050,087     169     1     1     1     1     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              19  clock_block(long *, long)   
 262,677,598     10,048,102     172     1     1     1     1     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              20  clock_block(long *, long)   
 272,734,469      1,014,454     175     1     1     1    32     1     1       32         0.000         0.000                                                     NVIDIA GeForce RTX 3060 (0)    1              21  sum(long *, int)            
 273,751,931          1,408     176                                                                                0.000              5.682  Device    Pinned    NVIDIA GeForce RTX 3060 (0)    1              21  [CUDA memcpy Device-to-Host]