# asyncAPI - Demonstrating Asynchornicity between CPU and GPU execution

## Description

The reference for this sample has been taken from https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/asyncAPI

### How is it different from original Sample from Nvidia

For better understanding , it prints outputs by showing the difference between CPU execution time and GPU execution time beside showing overlap in execution. 
Extra code of cpucheck() has been added here.

```
GPU Device 0: "Ampere" with compute capability 8.6

CUDA device [NVIDIA GeForce RTX 3060]
Processing 16777216 integers with below threads and blocks!!
Threads Dimension (X,Y,Z) : 512,1,1
Blocks Dimension (X,Y,Z) : 32768,1,1
time spent executing by the GPU: 47.30 ms
time spent by CPU in CUDA calls: 25.93 ms
CPU executed 129240 iterations while waiting for GPU to finish
time spent by CPU in incrementing operations: 44.06 ms
```
Note **47.30 ms by GPU** and **44.06 ms by CPU**
### Memory Copy takes most of the time (comment the memory copy operations)!!
``` [./asyncAPI] - Starting...
GPU Device 0: "Ampere" with compute capability 8.6

CUDA device [NVIDIA GeForce RTX 3060]
Processing 16777216 integers with below threads and blocks!!
Threads Dimension (X,Y,Z) : 512,1,1
Blocks Dimension (X,Y,Z) : 32768,1,1
time spent executing by the GPU: 0.52 ms
time spent by CPU in CUDA calls: 0.12 ms
CPU executed 1744 iterations while waiting for GPU to finish
Error! data[0] = 0, ref = 26
time spent by CPU in incrementing operations: 48.21 ms
```
1. Note **.52 ms by GPU** and **48.21ms by CPU**.
2. **GPU are 100 times faster here**
