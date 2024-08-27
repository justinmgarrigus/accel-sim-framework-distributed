/*
 * Demonstrates an all-reduce operation with NCCL. Taken from: 
 * https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
 */

#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

#define N_DEVICES 1

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

__global__ void test_kernel(int* ptr) {
    ptr[threadIdx.x] = 7;  
}

int main(int argc, char* argv[])
{
  ncclComm_t comms[N_DEVICES];

  //managing 4 devices
  int nDev = N_DEVICES;
  int size = 32*1024*1024;
  
  int devs[N_DEVICES];
  for (int i = 0; i < N_DEVICES; i++) 
    devs[i] = i; 

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));

    float *values = (float*)malloc(sizeof(float) * size); 
    for (int j = 0 ; j < size; j++) values[j] = 3.14f * (1 + i); 
    CUDACHECK(cudaMemcpy(sendbuff[i], values, size * sizeof(float), cudaMemcpyHostToDevice)); 
    free(values); 
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
    
  // Random code representing a kernel launch. 
  int* d_ptr; 
  cudaMalloc(&d_ptr, sizeof(int) * 256);
  test_kernel<<<1, 256>>>(d_ptr); 

   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());
    
  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  // Another kernel launch to demonstrate the time difference 
  test_kernel<<<1,256>>>(d_ptr);

  // Display the first 8 values of each output buffer. 
  printf("Displaying the first 8 values of the send/receive buffers:\n"); 
  float *sendbuff_h = (float*)malloc(sizeof(float) * 8); 
  float *recvbuff_h = (float*)malloc(sizeof(float) * 8); 
  cudaMemcpy(sendbuff_h, sendbuff[0], sizeof(float) * 8, 
    cudaMemcpyDeviceToHost); 
  cudaMemcpy(recvbuff_h, recvbuff[0], sizeof(float) * 8, 
    cudaMemcpyDeviceToHost); 
  for (int i = 0; i < 8; i++)
    printf("%d: %f, %f\n", i, sendbuff_h[i], recvbuff_h[i]);
  free(sendbuff_h);
  free(recvbuff_h); 

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
