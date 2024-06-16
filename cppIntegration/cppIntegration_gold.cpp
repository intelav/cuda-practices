/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Example of integrating CUDA functions into an existing
 * application / framework.
 * Reference solution computation.
 */

// Required header to support CUDA vector types
#include <vector_types.h>
#include <iostream>
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" void computeGold(char *reference, char *idata,
                            const unsigned int len);
extern "C" void computeGold2(int2 *reference, int2 *idata,
                             const unsigned int len);
using namespace std;
////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void computeGold(char *reference, char *idata, const unsigned int len) {
  for (unsigned int i = 0; i < len; ++i) {
    cout << "i,reference,idata=" << i << ","<< reference[i] << ","<<idata[i] <<  endl;
    reference[i] = idata[i] - 10;
    cout << "after operation:i,reference,idata=" << i << ","<< reference[i] << ","<<idata[i] <<  endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for int2 version
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void computeGold2(int2 *reference, int2 *idata, const unsigned int len) {
  for (unsigned int i = 0; i < len; ++i) {
    cout << "i,reference.x,reference.y,idata,x,idata.y=" << i << ","<< reference[i].x << ","<<reference[i].y << "," <<idata[i].x << "," << idata[i].y << endl;
    reference[i].x = idata[i].x - idata[i].y;
    reference[i].y = idata[i].y;
    cout << "after operation:i,reference.x,reference.y,idata,x,idata.y=" << i << ","<< reference[i].x << ","<<reference[i].y << "," <<idata[i].x << "," << idata[i].y << endl;
  }
}
