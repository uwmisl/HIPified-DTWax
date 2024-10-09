/*
// Not a contribution
// Changes made by NVIDIA CORPORATION & AFFILIATES enabling <XYZ> or otherwise
documented as
// NVIDIA-proprietary are not a contribution and subject to the following terms
and conditions:
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 
 # SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 #
 # NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 # property and proprietary rights in and to this material, related
 # documentation and any modifications thereto. Any use, reproduction,
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
 */
#ifndef HPC_HELPERS_HPP
#define HPC_HELPERS_HPP

#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <string>
#include <unistd.h>

#ifndef __HIPCC__
#include <chrono>
#endif

#ifndef __HIPCC__
#define TIMERSTART(label)                                                      \
  std::chrono::time_point<std::chrono::system_clock> a##label, b##label;       \
  a##label = std::chrono::system_clock::now();
#else
#define TIMERSTART(label)                                                      \
  hipSetDevice(0);                                                             \
  hipEvent_t start##label, stop##label;                                       \
  float time##label;                                                           \
  hipEventCreate(&start##label);                                              \
  hipEventCreate(&stop##label);                                               \
  hipEventRecord(start##label, 0);
#endif

#ifndef __HIPCC__
#define TIMERSTOP(label)                                                       \
  b##label = std::chrono::system_clock::now();                                 \
  std::chrono::duration<double> delta##label = b##label - a##label;            \
  std::cout << "# elapsed time (" << #label << "): " << delta##label.count()   \
            << "s" << std::endl;
#else
#define TIMERSTOP(label)                                                       \
  hipSetDevice(0);                                                             \
  hipEventRecord(stop##label, 0);                                              \
  hipError_t sync_error##label = hipEventSynchronize(stop##label);             \
  HIPERR;                                                                      \
  hipError_t timer_error##label = hipEventElapsedTime(&time##label, start##label, stop##label); \
  HIPERR;                                                                      \
  std::cout << "TIMING: " << time##label << " ms (" << #label << ")"           \
            << std::endl;
#endif

#ifdef __HIPCC__
#define HIPERR                                                                  \
  {                                                                            \
    hipError_t err;                                                           \
    if ((err = hipGetLastError()) != hipSuccess) {                           \
      std::cout << "HIP error: " << hipGetErrorString(err) << " : "          \
                << __FILE__ << ", line " << __LINE__ << std::endl;             \
      exit(1);                                                                 \
    }                                                                          \
  }
#endif

//------------------------------------------------------------time
// macros-----------------------------------------------------//
#define TIMERSTART_HIP(label)                                                  \
  hipSetDevice(0);                                                             \
  hipEvent_t start##label, stop##label;                                        \
  float time##label;                                                           \
  hipEventCreate(&start##label);                                               \
  hipEventCreate(&stop##label);                                                \
  hipEventRecord(start##label, 0);

#define TIMERSTOP_HIP(label, NUM_READS)                                        \
  hipSetDevice(0);                                                             \
  hipEventRecord(stop##label, 0);                                              \
  hipEventSynchronize(stop##label);                                            \
  hipError_t error##label = hipEventElapsedTime(&time##label, start##label, stop##label);        \
  HIPERR;                                                                      \
  std::cout << "TIMING: " << time##label << " ms "                             \
            << ((QUERY_LEN / (time##label * 1e3)) * NUM_READS / 10)            \
            << " Mbps (" << #label << ")" << std::endl;

//..........................................................other
// macros.......................................................//
#define ASSERT(ans)                                                            \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != hipSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// safe division
#define SDIV(x, y) (((x) + (y)-1) / (y))

// void input_parser(int argc, char *argv[], std::string &ip_path,
//                   std::string &model_file, std::string &ref_file) {
//   // int opt;
//   // std::cout << argc;
//   // while ((opt = getopt(argc, argv, “abc :”)) != -1) {
//   //   switch (opt) {
//   //   case ‘i’:
//   //     ip_path = optarg;
//   //     std::cout << "input path is " << ip_path << "\n";
//   //     break;
//   //   case ‘k’:
//   //     model_file = optarg;
//   //     std::cout << "model file is " << model_file << "\n";
//   //     break;
//   //   case ‘r’:
//   //     ref_file = optarg;
//   //     std::cout << "fasta reference file is " << ref_file << "\n";
//   //     break;
//   //   }
//   // }
// }
#endif