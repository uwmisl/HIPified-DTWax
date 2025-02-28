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
#All rights reserved. # SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 #
 # NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 # property and proprietary rights in and to this material, related
 # documentation and any modifications thereto. Any use, reproduction,
 # disclosure or distribution of this material and related documentation
 # without an express license agreement from NVIDIA CORPORATION or
 # its affiliates is strictly prohibited.
 */
#ifndef MAIN_PROG
#define MAIN_PROG

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <string>
#include <unistd.h>

#include "include/common.hpp"
#include "include/datatypes.hpp"
#include "include/data_dims.hpp"
#include "include/DTW.hpp"
#include "include/hpc_helpers.hpp"
#include "include/read_from_txt.hpp"

using namespace FullDTW;

hipStream_t stream_var[STREAM_NUM];

int main(int argc, char **argv)
{
  // The program needs one argument, the filename containing the reference and query signals
  if (argc != 2)
  {
    std::cerr << "Error: Invalid number of arguments." << std::endl;
    std::cerr << "Usage: " << argv[0] << " <data file name>" << std::endl;
    return 1;
  }
  std::string data_file = argv[1];

#ifdef HIP_DEBUG
  printf("REF_LEN=%0d\n", REF_LEN);
  printf("QUERY_LEN=%0d\n", QUERY_LEN);
  printf("NUM_READS=%0d\n", NUM_READS);
  printf("REF_TILE_SIZE=%0d\n", REF_TILE_SIZE);
  printf("(REF_LEN / REF_TILE_SIZE)=%0d\n", (REF_LEN / REF_TILE_SIZE));
  printf("SEGMENT_SIZE=%0d\n", SEGMENT_SIZE);
  printf("WARP_SIZE=%0d\n", WARP_SIZE);
#endif
// A number of important parameters are defined in data_dims.hpp, we sanity check them here
// MQ: There are other ones I should be checking here, but I haven't teased them apart yet
  if (QUERY_LEN < QUERY_BATCH_SIZE)
  {
    std::cerr << "Error: The QUERY_BATCH_SIZE must be no larger than QUERY_LEN" << std::endl;
    return 1;
  }

  if (REF_BATCH <= 0)
  {
    std::cerr << "Error: REF_BATCH <= 0, check REF_LEN and SEGMENT_SIZE. " << std::endl;
    return 1;
  }

  if (REF_LEN % REF_TILE_SIZE != 0)
  {
    std::cerr << "Error: REF_BATCH is fractional. Check REF_LEN, SEGMENT_SIZE, and WARP_SIZE." << std::endl;
    return 1;
  }

  if (QUERY_LEN % QUERY_BATCH_SIZE != 0)
  {
    std::cerr << "Error: QUERY_LEN does not divide evenly into QUERY_BATCH_SIZE" << std::endl;
    return 1;
  }

  // Storage on host (CPU) and device (GPU)
  value_ht *host_ref,               // Reference squiggle on CPU
      *temp_host_ref,               // Reference will be temporarily stored here, prior to memory coalescing
      *host_query,                  // Queries on CPU
      *host_dist,                   // distance results on CPU
      *device_ref,                  // One reference across all streams
      *device_query[STREAM_NUM],    // One device_query per stream
      *device_dist[STREAM_NUM],     // One device_dist (distance results) per stream
      *device_last_row[STREAM_NUM]; // Last row of sub-matrix (one per stream)

  // ~~~
  // Memory Allocations
  // ~~~
  TIMERSTART(malloc)
  // index_t NUM_READS = 1;
  // On the host:
  // MQ: value_ht or raw_t...?
  ASSERT(hipHostMalloc(&host_ref, sizeof(value_ht) * REF_LEN));
  ASSERT(hipHostMalloc(&temp_host_ref, sizeof(value_ht) * REF_LEN));
  ASSERT(hipHostMalloc(&host_query, sizeof(value_ht) * (NUM_READS * QUERY_LEN + WARP_SIZE)));
  ASSERT(hipHostMalloc(&host_dist, sizeof(value_ht) * NUM_READS));
  // ASSERT(hipHostMalloc(&query_squiggle, sizeof(raw_t) * (NUM_READS * QUERY_LEN)));
  // On the device:
  ASSERT(hipMalloc(&device_ref, sizeof(value_ht) * REF_LEN));
  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++)
  {
    ASSERT(hipMalloc(&device_query[stream_id], (sizeof(value_ht) * (BLOCK_NUM * QUERY_LEN + WARP_SIZE))));
    ASSERT(hipMalloc(&device_dist[stream_id], (sizeof(value_ht) * BLOCK_NUM)));
    ASSERT(hipMalloc(&device_last_row[stream_id], (sizeof(value_ht) * (REF_LEN * BLOCK_NUM))));
    // Create the stream (saving into global variable 'stream_var')
    ASSERT(hipStreamCreate(&stream_var[stream_id]));
  }
  TIMERSTOP(malloc)

  // TODO: Adjust this to read my template in from a file
  TIMERSTART(load_data)
  std::ifstream inFile(data_file);
  if (!inFile)
  {
    std::cerr << "Error: File '" << data_file << "' does not exist or cannot be opened." << std::endl;
    // MQ: Nothing is being freed when we exit here
    return 1;
  }
  readDataFromTxt(inFile, temp_host_ref, host_query);
  inFile.close();

#ifdef HIP_DEBUG
  // Output the data to verify it was read correctly
  std::cout << "Reference read from file:" << std::endl;
  for (index_t i = 0; i < REF_LEN; i++)
  {
    std::cout << temp_host_ref[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Queries read from file:" << std::endl;
  for (index_t i = 0; i < NUM_READS; i++)
  {
    for (index_t j = 0; j < QUERY_LEN; j++)
    {
      std::cout << host_query[i * QUERY_LEN + j] << " ";
    }
    std::cout << std::endl;
  }
#endif
  // Pad the remaining WARP_SIZE elements with zeros
  // MQ: Suuuuper not sure why we do this
  for (index_t i = 0; i < WARP_SIZE; i++)
  {
    host_query[NUM_READS * QUERY_LEN + i] = 0.0f;
  }

  // Rearrange the reference values for memory coalescing on the device
  // (optimization). Each WARP_SIZE length chunk of host_ref is populated
  // with values collected by taking strides of length SEGMENT_SIZE.
  idxt k = 0;
  for (idxt ref_batch = 0; ref_batch < REF_BATCH; ref_batch++)
  {
    for (idxt i = 0; i < SEGMENT_SIZE; i++)
    {
      for (idxt j = 0; j < WARP_SIZE; j++)
      {
        host_ref[k++] = temp_host_ref[(ref_batch * REF_TILE_SIZE) + (j * SEGMENT_SIZE) + i];
      }
    }
  }
#ifdef HIP_DEBUG
  std::cout << "host_ref after memory coalescing:" << std::endl;
  for (index_t i = 0; i < REF_LEN; i++)
  {
    std::cout << host_ref[i] << " ";
  }
  std::cout << std::endl;
#endif

  // Transfer this re-arranged reference onto the GPU device
  ASSERT(hipMemcpyAsync(
      device_ref,
      &host_ref[0],
      sizeof(value_ht) * REF_LEN, hipMemcpyHostToDevice));

  // Free the host memory
  ASSERT(hipHostFree(temp_host_ref));
  ASSERT(hipHostFree(host_ref));
  TIMERSTOP(load_data)

  // Ask the device (GPU) to prefer shared memory over cache memory
  // Shared memory size will be increased, L1 cache size will be decreased
  // (Optimization)
  ASSERT(hipDeviceSetCacheConfig(hipFuncCachePreferShared));

  //-------------total batches of concurrent workload to & fro
  // device---------------//

  TIMERSTART(concurrent_DTW_kernel_launch)
  // Ceiling arithmetic to calculate total number of batches needed (last batch may be partially filled)
  idxt batch_count = (NUM_READS + (BLOCK_NUM * STREAM_NUM) - 1) / (BLOCK_NUM * STREAM_NUM);
  std::cout << "Batch count: " << batch_count << " num_reads:" << NUM_READS
            << "\n";
  // For each batch:
  for (idxt batch_id = 0; batch_id < batch_count; batch_id += 1)
  {
    std::cout << "Processing batch_id: " << batch_id << "\n";
    // Number of reads that this batch is going to process
    idxt rds_in_batch = (BLOCK_NUM * STREAM_NUM);
    // Special processing for when the final batch is only partially filled
    if (batch_id == batch_count - 1 && NUM_READS % (BLOCK_NUM * STREAM_NUM) != 0)
    {
      // Final batch gets the remaining reads
      rds_in_batch = NUM_READS % (BLOCK_NUM * STREAM_NUM);
    }

    // Each batch will process using streams
    // It will use up to STREAM_NUM number of streams, or
    // as many as it takes to process the rds_in_batch (for a partially filled batch)
    for (idxt stream_id = 0; (stream_id < STREAM_NUM) && (rds_in_batch != 0);
         stream_id += 1)
    {
      // Each stream will process up to BLOCK_NUM amount of reads
      idxt rds_in_stream;
      if (rds_in_batch >= BLOCK_NUM)
      {
        rds_in_stream = BLOCK_NUM;
        rds_in_batch -= BLOCK_NUM;
      }
      else
      {
        // Edge case: fewer than BLOCK_NUM reads remaining
        rds_in_stream = rds_in_batch;
        rds_in_batch = 0;
      }
      std::cout << "Issuing " << rds_in_stream
                << " reads (blocks) from base addr:"
                << (batch_id * STREAM_NUM * BLOCK_NUM * QUERY_LEN) +
                       (stream_id * BLOCK_NUM * QUERY_LEN)
                << " to stream_id " << stream_id << "\n";
      // Copy host_query to device_query
      // Move the queries from CPU to GPU
      ASSERT(hipMemcpyAsync(
          device_query[stream_id],
          &host_query[(batch_id * STREAM_NUM * BLOCK_NUM * QUERY_LEN) +
                      (stream_id * BLOCK_NUM * QUERY_LEN)],
          (sizeof(value_ht) * (QUERY_LEN * rds_in_stream + WARP_SIZE)),
          hipMemcpyHostToDevice, stream_var[stream_id]));

      // Launch the kernel
      distances<value_ht, idxt>(device_ref, device_query[stream_id],
                                device_dist[stream_id], rds_in_stream,
                                stream_var[stream_id], device_last_row[stream_id]);

      // Copy the distance results back to the host
      // Move device_dist to host_dist
      ASSERT(hipMemcpyAsync(
          &host_dist[(batch_id * STREAM_NUM * BLOCK_NUM) +
                     ((stream_id)*BLOCK_NUM)],
          device_dist[stream_id], (sizeof(value_ht) * rds_in_stream),
          hipMemcpyDeviceToHost, stream_var[stream_id]));
    }
  }

  // Wait on all active streams
  ASSERT(hipDeviceSynchronize());
  TIMERSTOP(concurrent_DTW_kernel_launch)

  // Print final output
  std::cout << "Results:\n";
  std::cout << "QUERY_LEN\t"
            << "REF_LEN\t"
            << "DTW-score\n";
  for (index_t j = 0; j < NUM_READS; j++)
  {
    printf("%ld\t%d\t%d\t%.7f\n", j, QUERY_LEN, REF_LEN, host_dist[j]);
  }

  // Free remaining memory
  TIMERSTART(free)
  // On the host (CPU):
  ASSERT(hipHostFree(host_query));
  ASSERT(hipHostFree(host_dist));
  // On the device (GPU):
  ASSERT(hipFree(device_ref));
  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++)
  {
    ASSERT(hipFree(device_dist[stream_id]));
    ASSERT(hipFree(device_query[stream_id]));
    ASSERT(hipFree(device_last_row[stream_id]));
  }
  TIMERSTOP(free)

  return 0;
}

#endif
