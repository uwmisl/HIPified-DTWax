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

#include "include/common.hpp"
#include "include/datatypes.hpp"
#include <stdio.h>
#include <string>
#include <unistd.h>

#include "include/DTW.hpp"
#include "include/hpc_helpers.hpp"
#include "include/load_reference.hpp"
#include "include/cbf_generator.hpp"
#include <hip_runtime_api.h>

using namespace FullDTW;

hipStream_t stream_var[STREAM_NUM];

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cerr << "Error: Invalid number of arguments." << std::endl;
    std::cerr << "Usage: " << argv[0] << " <model_file> <ref_file>" << std::endl;
    std::abort();
  }
  std::string model_file = argv[1], ref_file = argv[2];

  // Storage on host (CPU) and device (GPU)
  value_ht *host_ref,               // Reference squiggle on CPU
      *host_query,                  // Queries on CPU
      *host_dist,                   // distance results on CPU
      *device_ref,                  // One reference across all streams
      *device_query[STREAM_NUM],    // One device_query per stream
      *device_dist[STREAM_NUM],     // One device_dist (distance results) per stream
      *device_last_row[STREAM_NUM]; // Last (column?) of sub-matrix (one per stream)

  raw_t *query_squiggle = NULL;
  std::vector<std::string> read_ids; // store read_ids to dump in output

  // ~~~
  // Memory Allocations
  // ~~~
  TIMERSTART(malloc)
  index_t NUM_READS = 1;
  // On the host:
  // MQ: value_ht or raw_t...?
  ASSERT(hipHostMalloc(&host_ref, sizeof(value_ht) * REF_LEN));
  ASSERT(hipHostMalloc(&host_query, sizeof(value_ht) * (NUM_READS * QUERY_LEN + WARP_SIZE)));
  ASSERT(hipHostMalloc(&host_dist, sizeof(value_ht) * NUM_READS));
  ASSERT(hipHostMalloc(&query_squiggle, sizeof(raw_t) * (NUM_READS * QUERY_LEN)));
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
  generate_cbf(query_squiggle, QUERY_LEN, NUM_READS);

  // This just copies query_squiggle into host_query? Calling FLOAT2HALF?
  for (index_t i = 0; i < NUM_READS; i++)
  {
    for (index_t j = 0; j < QUERY_LEN; j++)
    {
      host_query[(i * QUERY_LEN) + j] = FLOAT2HALF(query_squiggle[((i * QUERY_LEN) + j)]);
    }
  }

  // Pad the remaining WARP_SIZE elements with zeros
  for (index_t i = 0; i < WARP_SIZE; i++)
  {
    host_query[NUM_READS * QUERY_LEN + i] = FLOAT2HALF(0.0f);
  }

  // We assume that the first query_squiggle will be treated as the reference
  // Rearrange the reference values for memory coalescing on the device
  // (optimization). Each WARP_SIZE length chunk of host_ref is populated
  // with values collected by taking strides of length SEGMENT_SIZE.
  index_t k = 0;
  for (index_t i = 0; i < SEGMENT_SIZE; i++)
  {
    for (index_t j = 0; j < WARP_SIZE; j++)
    {
      host_ref[k++] = FLOAT2HALF(query_squiggle[i + (j * SEGMENT_SIZE)]);
    }
  }

  // Transfer this re-arranged reference onto the GPU device
  ASSERT(hipMemcpyAsync(
      device_ref,
      &host_ref[0],
      sizeof(value_ht) * REF_LEN, hipMemcpyHostToDevice));

  // Free the host memory
  ASSERT(hipHostFree(query_squiggle));
  ASSERT(hipHostFree(host_ref));
  TIMERSTOP(load_data)

  // Ask the device (GPU) to prefer shared memory over cache memory
  // Shared memory size will be increased, L1 cache size will be decreased
  // (Optimization)
  ASSERT(hipDeviceSetCacheConfig(hipFuncCachePreferShared));

  //-------------total batches of concurrent workload to & fro
  // device---------------//

  TIMERSTART_HIP(concurrent_DTW_kernel_launch)
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
                                FLOAT2HALF(0), stream_var[stream_id],
                                device_last_row[stream_id]);

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
  TIMERSTOP_HIP(concurrent_DTW_kernel_launch, NUM_READS)

// Print final output
#ifndef FP16
  std::cout << "Read_ID\t"
            << "QUERY_LEN\t"
            << "REF_LEN\t"
            << "sDTW-score\n";
  for (index_t j = 0; j < NUM_READS; j++)
  {
    std::cout << j << "\t" << read_ids[j] << "\t" << QUERY_LEN << "\t"
              << REF_LEN << "\t" << HALF2FLOAT(host_dist[j]) << "\n";
  }
#else
  std::cout << "Read_ID\t"
            << "QUERY_LEN\t"
            << "REF_LEN\t"
            << "sDTW score: fwd-strand\tsDTW score: rev-strand\n";
  for (index_t j = 0; j < NUM_READS; j++)
  {
    std::cout << j << "\t" << read_ids[j] << "\t" << QUERY_LEN << "\t"
              << REF_LEN << "\t" << HALF2FLOAT(host_dist[j].x) << "\t"
              << HALF2FLOAT(host_dist[j].y) << "\n";
  }

#endif

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
