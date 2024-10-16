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

#include "hip/hip_runtime.h"

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
#include <unistd.h>

using namespace FullDTW;

//---------------------------------------------------------global
// vars----------------------------------------------------------//
hipStream_t stream_var[STREAM_NUM];

int main(int argc, char **argv) {

  // create host storage and buffers on devices
  value_ht *host_query, // time series on CPU
      *host_dist,       // distance results on CPU
      // *host_ref_coeff1, *host_ref_coeff2,                // re-arranged ref
      // time series on CPU
      *device_query[STREAM_NUM], // time series on GPU
      *device_dist[STREAM_NUM];  // distance results on GPU

  value_ht *device_last_row[STREAM_NUM]; // stors last column of sub-matrix

  reference_coefficients *h_ref_coeffs, *d_ref_coeffs,
      *h_ref_coeffs_tmp; // struct stores reference genome's coeffs for DTW;
                         // *tmp is before restructuring for better mem
                         // coalescing
  raw_t *squiggle_data = NULL;
  std::vector<std::string> read_ids; // store read_ids to dump in output
  //****************************************************Target ref loading &
  // re-organization for better mem coalescing & target
  // loading****************************************//

  TIMERSTART(load_target)
  if (argc != 3) {
      std::cerr << "Error: Invalid number of arguments." << std::endl;
      std::cerr << "Usage: " << argv[0] << " <model_file> <ref_file>" << std::endl;
      std::abort(); // Abort the program
  }
  std::string model_file = argv[1], ref_file = argv[2];

  load_reference *REF_LD = new load_reference;

  REF_LD->ref_loader(ref_file);
  REF_LD->read_kmer_model(model_file);
  ASSERT(hipMallocManaged(&h_ref_coeffs_tmp,
                           (sizeof(reference_coefficients) *
                            (REF_LEN)))); // host pinned memory for reference
  ASSERT(hipMallocManaged(&h_ref_coeffs,
                           (sizeof(reference_coefficients) *
                            (REF_LEN)))); // host pinned memory for reference
  REF_LD->load_ref_coeffs(h_ref_coeffs_tmp);

  delete REF_LD;

  idxt k = 0;
  for (idxt l = 0; l < (REF_LEN / REF_TILE_SIZE); l += 1) {
    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      for (idxt j = 0; j < WARP_SIZE; j++) {
        h_ref_coeffs[k].coeff1 =
            h_ref_coeffs_tmp[(l * REF_TILE_SIZE) + (j * SEGMENT_SIZE) + i]
                .coeff1;
        // h_ref_coeffs[k].coeff2 =
        //     h_ref_coeffs_tmp[(l * REF_TILE_SIZE) + (j * SEGMENT_SIZE) + i]
        //         .coeff2;

        // std::cout << HALF2FLOAT(h_ref_coeffs[k].coeff1) << ","
        //           << HALF2FLOAT(h_ref_coeffs[k].coeff2) << "\n";
        k++;
      }
      // std::cout << "warp\n";
    }
  }

  ASSERT(
    hipFree(h_ref_coeffs_tmp)); // delete the tmp array

  ASSERT(
    hipMalloc(&(d_ref_coeffs), (sizeof(reference_coefficients) * REF_LEN)));

  ASSERT(hipMemcpyAsync(
      d_ref_coeffs,
      h_ref_coeffs, //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
      (sizeof(reference_coefficients) * REF_LEN), hipMemcpyHostToDevice));

  TIMERSTOP(load_target)

  //*************************************************************LOAD FROM
  // FILE********************************************************//
  TIMERSTART(load_data)
  index_t NUM_READS=0; // counter to count number of reads to be
                     // processed + reference length
  generate_cbf(squiggle_data, QUERY_LEN, NUM_READS);

  // NUM_READS = 1; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!lkdnsknefkwnef
  ASSERT(hipHostMalloc(
      &squiggle_data,
      (sizeof(raw_t) *
       (NUM_READS * QUERY_LEN)))); // host pinned memory for raw data from FAST5

  //****************************************************NORMALIZER****************************************//
  // normalizer instance - does h2h pinned mem transfer, CUDNN setup andzscore
  // normalization, normalized raw_t output is returned in same array as input
//   normalizer *NMZR = new normalizer(NUM_READS);
//   TIMERSTART(normalizer_kernel)

//   NMZR->normalize(raw_array, NUM_READS, QUERY_LEN);

//   TIMERSTOP(normalizer_kernel)
// #ifdef NV_DEBUG
//   std::cout << "cuDTW:: Normalizer processed  " << (QUERY_LEN * NUM_READS)
//             << " raw samples in this time\n";
// #endif

// #ifdef NV_DEBUG
//   NMZR->print_normalized_query(raw_array, NUM_READS, read_ids);
// #endif

//   delete NMZR;
  // normalization completed

  //****************************************************FLOAT to
  //__half2****************************************//
  ASSERT(hipHostMalloc(
      &host_query, sizeof(value_ht) * (NUM_READS * QUERY_LEN + WARP_SIZE))); /*
                     input */
  std::cout << "Normalized data:\n";

  for (index_t i = 0; i < NUM_READS; i++) {
    for (index_t j = 0; j < QUERY_LEN; j++) {
      host_query[(i * QUERY_LEN + j)] =
          FLOAT2HALF2(squiggle_data[(i * QUERY_LEN + j)]);
    }
  }
  for (index_t i = 0; i < WARP_SIZE; i++) {
    host_query[NUM_READS * QUERY_LEN + i] = FLOAT2HALF2(0.0f);
  }
  ASSERT(
    hipHostFree(squiggle_data));
  TIMERSTOP(load_data)

  //****************************************************MEM
  // allocation****************************************//
  TIMERSTART(malloc)
  //--------------------------------------------------------host mem
  // allocation--------------------------------------------------//

  // ASSERT(hipHostMalloc(&host_ref, sizeof(value_ht) * REF_LEN)); /* input

  ASSERT(hipHostMalloc(&host_dist, sizeof(value_ht) * NUM_READS)); /* results
                                                                     */

  //-------------------------------------------------------------dev mem
  // allocation-------------------------------------------------//

  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    ASSERT(
        hipMalloc(&device_query[stream_id],
                   (sizeof(value_ht) * (BLOCK_NUM * QUERY_LEN + WARP_SIZE))));
    ASSERT(hipMalloc(&device_dist[stream_id], (sizeof(value_ht) * BLOCK_NUM)));
    ASSERT(hipStreamCreate(&stream_var[stream_id]));
    ASSERT(hipMalloc(&device_last_row[stream_id],
                      (sizeof(value_ht) * (REF_LEN * BLOCK_NUM))));
  }

  TIMERSTOP(malloc)
  ASSERT(
    hipDeviceSetCacheConfig(hipFuncCachePreferShared)); //
  //****************************************************Mem I/O and DTW
  // computation****************************************//

  //-------------total batches of concurrent workload to & fro
  // device---------------//
  idxt batch_count = NUM_READS / (BLOCK_NUM * STREAM_NUM);
  std::cout << "Batch count: " << batch_count << " num_reads:" << NUM_READS
            << "\n";
  TIMERSTART_HIP(concurrent_DTW_kernel_launch)
  for (idxt batch_id = 0; batch_id <= batch_count; batch_id += 1) {
    std::cout << "Processing batch_id: " << batch_id << "\n";

    idxt rds_in_batch = (BLOCK_NUM * STREAM_NUM);
    if (batch_id < batch_count)
      rds_in_batch = (BLOCK_NUM * STREAM_NUM);
    else if ((batch_id == batch_count) &&
             ((NUM_READS % (BLOCK_NUM * STREAM_NUM)) == 0)) {
      if (batch_count != 0)
        break;
      else
        rds_in_batch = NUM_READS;
    } else if ((batch_id == batch_count) &&
               ((NUM_READS % (BLOCK_NUM * STREAM_NUM)) != 0)) {
      rds_in_batch = NUM_READS % (BLOCK_NUM * STREAM_NUM);
    }
    for (idxt stream_id = 1; (stream_id <= STREAM_NUM) && (rds_in_batch != 0);
         stream_id += 1) {

      idxt rds_in_stream = BLOCK_NUM;

      if ((rds_in_batch - BLOCK_NUM) < 0) {
        rds_in_stream = rds_in_batch;
        rds_in_batch = 0;
      } else {
        rds_in_batch -= BLOCK_NUM;
        rds_in_stream = BLOCK_NUM;
      }
      std::cout << "Issuing " << rds_in_stream
                << " reads (blocks) from base addr:"
                << (batch_id * STREAM_NUM * BLOCK_NUM * QUERY_LEN) +
                       ((stream_id - 1) * BLOCK_NUM * QUERY_LEN)
                << " to stream_id " << (stream_id - 1) << "\n";
      //----h2d copy-------------//
      ASSERT(hipMemcpyAsync(
          device_query[stream_id - 1],
          &host_query[(batch_id * STREAM_NUM * BLOCK_NUM * QUERY_LEN) +
                      ((stream_id - 1) * BLOCK_NUM * QUERY_LEN)],
          (sizeof(value_ht) * (QUERY_LEN * rds_in_stream + WARP_SIZE)),
          hipMemcpyHostToDevice, stream_var[stream_id - 1]));

      //---------launch kernels------------//
      distances<value_ht, idxt>(d_ref_coeffs, device_query[stream_id - 1],
                                device_dist[stream_id - 1], rds_in_stream,
                                FLOAT2HALF2(0), stream_var[stream_id - 1],
                                device_last_row[stream_id - 1]);

      //-----d2h copy--------------//
      ASSERT(hipMemcpyAsync(
          &host_dist[(batch_id * STREAM_NUM * BLOCK_NUM) +
                     ((stream_id - 1) * BLOCK_NUM)],
          device_dist[stream_id - 1], (sizeof(value_ht) * rds_in_stream),
          hipMemcpyDeviceToHost, stream_var[stream_id - 1]));
    }
  }

  ASSERT(hipDeviceSynchronize());
  TIMERSTOP_HIP(concurrent_DTW_kernel_launch, NUM_READS)

  /* -----------------------------------------------------------------print
   * output -----------------------------------------------------*/

#ifndef FP16
  std::cout << "Read_ID\t"
            << "QUERY_LEN\t"
            << "REF_LEN\t"
            << "sDTW-score\n";
  for (index_t j = 0; j < NUM_READS; j++) {
    std::cout << j << "\t" << read_ids[j] << "\t" << QUERY_LEN << "\t"
              << REF_LEN << "\t" << HALF2FLOAT(host_dist[j]) << "\n";
  }
#else
  std::cout << "Read_ID\t"
            << "QUERY_LEN\t"
            << "REF_LEN\t"
            << "sDTW score: fwd-strand\tsDTW score: rev-strand\n";
  for (index_t j = 0; j < NUM_READS; j++) {
    std::cout << j << "\t" << read_ids[j] << "\t" << QUERY_LEN << "\t"
              << REF_LEN << "\t" << HALF2FLOAT(host_dist[j].x) << "\t"
              << HALF2FLOAT(host_dist[j].y) << "\n";
  }

#endif

  /* -----------------------------------------------------------------free
   * memory -----------------------------------------------------*/
  TIMERSTART(free)
  for (int stream_id = 0; stream_id < STREAM_NUM; stream_id++) {
    ASSERT(hipFree(device_dist[stream_id]));
    ASSERT(hipFree(device_query[stream_id]));
    ASSERT(hipFree(device_last_row[stream_id]));
  }

  ASSERT(hipHostFree(host_query));
  ASSERT(hipHostFree(host_dist));
  ASSERT(hipFree(h_ref_coeffs));
  ASSERT(hipFree(d_ref_coeffs));

  TIMERSTOP(free)

  return 0;
}

#endif
