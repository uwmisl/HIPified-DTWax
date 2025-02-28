/*
This file has been autogenerated by /home1/mqueen/DTWax/build/build_DTWax.sh for the data in /home1/mqueen/DTWax/test/data.temp
If you have a new data file, rebuilding with build_DTWax.sh is recommended.
*/

#ifndef DATA_DIMS_HPP
#define DATA_DIMS_HPP

// The number of threads in each warp. This is 64 for AMD and 32 for NVIDIA.
// This version of DTWax has not been tested on NVIDIA hardware.
// Because of the shfl_up operations, the calculations will NOT be correct if you have the incorrect number.
#define WARP_SIZE 64

// Must be no larger than QUERY_LEN. This is the number of query elements in each query batch.
#define QUERY_BATCH_SIZE 64

// Each thread will process SEGMENT_SIZE elements of the reference sequence at once
#define SEGMENT_SIZE 1

// Data dimensions
#define REF_LEN 320
#define QUERY_LEN 320
#define NUM_READS 100

// Derived constants
#define REF_TILE_SIZE (SEGMENT_SIZE * WARP_SIZE)
#define REF_BATCH (REF_LEN / REF_TILE_SIZE)
#define QUERY_BATCH (QUERY_LEN / QUERY_BATCH_SIZE)
#define NUM_WAVES (QUERY_BATCH_SIZE + (REF_TILE_SIZE - 1) / (SEGMENT_SIZE))
#define WARP_SIZE_MINUS_ONE (WARP_SIZE - 1)
#define NUM_WAVES_BY_WARP_SIZE ((NUM_WAVES / WARP_SIZE) * WARP_SIZE)
#define REF_BATCH_MINUS_ONE (REF_BATCH - 1)
#define TWICE_WARP_SIZE_MINUS_ONE ((2 * WARP_SIZE) - 1)
#define RESULT_REG (SEGMENT_SIZE - 1)

#endif // DATA_DIMS_HPP
