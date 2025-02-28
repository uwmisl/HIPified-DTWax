#include "hip/hip_runtime.h"
#ifndef FULLDTW
#define FULLDTW

#include "common.hpp"
#include "datatypes.hpp"
#include "data_dims.hpp"
#include <hip/hip_cooperative_groups.h>
#include <hip/amd_detail/host_defines.h>

// Cost function if reference deletions are permitted
// Cost = (r1-q)^2 + min(left, top, diag)
#ifndef NO_REF_DEL
#define COST_FUNCTION(q, r, left, top, diag) \
  FMA(FMA(SUB(r, q), SUB(r, q), 0.0f), 1.0f, \
      FIND_MIN(left, FIND_MIN(top, diag)))

// Cost function if reference deletions are not permitted
// Cost = cost_here + min(top, diag)
// (cannot move directly left to right, as this is a reference deletion)
#else
#define COST_FUNCTION(q, r, left, top, diag) \
  FMA(FMA(SUB(r, q), SUB(r, q), 0.0f), 1.0f, \
      FIND_MIN(top, diag))
#endif

// Compute SEGMENT_SIZE cells of the DTW matrix
// Computes a SEGMENT_SIZE portion of a row, for which the query value is query_val
// and the reference values are ref[]. To compute this, it is given
// the penalty_left and penalty_diag for the first cell, and then the penalty_here values
// of the previous row, used to compute the remaining cells.
// [penalty_diag] [penalties]
// [penalty_left] [penalties to compute]
// It uses temporary buffers in order to do an inline replacement of the penalty_here array.
template <typename idx_t, typename val_t>
__device__ __forceinline__ void
compute_segment(val_t &query_val,
                val_t (&ref)[SEGMENT_SIZE], val_t &penalty_left,
                val_t (&penalty_here)[SEGMENT_SIZE], val_t &penalty_diag,
                val_t (&penalty_temp)[2], idxt query_batch
#ifdef HIP_DEBUG
                ,
                idxt &wave, const idx_t &thread_id, idx_t block_id, idxt ref_batch
#endif
)
{
  penalty_temp[0] = penalty_here[0];

// The very first cell uses penalty_left and penalty_diag, and then the first value
// of penalty_here (representing the penalty of the cell directly above this one)
#ifdef HIP_DEBUG
  int column = ref_batch * REF_TILE_SIZE + thread_id * SEGMENT_SIZE;
  int row = query_batch * QUERY_BATCH_SIZE + wave - thread_id - 1;
  val_t penalty_top = penalty_here[0];
  // printf("[%0d,%0d,%0d]=(%.7f,%.7f,%.7f)\n", block_id, row, column, penalty_left, penalty_here[0], penalty_diag);
#endif
  penalty_here[0] = COST_FUNCTION(query_val, ref[0], penalty_left,
                                  penalty_here[0], penalty_diag);
#ifdef HIP_DEBUG
  // printf("[%0d,%0d,%0d]=%.7f,\n", block_id, row, column, penalty_here[0]);
  printf("[%0d,%0d,%0d]=(%.7f,%.7f,%.7f,%.7f)\n", block_id, row, column, penalty_here[0], penalty_left, penalty_top, penalty_diag);
#endif

#if ((SEGMENT_SIZE % 2) == 0)
  for (int i = 1; i < SEGMENT_SIZE - 2; i += 2)
  {
#else
  for (int i = 1; i < SEGMENT_SIZE - 1; i += 2)
  {
#endif
    // Updates the new penalty_here values for indices i and i+1
    penalty_temp[1] = penalty_here[i];
#ifdef HIP_DEBUG
    column = ref_batch * REF_TILE_SIZE + thread_id * SEGMENT_SIZE + i;
    penalty_top = penalty_here[i];
    // printf("[%0d,%0d,%0d]=(%.7f,%.7f,%.7f)\n", block_id, row, column, penalty_here[i - 1], penalty_here[i], penalty_temp[0]);
#endif
    penalty_here[i] =
        COST_FUNCTION(query_val, ref[i], penalty_here[i - 1],
                      penalty_here[i], penalty_temp[0]);
#ifdef HIP_DEBUG
    printf("[%0d,%0d,%0d]=(%.7f,%.7f,%.7f,%.7f)\n", block_id, row, column, penalty_here[i], penalty_here[i - 1], penalty_top, penalty_temp[0]);
    // printf("[%0d,%0d,%0d]=%.7f\n", block_id, row, column, penalty_here[i]);
#endif

    penalty_temp[0] = penalty_here[i + 1];
#ifdef HIP_DEBUG
    column = ref_batch * REF_TILE_SIZE + thread_id * SEGMENT_SIZE + i + 1;
    penalty_top = penalty_here[i + 1];
    // printf("[%0d,%0d,%0d]=(%.7f,%.7f,%.7f)\n", block_id, row, column, penalty_here[i], penalty_here[i + 1], penalty_temp[1]);
#endif
    penalty_here[i + 1] =
        COST_FUNCTION(query_val, ref[i + 1], penalty_here[i],
                      penalty_here[i + 1], penalty_temp[1]);
#ifdef HIP_DEBUG
    printf("[%0d,%0d,%0d]=(%.7f,%.7f,%.7f,%.7f)\n", block_id, row, column, penalty_here[i + 1], penalty_here[i], penalty_top, penalty_temp[1]);
    // printf("[%0d,%0d,%0d]=%.7f\n", block_id, row, column, penalty_here[i + 1]);
#endif
  }
#if ((SEGMENT_SIZE > 1) && ((SEGMENT_SIZE % 2) == 0))
#ifdef HIP_DEBUG
  column = ref_batch * REF_TILE_SIZE + thread_id * SEGMENT_SIZE + SEGMENT_SIZE - 1;
  penalty_top = penalty_here[SEGMENT_SIZE - 1];
  // printf("[%0d,%0d,%0d]=(%.7f,%.7f,%.7f)\n", block_id, row, column, penalty_here[SEGMENT_SIZE - 2], penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);
#endif
  penalty_here[SEGMENT_SIZE - 1] =
      COST_FUNCTION(query_val, ref[SEGMENT_SIZE - 1], penalty_here[SEGMENT_SIZE - 2],
                    penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);
#ifdef HIP_DEBUG
  printf("[%0d,%0d,%0d]=(%.7f,%.7f,%.7f,%.7f)\n", block_id, row, column, penalty_here[SEGMENT_SIZE - 1], penalty_here[SEGMENT_SIZE - 2], penalty_top, penalty_temp[0]);
  // printf("[%0d,%0d,%0d]=%.7f\n", block_id, row, column, penalty_here[SEGMENT_SIZE - 1]);
#endif
#endif
}

template <typename idx_t, typename val_t>
__global__ void DTW(val_t *ref, val_t *query, val_t *dist,
                    idx_t num_entries, val_t *device_last_row)
{
  // Indexing variables
  const idx_t block_id = blockIdx.x;
  const idx_t thread_id = threadIdx.x;

  // Penalties
  val_t penalty_temp[2];
  val_t penalty_here[SEGMENT_SIZE];
  val_t ref_segment[SEGMENT_SIZE];

#if REF_BATCH > 1
  // used to store last col of matrix (when reference is broken up into batches)
  __shared__ val_t penalty_last_col[QUERY_BATCH_SIZE];
  val_t last_col_penalty_shuffled;
#endif

#pragma unroll
  for (idxt query_batch = 0; query_batch < QUERY_BATCH; query_batch++)
  {
    // ~~~~ Initialization ~~~~
    val_t penalty_left = INFINITY;
    val_t penalty_diag = INFINITY;
    for (int i = 0; i < SEGMENT_SIZE; i++)
    {
      penalty_here[i] = INFINITY;
    }
    val_t query_val = INFINITY;
    val_t new_query_val =
        query[(block_id * QUERY_LEN) + (query_batch * QUERY_BATCH_SIZE) + thread_id];

#pragma unroll
    for (idxt i = 0; i < SEGMENT_SIZE; i++)
    {
      // The reference has been rearranged in ref for memory access optimizations
      // (Hence the striding by 'WARP_SIZE')
      ref_segment[i] = ref[thread_id + i * WARP_SIZE];
    }

// ~~~~ Special case initialization ~~~~
#if QUERY_BATCH > 1
    if (query_batch > 0)
    {
#pragma unroll
      for (idxt i = 0; i < SEGMENT_SIZE; i++)
      {
        penalty_here[i] =
            device_last_row[block_id * REF_LEN + thread_id + i * WARP_SIZE];
      }
      penalty_diag = __shfl_up(penalty_here[SEGMENT_SIZE - 1], 1);
      if (thread_id == WARP_SIZE_MINUS_ONE)
      {
        // When on query_batch > 0, we penalty_here is the last row of the submatrix above this one
        // The last element of this row will be the penalty_diag for the initial cell in the next submatrix
        // We save this value into dist for the next submatrix (we will overwrite last_row with our own values)
        dist[block_id] = penalty_here[SEGMENT_SIZE - 1];
      }
    }
#endif

    if (thread_id == 0)
    {
      // thread 0 will be using new_query_val right away (other threads will join in later waves)
      query_val = new_query_val;
      if (query_batch == 0)
      {
        // For the very first cell in the matrix, there is no penalty coming from the diagonal
        penalty_diag = 0.0f;
      }
      else
      {
        penalty_diag = INFINITY;
      }
    }

    // Fill in the matrix in wavefront parallel manner
    new_query_val = __shfl_down(new_query_val, 1);
    for (idxt wave = 1; wave <= NUM_WAVES; wave++)
    {
      // If thead_id is between (wave-QUERY_BATCH_SIZE) and (wave-1), it participates in this wave
      if (((wave - QUERY_BATCH_SIZE) <= thread_id) && (thread_id <= (wave - 1)))
      {
        compute_segment<idx_t, val_t>(query_val, ref_segment,
                                      penalty_left, penalty_here, penalty_diag,
                                      penalty_temp, query_batch
#ifdef HIP_DEBUG
                                      ,
                                      wave, thread_id, block_id, 0
#endif
        );
        penalty_diag = penalty_left;
      }

      // ~~~~ Update variables for next wave ~~~~
      query_val = __shfl_up(query_val, 1);
      // penalty_diag = penalty_left;
      penalty_left = __shfl_up(penalty_here[SEGMENT_SIZE - 1], 1);

      // ~~~~ Special case updates ~~~~
      // When wave is a multiple of WARP_SIZE (WARP_SIZE, 2*WARP_SIZE, ...)
      // we've processed through all the query values originally loaded into new_query_val
      // so we need to load the next set of query values.
      if ((wave & WARP_SIZE_MINUS_ONE) == 0)
      {
        new_query_val = query[(block_id * QUERY_LEN) + (query_batch * QUERY_BATCH_SIZE) + wave + thread_id];
      }

      if (thread_id == 0)
      {
        query_val = new_query_val;
        penalty_left = INFINITY;
      }
      new_query_val = __shfl_down(new_query_val, 1);

#if REF_BATCH > 1
      // Get the last column penalty from thread_id+1
      last_col_penalty_shuffled = __shfl_down(last_col_penalty_shuffled, 1);
      // If this is the last thread in the warp, use penalty_here[SEGMENT_SIZE - 1] instead
      if (thread_id == WARP_SIZE_MINUS_ONE)
      {
        last_col_penalty_shuffled = penalty_here[SEGMENT_SIZE - 1];
      }

      // If this wave is >= 2*warp-1, and is a multiple of WARP_SIZE_MINUS_ONE
      if ((wave >= TWICE_WARP_SIZE_MINUS_ONE) && ((wave & WARP_SIZE_MINUS_ONE) == WARP_SIZE_MINUS_ONE))
      {
        penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE) + thread_id] = last_col_penalty_shuffled;
      }
      else if ((wave >= NUM_WAVES_BY_WARP_SIZE) && (thread_id == WARP_SIZE_MINUS_ONE))
      {
        penalty_last_col[wave - WARP_SIZE] = penalty_here[SEGMENT_SIZE - 1];
      }

#endif
    }

// write last row to shared memory
#if QUERY_BATCH > 1
    for (idxt i = 0; i < SEGMENT_SIZE; i++)
    {
      device_last_row[block_id * REF_LEN + thread_id + i * WARP_SIZE] =
          penalty_here[i];
    }
#endif

    // For all ref batches > 0, except the last one
#if REF_BATCH > 1
    for (idxt ref_batch = 1; ref_batch < REF_BATCH_MINUS_ONE; ref_batch++)
    {
      // Initialize penalties
      penalty_left = INFINITY;
      if (query_batch == 0)
      {
        for (auto i = 0; i < SEGMENT_SIZE; i++)
        {
          penalty_here[i] = INFINITY;
        }
        penalty_diag = INFINITY;
      }
      else
      {
        for (auto i = 0; i < SEGMENT_SIZE; i++)
        {
          penalty_here[i] =
              device_last_row[block_id * REF_LEN + ref_batch * REF_TILE_SIZE +
                              thread_id + i * WARP_SIZE];
        }
        penalty_diag = __shfl_up(penalty_here[SEGMENT_SIZE - 1], 1);
        // MQ: Attempt and getting the correct diag element
        if (thread_id == 0)
        {
          penalty_diag = dist[block_id];
          // MQ: INTENTIONALLY BREAKING THIS FOR PRODUCING DEBUG PLOTS
          // penalty_diag = INFINITY;
        }
        if (thread_id == WARP_SIZE_MINUS_ONE)
        {
          dist[block_id] = penalty_here[SEGMENT_SIZE - 1];
        }
      }

      // Load next WARP_SIZE query values from memory into new_query_val buffer
      query_val = INFINITY;
      new_query_val = query[(block_id * QUERY_LEN) + (query_batch * QUERY_BATCH_SIZE) + thread_id];

      // Initialize first thread's chunk
      if (thread_id == 0)
      {
        query_val = new_query_val;
        penalty_left = penalty_last_col[0];
      }
      new_query_val = __shfl_down(new_query_val, 1);
      for (idxt i = 0; i < SEGMENT_SIZE; i++)
      {
        ref_segment[i] = ref[ref_batch * REF_TILE_SIZE + thread_id + i * WARP_SIZE];
      }
      // Calculate full matrix in wavefront parallel manner, multiple cells per thread
      for (idxt wave = 1; wave <= NUM_WAVES; wave++)
      {
        if (((wave - QUERY_BATCH_SIZE) <= thread_id) && (thread_id <= (wave - 1)))
        {
          compute_segment<idx_t, val_t>(
              query_val, ref_segment, penalty_left,
              penalty_here, penalty_diag, penalty_temp, query_batch
#ifdef HIP_DEBUG
              ,
              wave, thread_id, block_id, ref_batch
#endif
          );
          penalty_diag = penalty_left;
        }

        // new_query_val buffer is empty, reload
        // If wave is a multiple of WARP_SIZE
        // MQ: I don't know why it needs to load a new 'new_query_val' here...
        // MQ: Also, this goes beyond the length of the query...?
        if ((wave & WARP_SIZE_MINUS_ONE) == 0)
        {
          new_query_val = query[(block_id * QUERY_LEN) + wave + thread_id + (query_batch * QUERY_BATCH_SIZE)];
        }

        // Pass next query_value to each thread
        query_val = __shfl_up(query_val, 1);
        if (thread_id == 0)
        {
          query_val = new_query_val;
        }

        // MQ: I'm not sure what's happening here
        last_col_penalty_shuffled = __shfl_down(last_col_penalty_shuffled, 1);
        if (thread_id == WARP_SIZE_MINUS_ONE)
        {
          last_col_penalty_shuffled = penalty_here[RESULT_REG];
        }
        // If this wave is >= 2*warp-1, and is a multiple of WARP_SIZE_MINUS_ONE
        if ((wave >= TWICE_WARP_SIZE_MINUS_ONE) && ((wave & WARP_SIZE_MINUS_ONE) == WARP_SIZE_MINUS_ONE))
        {
          penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE) + thread_id] = last_col_penalty_shuffled;
        }
        else if ((wave >= NUM_WAVES_BY_WARP_SIZE) && (thread_id == WARP_SIZE_MINUS_ONE))
        {
          // MQ: Changed from (wave - TWICE_WARP_SIZE_MINUS_ONE) to (wave - WARP_SIZE), differing from
          // the main branch, idk if still correct, but was getting out of bounds errors
          penalty_last_col[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
        }

        new_query_val = __shfl_down(new_query_val, 1);
        penalty_left = __shfl_up(penalty_here[SEGMENT_SIZE - 1], 1);
        if (thread_id == 0)
        {
          penalty_left = penalty_last_col[wave];
        }
      }

// write last row to smem
#if QUERY_BATCH > 1
      for (idxt i = 0; i < SEGMENT_SIZE; i++)
      {
        device_last_row[block_id * REF_LEN + ref_batch * REF_TILE_SIZE +
                        thread_id + i * WARP_SIZE] = penalty_here[i];
      }
#endif
    }

    // Last sub-matrix (ref batch) calculation for this query batch
    // Initialize penalties
    penalty_left = INFINITY;
    if (query_batch == 0)
    {
      for (auto i = 0; i < SEGMENT_SIZE; i++)
      {
        penalty_here[i] = INFINITY;
      }
      penalty_diag = INFINITY;
    }
    else
    {
      for (auto i = 0; i < SEGMENT_SIZE; i++)
      {
        penalty_here[i] =
            // device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE +
            //                 thread_id + i * WARP_SIZE];
            device_last_row[block_id * REF_LEN + REF_BATCH_MINUS_ONE * REF_TILE_SIZE +
                            thread_id + i * WARP_SIZE];
      }
      penalty_diag = __shfl_up(penalty_here[SEGMENT_SIZE - 1], 1);
      // MQ: Attempt at getting the correct diag element
      if (thread_id == 0)
      {
        penalty_diag = dist[block_id];
        // MQ: INTENTIONALLY BREAKING THIS FOR PRODUCING DEBUG PLOTS
        // penalty_diag = INFINITY;
      }
    }

    // Load next WARP_SIZE query values from memory into new_query_val buffer
    query_val = INFINITY;
    new_query_val =
        query[(block_id * QUERY_LEN) + (query_batch * QUERY_BATCH_SIZE) + thread_id];

    // Initialize first thread's chunk
    if (thread_id == 0)
    {
      query_val = new_query_val;
      // MQ: INTENTIONALLY BREAKING THIS FOR PRODUCING DEBUG PLOTS
      penalty_left = penalty_last_col[0];
    }
    new_query_val = __shfl_down(new_query_val, 1);
    for (idxt i = 0; i < SEGMENT_SIZE; i++)
    {
      ref_segment[i] =
          ref[REF_BATCH_MINUS_ONE * REF_TILE_SIZE + thread_id + i * WARP_SIZE];
    }
    // Calculate full matrix in wavefront parallel manner, multiple cells per thread
    for (idxt wave = 1; wave <= NUM_WAVES; wave++)
    {
      // If thread 'thread_id' is participating in this wave, compute it's segment
      if (((wave - QUERY_BATCH_SIZE) <= thread_id) && (thread_id <= (wave - 1)))
      {
        compute_segment<idx_t, val_t>(
            query_val, ref_segment, penalty_left,
            penalty_here, penalty_diag, penalty_temp, query_batch
#ifdef HIP_DEBUG
            ,
            wave, thread_id, block_id, REF_BATCH_MINUS_ONE
#endif
        );
        // only update penalty_diag if you actually participated in the wave
        penalty_diag = penalty_left;
      }

      // If wave is a multiple of WARP_SIZE
      // MQ: I don't know why it needs to load a new 'new_query_val' here...
      // MQ: Also, this goes beyond the length of the query...?
      if ((wave & WARP_SIZE_MINUS_ONE) == 0)
      {
        new_query_val = query[(block_id * QUERY_LEN) + wave + thread_id + (query_batch * QUERY_BATCH_SIZE)];
      }

      // Pass next query_value to each thread
      query_val = __shfl_up(query_val, 1);
      if (thread_id == 0)
      {
        query_val = new_query_val;
      }
      new_query_val = __shfl_down(new_query_val, 1);

      // Transfer border cell info
      // penalty_diag = penalty_left;
      penalty_left = __shfl_up(penalty_here[SEGMENT_SIZE - 1], 1);
      if (thread_id == 0)
      {
        penalty_left = penalty_last_col[wave];
      }
    }

    // write last row to smem
#if QUERY_BATCH > 1
    for (idxt i = 0; i < SEGMENT_SIZE; i++)
    {
      device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + thread_id +
                      i * WARP_SIZE] = penalty_here[i];
    }

    // for (idxt i = 0; i < SEGMENT_SIZE; i++)
    // {
    //   device_last_row[block_id * REF_LEN + REF_BATCH_MINUS_ONE * REF_TILE_SIZE +
    //                   thread_id + i * WARP_SIZE] = penalty_here[i];
    // }
#endif

#endif

  } // end of the query_batch loop

  // return result
  if (thread_id == WARP_SIZE_MINUS_ONE)
  {
    dist[block_id] = penalty_here[SEGMENT_SIZE - 1];
  }
  return;
}
#endif