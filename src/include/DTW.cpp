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
compute_segment(idxt &wave, const idx_t &thread_id, val_t &query_val,
                val_t (&ref)[SEGMENT_SIZE], val_t &penalty_left,
                val_t (&penalty_here)[SEGMENT_SIZE], val_t &penalty_diag,
                val_t (&penalty_temp)[2], idxt query_batch)
{
  penalty_temp[0] = penalty_here[0];
  // (thread_id != (wave - 1)) The wave counting starts at 1
  // When thread_id == (wave - 1) it's the first time thread_id is doing work, and it
  // is calculating matrix element in row 0, column thread_id (when SEGMENT_SIZE is 1, for simplicity)
  // This will require special handling. Otherwise, the thread is working on an interior matrix element
  // When query_batch > 0, we are also working on interior matrix elements.
  if ((thread_id != (wave - 1)) || (query_batch))
  {
    // The very first cell uses penalty_left and penalty_diag, and then the first value
    // of penalty_here (representing the penalty of the cell directly above this one)
    penalty_here[0] = COST_FUNCTION(query_val, ref[0], penalty_left,
                                    penalty_here[0], penalty_diag);

#if ((SEGMENT_SIZE % 2) == 0)
    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2)
    {
#else
    for (int i = 1; i < SEGMENT_SIZE - 1; i += 2)
    {
#endif
      // Updates the new penalty_here values for indices i and i+1
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          COST_FUNCTION(query_val, ref[i], penalty_here[i - 1],
                        penalty_here[i], penalty_temp[0]);

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref[i + 1], penalty_here[i],
                        penalty_here[i + 1], penalty_temp[1]);
    }
#if ((SEGMENT_SIZE > 1) && ((SEGMENT_SIZE % 2) == 0))
    penalty_here[SEGMENT_SIZE - 1] =
        COST_FUNCTION(query_val, ref[SEGMENT_SIZE - 1], penalty_here[SEGMENT_SIZE - 2],
                      penalty_here[SEGMENT_SIZE - 1], penalty_temp[0]);
#endif
  }

  // Else, query_batch==0 and thread_id==(wave-1)
  // This is the first time thread_id has entered the wavefront,
  // and will be computing a matrix element in the first row (row 0, column thread_id, when SEGMENT_SIZE==1)
  // MQ: Isn't this already handled by the calling function, in the way penalty_here is initialized?
  else
  {
    penalty_here[0] = COST_FUNCTION(query_val, ref[0], penalty_left,
                                    INFINITY, penalty_diag);
    // if (thread_id == 0 && wave == 1)
    // {
    //   penalty_here[0] = COST_FUNCTION(query_val, ref[0], 0, 0, 0);
    // }

#if ((SEGMENT_SIZE % 2) == 0)
    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2)
    {
#else
    for (int i = 1; i < SEGMENT_SIZE - 1; i += 2)
    {
#endif
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] =
          COST_FUNCTION(query_val, ref[i], penalty_here[i - 1], INFINITY, INFINITY);

      penalty_temp[0] = penalty_here[i + 1];
      penalty_here[i + 1] =
          COST_FUNCTION(query_val, ref[i + 1], penalty_here[i], INFINITY, INFINITY);
    }
#if ((SEGMENT_SIZE > 1) && ((SEGMENT_SIZE % 2) == 0))
    penalty_here[SEGMENT_SIZE - 1] =
        COST_FUNCTION(query_val, ref[SEGMENT_SIZE - 1], penalty_here[SEGMENT_SIZE - 2],
                      INFINITY, INFINITY);
#endif
  }
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
  val_t min_segment = INFINITY; // finds min of segment for sDTW
  val_t ref_segment[SEGMENT_SIZE];

#if REF_BATCH > 1
  __shared__ val_t penalty_last_col[QUERY_BATCH_SIZE];
  val_t last_col_penalty_shuffled; // used to store last col of matrix (when reference is broken up into batches)
#endif

#pragma unroll
  for (idxt query_batch = 0; query_batch < QUERY_BATCH; query_batch++)
  {
    val_t penalty_left = INFINITY;
    val_t penalty_diag = INFINITY;
    if (query_batch == 0)
    {
      penalty_diag = 0.0f;
    }

    for (int i = 0; i < SEGMENT_SIZE; i++)
    {
      penalty_here[i] = INFINITY; // For global alignment
    }
    val_t query_val = INFINITY;
    val_t new_query_val =
        query[(block_id * QUERY_LEN) + (query_batch * QUERY_BATCH_SIZE) + thread_id];
    // Initialize first thread's query_val
    if (thread_id == 0)
    {
      query_val = new_query_val;
    }
    new_query_val = __shfl_down(new_query_val, 1);
#pragma unroll
    for (idxt i = 0; i < SEGMENT_SIZE; i++)
    {
      // The reference has been rearranged in ref for memory access optimizations
      // (Hence the striding by 'WARP_SIZE')
      ref_segment[i] = ref[thread_id + i * WARP_SIZE];
#if QUERY_BATCH > 1
      if (query_batch > 0)
      {
        penalty_here[i] =
            device_last_row[block_id * REF_LEN + thread_id + i * WARP_SIZE];
      }
#endif
    }
    // MQ: Attempt and getting the correct diag element
    if (thread_id == WARP_SIZE_MINUS_ONE and query_batch > 0)
    {
      // Save the corner element, it will be used as the penalty_diag for the next batch
      dist[block_id] = penalty_here[SEGMENT_SIZE - 1];
      // printf("(176)dist[%0d]=%.2f\n", block_id, penalty_here[SEGMENT_SIZE - 1]);
    }
    // calculate full matrix in wavefront parallel manner, multiple cells per thread
    for (idxt wave = 1; wave <= NUM_WAVES; wave++)
    {
      // If thead_id is between (wave-QUERY_BATCH_SIZE) and (wave-1), it participates in this wave
      if (((wave - QUERY_BATCH_SIZE) <= thread_id) && (thread_id <= (wave - 1)))
      {
        compute_segment<idx_t, val_t>(wave, thread_id, query_val, ref_segment,
                                      penalty_left, penalty_here, penalty_diag,
                                      penalty_temp, query_batch);
#ifdef HIP_DEBUG
        for (auto i = 0; i < SEGMENT_SIZE; i++)
        {
          int column = thread_id * SEGMENT_SIZE + i;
          int row = query_batch * QUERY_BATCH_SIZE + wave - thread_id - 1;
          printf("[%0d,%0d,%0d]=%.2f\n", block_id, row, column, penalty_here[i]);
        }
#endif
      }

      // new_query_val buffer is empty, reload
      // If wave is a multiple of WARP_SIZE
      // MQ: I don't know why it needs to load a new 'new_query_val' here...
      // MQ: Also, this goes beyond the length of the query...?
      if ((wave & WARP_SIZE_MINUS_ONE) == 0) // MQ: I think this is the last wave? maybe??
      {
        new_query_val = query[(block_id * QUERY_LEN) + (query_batch * QUERY_BATCH_SIZE) + wave + thread_id];
      }

      // Shuffle up query_value and penalties for each thread
      query_val = __shfl_up(query_val, 1);
      penalty_diag = penalty_left;
      penalty_left = __shfl_up(penalty_here[SEGMENT_SIZE - 1], 1);

      // Edge case adjustments for thread 0
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
      // Find min of segment and then shuffle up for sDTW
      if ((wave >= QUERY_BATCH_SIZE) && (query_batch == (QUERY_BATCH - 1)))
      {
        if (wave == NUM_WAVES)
        {
          min_segment = penalty_here[SEGMENT_SIZE - 1];
        }
      }
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
      penalty_diag = INFINITY;
      if (query_batch == 0)
      {
        for (auto i = 0; i < SEGMENT_SIZE; i++)
        {
          penalty_here[i] = INFINITY;
        }
      }
      else
      {
        for (auto i = 0; i < SEGMENT_SIZE; i++)
        {
          penalty_here[i] =
              device_last_row[block_id * REF_LEN + ref_batch * REF_TILE_SIZE +
                              thread_id + i * WARP_SIZE];
        }
        // MQ: Attempt and getting the correct diag element
        if (thread_id == 0)
        {
          penalty_diag = dist[block_id];
          // printf("(284)penalty_diag=%.2f\n", penalty_diag);
        }
        if (thread_id == WARP_SIZE_MINUS_ONE)
        {
          dist[block_id] = penalty_here[SEGMENT_SIZE - 1];
          // printf("(289)dist[%0d]=%.2f\n", block_id, penalty_here[SEGMENT_SIZE - 1]);
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
              wave, thread_id, query_val, ref_segment, penalty_left,
              penalty_here, penalty_diag, penalty_temp, query_batch);
#ifdef HIP_DEBUG
          for (auto i = 0; i < SEGMENT_SIZE; i++)
          {
            int column = ref_batch * REF_TILE_SIZE + thread_id * SEGMENT_SIZE + i;
            int row = query_batch * QUERY_BATCH_SIZE + (wave - thread_id - 1);
            printf("[%0d,%0d,%0d]=%.2f\n", block_id, row, column, penalty_here[i]);
          }
#endif
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
          // the main branch, idk if still correct, but was getting oom errors
          penalty_last_col[(wave - WARP_SIZE)] = penalty_here[RESULT_REG];
        }

        new_query_val = __shfl_down(new_query_val, 1);

        // Transfer border cell info
        penalty_diag = penalty_left;
        penalty_left = __shfl_up(penalty_here[SEGMENT_SIZE - 1], 1);
        if (thread_id == 0)
        {
          penalty_left = penalty_last_col[wave];
        }

        // Find min of segment and then shuffle up for sDTW
        if ((wave >= QUERY_BATCH_SIZE) && (query_batch == (QUERY_BATCH - 1)))
        {
          if (wave == NUM_WAVES)
          {
            min_segment = penalty_here[SEGMENT_SIZE - 1];
          }
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
    penalty_diag = INFINITY;
    if (query_batch == 0)
    {
      for (auto i = 0; i < SEGMENT_SIZE; i++)
      {
        penalty_here[i] = INFINITY;
      }
    }
    else
    {
      for (auto i = 0; i < SEGMENT_SIZE; i++)
      {
        penalty_here[i] =
            device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE +
                            thread_id + i * WARP_SIZE];
      }
      // MQ: Attempt at getting the correct diag element
      if (thread_id == 0)
      {
        penalty_diag = dist[block_id];
        // printf("(413)(thread %0d) penalty_diag=%.2f\n", thread_id, penalty_diag);
      }
      else
      {
        penalty_diag = device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + (thread_id - 1)];
        // printf("(418)(thread %0d) penalty_diag=%.2f\n", thread_id, penalty_diag);
      }

      if (thread_id == WARP_SIZE_MINUS_ONE)
      {
        dist[block_id] = penalty_here[SEGMENT_SIZE - 1];
        // printf("(418)dist[%0d]=%.2f\n", block_id, penalty_here[SEGMENT_SIZE - 1]);
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
#ifdef HIP_DEBUG
        int column = REF_BATCH_MINUS_ONE * REF_TILE_SIZE + thread_id * SEGMENT_SIZE;
        int row = query_batch * QUERY_BATCH_SIZE + (wave - thread_id - 1);
        if (column == 65 && row == 64)
        {
          printf("(65,64)penalty_diag=%.2f, penalty_left=%.2f, penalty_here[0]=%.2f, q=%.2f, r=%.2f, thread%0d\n",
                 penalty_diag, penalty_left, penalty_here[0], query_val, ref_segment[0], thread_id);
        }
#endif
        compute_segment<idx_t, val_t>(wave, thread_id, query_val, ref_segment,
                                      penalty_left, penalty_here, penalty_diag,
                                      penalty_temp, query_batch);
#ifdef HIP_DEBUG
        for (auto i = 0; i < SEGMENT_SIZE; i++)
        {
          int column = REF_BATCH_MINUS_ONE * REF_TILE_SIZE + thread_id * SEGMENT_SIZE + i;
          int row = query_batch * QUERY_BATCH_SIZE + (wave - thread_id - 1);
          printf("[%0d,%0d,%0d]=%.2f\n", block_id, row, column, penalty_here[i]);
        }
#endif
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

      // Find min of segment and then shuffle up for sDTW
      if ((wave >= QUERY_BATCH_SIZE) && (query_batch == (QUERY_BATCH - 1)))
      {
        if (wave == (NUM_WAVES))
          min_segment = penalty_here[SEGMENT_SIZE - 1];
      }
    }
    // write last row to smem
#if QUERY_BATCH > 1
    for (idxt i = 0; i < SEGMENT_SIZE; i++)
    {
      device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + thread_id +
                      i * WARP_SIZE] = penalty_here[i];
    }
#endif

#endif

  } // end of the query_batch loop

  // return result
  if (thread_id == WARP_SIZE_MINUS_ONE)
  {
    dist[block_id] = min_segment;
    // printf("dist[%0d]=%.7f\n", block_id, min_segment);
    // dist[block_id] = penalty_here[SEGMENT_SIZE-1]; // For global alignment?

#ifdef HIP_DEBUG
    printf("min_segment=%0f, tid=%0d, blockid=%0d\n", min_segment, thread_id,
           block_id);
    printf("penalty_temp[0]=%.2f, penalty_temp[1]=%.2f\n", penalty_temp[0], penalty_temp[1]);
    for (auto i = 0; i < SEGMENT_SIZE; i++)
    {
      printf("penalty_here[%0d]=%.2f\n", i, penalty_here[i]);
    }
#endif
  }
  return;
}
#endif