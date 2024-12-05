# ./src/include/normalizer.cu

- Found no inline PTX assembly (searched for 'asm')
- Found no hardcoded dependencies on warp size (searched for '32')
- Ignored the hipify-perl warnings below, as they still seemed to be successfully replaced with HIP equivalents

```bash
[mqueen@login1 include]$ hipify-perl normalizer.cu > normalizer.cpp
  warning: normalizer.cu:87: deprecated identifier "cudnnBatchNormMode_t" since CUDNN 9.0.0
  warning: normalizer.cu:87: deprecated identifier "CUDNN_BATCHNORM_SPATIAL" since CUDNN 9.0.0
  warning: normalizer.cu:95: deprecated identifier "cudnnDeriveBNTensorDescriptor" since CUDNN 9.0.0
  warning: normalizer.cu:98: deprecated identifier "cudnnBatchNormalizationForwardTraining" since CUDNN 9.0.0
```

  - Changed hipDNN calls to miopen calls (The initial hipification replaced cuDNN with hipDNN, but the actual recommended replacement is miopen)
  - Added a ‘strides’ parameter to miopenSetTensorDescriptor (followed ChatGPT recommended values)

# ./src/include/DTW.cu

- Found no inline PTX assembly (searched for 'asm')
- Found no hardcoded dependencies on warp size (searched for '32')


- Replaced '__shfl_down_sync' with '__shfl_down' and dropped the sync mask parameter
- Replaced '__shfl_up_sync' with '__shfl_up' and dropped the sync mask parameter


```bash
[mqueen@login1 include]$ hipify-perl DTW.cu > DTW.cpp
  warning: DTW.cu:189: unsupported device function "__shfl_down_sync":     new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
  warning: DTW.cu:216: unsupported device function "__shfl_up_sync":         min_segment = __shfl_up_sync((ALL), min_segment, 1);
  warning: DTW.cu:234: unsupported device function "__shfl_up_sync":       query_val = __shfl_up_sync(ALL, query_val, 1);
  warning: DTW.cu:240: unsupported device function "__shfl_up_sync":       penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
  warning: DTW.cu:245: unsupported device function "__shfl_down_sync":       new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
  warning: DTW.cu:252: unsupported device function "__shfl_down_sync":           __shfl_down_sync(ALL, last_col_penalty_shuffled, 1);
  warning: DTW.cu:311: unsupported device function "__shfl_down_sync":         min_segment = __shfl_down_sync((ALL), min_segment, 31);
  warning: DTW.cu:344: unsupported device function "__shfl_down_sync":       new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
  warning: DTW.cu:372: unsupported device function "__shfl_up_sync":         query_val = __shfl_up_sync(ALL, query_val, 1);
  warning: DTW.cu:378: unsupported device function "__shfl_down_sync":             __shfl_down_sync(ALL, last_col_penalty_shuffled, 1);
  warning: DTW.cu:389: unsupported device function "__shfl_down_sync":         new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
  warning: DTW.cu:393: unsupported device function "__shfl_up_sync":         penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
  warning: DTW.cu:412: unsupported device function "__shfl_up_sync":             min_segment = __shfl_up_sync((ALL), min_segment, 1);
  warning: DTW.cu:438: unsupported device function "__shfl_down_sync":       min_segment = __shfl_down_sync((ALL), min_segment, 31);
  warning: DTW.cu:480: unsupported device function "__shfl_down_sync":     new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
  warning: DTW.cu:508: unsupported device function "__shfl_up_sync":       query_val = __shfl_up_sync(ALL, query_val, 1);
  warning: DTW.cu:513: unsupported device function "__shfl_down_sync":       new_query_val = __shfl_down_sync(ALL, new_query_val, 1);
  warning: DTW.cu:517: unsupported device function "__shfl_up_sync":       penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);
  warning: DTW.cu:540: unsupported device function "__shfl_up_sync":           min_segment = __shfl_up_sync((ALL), min_segment, 1);
  ```

  # ./src/main.cu

  - Found no inline PTX assembly (searched for 'asm')
  - Found no hardcoded dependencies on warp size (searched for '32')
  - hipify-perl returned no errors or warnings


  ```bash
  [mqueen@login1 src]$ hipify-perl main.cu > main.cpp
  ```

  - Updated to include (the new) normalizer.cpp instead of normalizer.cu

  # ./src/ru_main.cu

  - Found no inline PTX assembly (searched for 'asm')
  - Found no hardcoded dependencies on warp size (searched for '32')
  - hipify-perl returned no errors or warnings

  ```bash
  [mqueen@login1 src]$ hipify-perl ru_main.cu > ru_main.cpp
  ```

  # ./src/include/common.hpp

  - Replaced `#include <cuda_fp16.h>` with `#include <hip/hip_fp16.h>`
  - (UNDID) Replaced `#define COMON_HPP` with `#define COMMON_HPP` (not a hipification thing, but a warning I got when building)

  # ./src/include/DTW.hpp

  - Replaced `#include <cuda_fp16.h>` with `#include <hip/hip_fp16.h>`
  - Replaced `#include "DTW.cu"` with `#include "DTW.cpp"`

  # ./src/include/hpc_helpers.hpp

  - Updated macros to use HIP versions instead of CUDA versions
  - Updated macros to check for __HIPCC__ rather than __CUDACC__
  - The HIP functions specify that the return values are [[nodiscard]] – wrapped the calls in ASSERTs to check for success

  # ./src/include/common.hpp

  - Replaced the cuda fused-multiply-add  intrinsic with the HIP version (replaced __fmaf_ieee_rn with __fmaf_rn). ChatGPT was hilariously bad at this one and made up a number of non-existent fma functions. Going right to the [actual documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html)
 was far more effective! 
