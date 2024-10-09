#ifndef NORMALIZER
#define NORMALIZER

#include "common.hpp"
#include <hip/hip_runtime.h>
#include <miopen.h>

#define CUDNN_ASSERT(func)                                                     \
  {                                                                            \
    miopenStatus_t e = (func);                                                  \
    std::cout << "\ncuDTW::  cuDNN Normalizer returned: "                      \
              << miopenGetErrorString(e) << "\n";                               \
  }

// normalizer class
class normalizer {
public:
  void normalize(raw_t *raw_squiggle_array, index_t num_reads, index_t length);
  normalizer(idxt NUM_READS); // CUDNN normalizer
  ~normalizer();
  void print_normalized_query(raw_t *raw_array, index_t NUM_READS,
                              std::vector<std::string> &read_ids);

private:
  float *bnScale, *bnBias, *bnScale_h, *bnBias_h;
  idxt NUM_READS;
  float alpha[1] = {1};
  float beta[1] = {0.0};
};

void normalizer::print_normalized_query(raw_t *raw_array, index_t NUM_READS,
                                        std::vector<std::string> &read_ids) {
  std::cout << "Normalized query:\n";
  for (index_t i = 0; i < NUM_READS; i++) {
    std::cout << "cuDTW:: " << read_ids[i] << "\n";
    for (index_t j = 0; j < QUERY_LEN; j++) {
      std::cout << raw_array[(i * QUERY_LEN + j)] << ",";
    }
    std::cout << "\n";
  }
  std::cout << "\n=================\n";
}
normalizer::~normalizer() {
  hipFree(bnScale);
  hipFree(bnBias);
  hipHostFree(bnScale_h);
  hipHostFree(bnBias_h);
}
normalizer::normalizer(idxt NUM_READS) {

  // create scale and bias vectors
  hipHostMalloc(&bnScale_h, (QUERY_LEN * sizeof(float) * NUM_READS));
  hipHostMalloc(&bnBias_h, (QUERY_LEN * sizeof(float) * NUM_READS));
  for (int i = 0; i < QUERY_LEN * NUM_READS; i++) {
    bnScale_h[i] = 1.0f;
    bnBias_h[i] = 0.0f;
  }
  hipMalloc(&bnScale, (QUERY_LEN * sizeof(float) * NUM_READS));
  hipMalloc(&bnBias, (QUERY_LEN * sizeof(float) * NUM_READS));

  hipMemcpyAsync(bnScale,
                  &bnScale_h[0], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
                  sizeof(float) * NUM_READS * QUERY_LEN,
                  hipMemcpyHostToDevice);
  hipMemcpyAsync(bnBias,
                  &bnBias_h[0], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
                  sizeof(float) * NUM_READS * QUERY_LEN,
                  hipMemcpyHostToDevice);
}

__inline__ void normalizer::normalize(raw_t *raw_squiggle_array,
                                      index_t num_reads, index_t length) {

  int c = num_reads, h = length; // nchw format for cudnn

  raw_t *x; // output,input array

  hipMalloc(&x, (sizeof(raw_t) * c * h));

  hipMemcpy(x, &raw_squiggle_array[0], (sizeof(raw_t) * c * h),
             hipMemcpyHostToDevice);

  miopenHandle_t handle_;
  miopenCreate(&handle_);
  miopenDataType_t dtype = miopenFloat;
  miopenTensorLayout_t format = miopenTensorNCHW;
  miopenBatchNormMode_t mode = miopenBNSpatial;

  // descriptors
  miopenTensorDescriptor_t x_desc, bnScaleBiasMeanVarDesc;
  miopenCreateTensorDescriptor(&x_desc);
  int dimensions[4] = {1, c, h, 1};
  int strides[4];
  strides[3] = 1;                         // Stride for the last dimension (W)
  strides[2] = dimensions[3] * strides[3]; // Stride for the height (H)
  strides[1] = dimensions[2] * strides[2]; // Stride for the channels (C)
  strides[0] = dimensions[1] * strides[1]; // Stride for the batch (N)
  miopenSetTensorDescriptor(x_desc, dtype, 4, dimensions, strides);

  miopenCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
  miopenDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc, x_desc, mode);

  // normalize
  miopenBatchNormalizationForwardTraining(
      handle_, mode, alpha, beta, x_desc, x, x_desc, x, bnScaleBiasMeanVarDesc,
      bnScale, bnBias, 1.0 / (1.0 + h), NULL, NULL, 0.0001f, NULL, NULL);

  hipMemcpy(

      &raw_squiggle_array[0], x, (sizeof(raw_t) * c * h),
      hipMemcpyDeviceToHost);

  // std::cout << "cudnn normalized output:\n";
  // for (uint64_t i = 0; i < (c * h); i++) {

  //   std::cout << raw_squiggle_array[i] << ",";
  // }
  // miopenDestroy(handle_);
  return;
}

#endif