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
#ifndef COMMON_HPP
#define COMMON_HPP

//...............global variables..........................//
typedef float value_ht;
#define FLOAT2HALF(a) a
#define HALF2FLOAT(a) a
#define FIND_MIN(a, b) min(a, b)
#define FIND_MAX(a, b) max(a, b)
#define FMA(a, b, c) __fmaf_rn(a, b, c)
#define ADD(a, b) (a + b)
#define SUB(a, b) (a - b) // make sure b is power of 2
#define SQRT(a) sqrtf(a)  // a is to be float
#define GT(a, b) (a > b)

#define BLOCK_NUM 1024 // Multiple of 104 (CUs)
#define STREAM_NUM 8 //16

#endif
