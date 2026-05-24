// Copyright © 2023-2025 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"   
#include "mlx/backend/metal/kernels/quantized_utils.h"    
#include "mlx/backend/metal/kernels/quantized.h"
#include "mlx/backend/metal/kernels/quantized_zigzag.h"


template [[host_name("zigzag_qmv_dense_half_gs_64_b_4_nsg_4_tg_4")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 4, 4, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]); 

template [[host_name("zigzag_qmv_dense_half_gs_64_b_4_nsg_2_tg_2")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 4, 2, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]); 

template [[host_name("zigzag_qmv_dense_half_gs_64_b_4_nsg_4_tg_2")]]                                       
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 4, 4, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);                            
                                                                                                            
template [[host_name("zigzag_qmv_dense_half_gs_64_b_4_nsg_4_tg_8")]]                                       
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 4, 4, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);                                     
                                                                                                            
template [[host_name("zigzag_qmv_dense_half_gs_64_b_4_nsg_2_tg_4")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 4, 2, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);                                      
                                                                                                            
template [[host_name("zigzag_qmv_dense_half_gs_64_b_4_nsg_8_tg_2")]]                                       
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 4, 8, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]); 


template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_4_tg_4")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 4, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]); 

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_2_tg_2")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 2, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]); 

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_4_tg_2")]]                                       
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 4, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);           

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_4_tg_1")]]                                       
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 4, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);                   
                                                                                                            
template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_4_tg_8")]]                                       
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 4, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);                                     
                                                                                                            
template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_2_tg_4")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 2, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);                                      
                                                                                                            
template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_8_tg_2")]]                                       
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 8, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]); 

// nsg 1, tg 4
template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_1_tg_4")]]                                       
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 1, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]); 

// nsg 2, tg 1
template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_2_tg_1")]]                                       
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 2, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

// nsg 2, tg 8
template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_2_tg_8")]]                                       
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 2, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

// nsg 8, tg 1
template [[host_name("zigzag_qmv_sparse_half_gs_64_b_4_nsg_8_tg_1")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 4, 8, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);


///////////////////////////////////////////////////////////////////////////////
// B3 extra-parallelism instantiations: tg=16 for bits=4/6/8.
// Goal: see if more cross-TG parallelism helps at small CD shapes.
///////////////////////////////////////////////////////////////////////////////

#define ZIGZAG_DENSE_INST(BITS, NSG, TG) \
template [[host_name("zigzag_qmv_dense_half_gs_64_b_" #BITS "_nsg_" #NSG "_tg_" #TG)]] \
[[kernel]] void zigzag_qmv_dense_impl<half, 64, BITS, NSG, TG>( \
    device const uint32_t*      w_zz      [[buffer(0)]], \
    device const half*          x         [[buffer(1)]], \
    device const half*          scales_zz [[buffer(2)]], \
    device const half*          biases_zz [[buffer(3)]], \
    device atomic<float>*       out       [[buffer(4)]], \
    constant const int&         K         [[buffer(5)]], \
    uint3  tid        [[threadgroup_position_in_grid]], \
    uint   simd_gid   [[simdgroup_index_in_threadgroup]], \
    uint   simd_lid   [[thread_index_in_simdgroup]]);

#define ZIGZAG_SPARSE_INST(BITS, NSG, TG) \
template [[host_name("zigzag_qmv_sparse_half_gs_64_b_" #BITS "_nsg_" #NSG "_tg_" #TG)]] \
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, BITS, NSG, TG>( \
    device const uint32_t*      w_zz      [[buffer(0)]], \
    device const half*          x         [[buffer(1)]], \
    device const half*          scales_zz [[buffer(2)]], \
    device const half*          biases_zz [[buffer(3)]], \
    device const int*           sparse_indices [[buffer(4)]], \
    device atomic<float>*       out       [[buffer(5)]], \
    constant const int&         K         [[buffer(6)]], \
    uint3  tid        [[threadgroup_position_in_grid]], \
    uint   simd_gid   [[simdgroup_index_in_threadgroup]], \
    uint   simd_lid   [[thread_index_in_simdgroup]]);

ZIGZAG_DENSE_INST(8, 2, 16)
ZIGZAG_DENSE_INST(8, 4, 16)
ZIGZAG_SPARSE_INST(8, 2, 16)
ZIGZAG_SPARSE_INST(8, 4, 16)
ZIGZAG_SPARSE_INST(8, 1, 16)

ZIGZAG_DENSE_INST(6, 2, 16)
ZIGZAG_DENSE_INST(6, 4, 16)
ZIGZAG_SPARSE_INST(6, 2, 16)
ZIGZAG_SPARSE_INST(6, 4, 16)
ZIGZAG_SPARSE_INST(6, 1, 16)

ZIGZAG_DENSE_INST(4, 2, 16)
ZIGZAG_DENSE_INST(4, 4, 16)
ZIGZAG_SPARSE_INST(4, 2, 16)
ZIGZAG_SPARSE_INST(4, 4, 16)
ZIGZAG_SPARSE_INST(4, 1, 16)

#undef ZIGZAG_DENSE_INST
#undef ZIGZAG_SPARSE_INST


///////////////////////////////////////////////////////////////////////////////
// bits=8 dense instantiations (A2 — same (nsg, tg) sweep as bits=4 dense).
///////////////////////////////////////////////////////////////////////////////

template [[host_name("zigzag_qmv_dense_half_gs_64_b_8_nsg_4_tg_4")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 8, 4, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_8_nsg_2_tg_2")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 8, 2, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_8_nsg_4_tg_2")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 8, 4, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_8_nsg_4_tg_8")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 8, 4, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_8_nsg_2_tg_4")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 8, 2, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_8_nsg_8_tg_2")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 8, 8, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);


///////////////////////////////////////////////////////////////////////////////
// bits=8 sparse instantiations (A2 — same (nsg, tg) sweep as bits=4 sparse).
///////////////////////////////////////////////////////////////////////////////

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_4_tg_4")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 4, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_2_tg_2")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 2, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_4_tg_2")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 4, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_4_tg_1")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 4, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_4_tg_8")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 4, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_2_tg_4")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 2, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_8_tg_2")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 8, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_1_tg_4")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 1, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_2_tg_1")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 2, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_2_tg_8")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 2, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_8_nsg_8_tg_1")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 8, 8, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);


///////////////////////////////////////////////////////////////////////////////
// bits=6 dense instantiations (A3 — same (nsg, tg) sweep as bits=4 dense).
///////////////////////////////////////////////////////////////////////////////

template [[host_name("zigzag_qmv_dense_half_gs_64_b_6_nsg_4_tg_4")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 6, 4, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_6_nsg_2_tg_2")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 6, 2, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_6_nsg_4_tg_2")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 6, 4, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_6_nsg_4_tg_8")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 6, 4, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_6_nsg_2_tg_4")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 6, 2, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_dense_half_gs_64_b_6_nsg_8_tg_2")]]
[[kernel]] void zigzag_qmv_dense_impl<half, 64, 6, 8, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device atomic<float>*       out       [[buffer(4)]],
    constant const int&         K         [[buffer(5)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);


///////////////////////////////////////////////////////////////////////////////
// bits=6 sparse instantiations (A3 — same (nsg, tg) sweep as bits=4 sparse).
///////////////////////////////////////////////////////////////////////////////

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_4_tg_4")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 4, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_2_tg_2")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 2, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_4_tg_2")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 4, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_4_tg_1")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 4, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_4_tg_8")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 4, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_2_tg_4")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 2, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_8_tg_2")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 8, 2>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_1_tg_4")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 1, 4>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_2_tg_1")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 2, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_2_tg_8")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 2, 8>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

template [[host_name("zigzag_qmv_sparse_half_gs_64_b_6_nsg_8_tg_1")]]
[[kernel]] void zigzag_qmv_sparse_impl<half, 64, 6, 8, 1>(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const half*          x         [[buffer(1)]],
    device const half*          scales_zz [[buffer(2)]],
    device const half*          biases_zz [[buffer(3)]],
    device const int*           sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]);

///////////////////////////////////////////////////////////////////////////////
// zigzag_qmv_fast_impl instantiations (B3 — optimized for ≤4 MB shapes).
// Format: zigzag_qmv_fast_half_gs_{group_size}_b_{bits}_nsg_{N}_rps_{R}
// where N * R == group_size and R % 4 == 0.
///////////////////////////////////////////////////////////////////////////////

#define ZIGZAG_FAST_INST(BITS, NSG, RPS) \
template [[host_name("zigzag_qmv_fast_half_gs_64_b_" #BITS "_nsg_" #NSG "_rps_" #RPS)]] \
[[kernel]] void zigzag_qmv_fast_impl<half, 64, BITS, NSG, RPS>( \
    device const uint32_t*      w_zz      [[buffer(0)]], \
    device const half*          x         [[buffer(1)]], \
    device const half*          scales_zz [[buffer(2)]], \
    device const half*          biases_zz [[buffer(3)]], \
    device float*               out       [[buffer(4)]], \
    constant const int&         K         [[buffer(5)]], \
    uint3  tid        [[threadgroup_position_in_grid]], \
    uint   simd_gid   [[simdgroup_index_in_threadgroup]], \
    uint   simd_lid   [[thread_index_in_simdgroup]]);

// bits=8
ZIGZAG_FAST_INST(8, 16, 4)
ZIGZAG_FAST_INST(8,  8, 8)
ZIGZAG_FAST_INST(8,  4, 16)
ZIGZAG_FAST_INST(8,  2, 32)

// bits=6
ZIGZAG_FAST_INST(6, 16, 4)
ZIGZAG_FAST_INST(6,  8, 8)
ZIGZAG_FAST_INST(6,  4, 16)
ZIGZAG_FAST_INST(6,  2, 32)

// bits=4
ZIGZAG_FAST_INST(4, 16, 4)
ZIGZAG_FAST_INST(4,  8, 8)
ZIGZAG_FAST_INST(4,  4, 16)
ZIGZAG_FAST_INST(4,  2, 32)

#undef ZIGZAG_FAST_INST


///////////////////////////////////////////////////////////////////////////////
// mskip_qmv_impl instantiations (C2 — M-major mask-aware M-skip GEMV).
// Format: mskip_qmv_half_gs_64_b_{bits}_nsg_{N}
///////////////////////////////////////////////////////////////////////////////

#define MSKIP_QMV_INST(BITS, NSG) \
template [[host_name("mskip_qmv_half_gs_64_b_" #BITS "_nsg_" #NSG)]] \
[[kernel]] void mskip_qmv_impl<half, 64, BITS, NSG>( \
    const device uint32_t*    w           [[buffer(0)]], \
    const device half*        scales      [[buffer(1)]], \
    const device half*        biases      [[buffer(2)]], \
    const device half*        x           [[buffer(3)]], \
    const device uchar*       mask        [[buffer(4)]], \
    device half*              y           [[buffer(5)]], \
    constant const int&       in_vec_size [[buffer(6)]], \
    constant const int&       out_vec_size [[buffer(7)]], \
    uint3  tid       [[threadgroup_position_in_grid]], \
    uint   simd_gid  [[simdgroup_index_in_threadgroup]], \
    uint   simd_lid  [[thread_index_in_simdgroup]]);

MSKIP_QMV_INST(8, 2)
MSKIP_QMV_INST(8, 4)
MSKIP_QMV_INST(8, 8)
MSKIP_QMV_INST(6, 2)
MSKIP_QMV_INST(6, 4)
MSKIP_QMV_INST(6, 8)
MSKIP_QMV_INST(4, 2)
MSKIP_QMV_INST(4, 4)
MSKIP_QMV_INST(4, 8)

#undef MSKIP_QMV_INST


///////////////////////////////////////////////////////////////////////////////
// fused_silu_mskip_qmv instantiations (D2 — FFN block fusion).
// Format: fused_silu_mskip_qmv_{type}_gs_64_b_{bits}_nsg_{N}
///////////////////////////////////////////////////////////////////////////////

#define FUSED_SILU_MSKIP_INST(TYPE, BITS, NSG) \
template [[host_name("fused_silu_mskip_qmv_" #TYPE "_gs_64_b_" #BITS "_nsg_" #NSG)]] \
[[kernel]] void fused_silu_mskip_qmv_impl<TYPE, 64, BITS, NSG>( \
    const device uint32_t*    w           [[buffer(0)]], \
    const device TYPE*        scales      [[buffer(1)]], \
    const device TYPE*        biases      [[buffer(2)]], \
    const device TYPE*        x           [[buffer(3)]], \
    const device TYPE*        gate_out    [[buffer(4)]], \
    device TYPE*              y           [[buffer(5)]], \
    constant const int&       in_vec_size [[buffer(6)]], \
    constant const int&       out_vec_size [[buffer(7)]], \
    constant const float&     threshold   [[buffer(8)]], \
    uint3  tid       [[threadgroup_position_in_grid]], \
    uint   simd_gid  [[simdgroup_index_in_threadgroup]], \
    uint   simd_lid  [[thread_index_in_simdgroup]]);

// fp16 variants
FUSED_SILU_MSKIP_INST(half, 8, 2)
FUSED_SILU_MSKIP_INST(half, 8, 4)
FUSED_SILU_MSKIP_INST(half, 8, 8)
FUSED_SILU_MSKIP_INST(half, 6, 2)
FUSED_SILU_MSKIP_INST(half, 6, 4)
FUSED_SILU_MSKIP_INST(half, 6, 8)
FUSED_SILU_MSKIP_INST(half, 4, 2)
FUSED_SILU_MSKIP_INST(half, 4, 4)
FUSED_SILU_MSKIP_INST(half, 4, 8)

// bf16 variants — the talker is bf16, so this saves the per-call cast.
FUSED_SILU_MSKIP_INST(bfloat16_t, 8, 2)
FUSED_SILU_MSKIP_INST(bfloat16_t, 8, 4)
FUSED_SILU_MSKIP_INST(bfloat16_t, 8, 8)
FUSED_SILU_MSKIP_INST(bfloat16_t, 6, 4)
FUSED_SILU_MSKIP_INST(bfloat16_t, 6, 8)
FUSED_SILU_MSKIP_INST(bfloat16_t, 4, 4)
FUSED_SILU_MSKIP_INST(bfloat16_t, 4, 8)

#undef FUSED_SILU_MSKIP_INST


///////////////////////////////////////////////////////////////////////////////
// zigzag_sparse_indexing_v2 instantiations (D1 — bf16 + fp16 entry points).
///////////////////////////////////////////////////////////////////////////////

template [[host_name("zigzag_sparse_indexing_v2_half")]]
[[kernel]] void zigzag_sparse_indexing_v2_impl<half>(
    device const half*    x                  [[buffer(0)]],
    device int*           dst                [[buffer(1)]],
    constant const int&   K                  [[buffer(2)]],
    constant const float& threshold          [[buffer(3)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tiitg [[thread_index_in_threadgroup]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("zigzag_sparse_indexing_v2_bfloat16_t")]]
[[kernel]] void zigzag_sparse_indexing_v2_impl<bfloat16_t>(
    device const bfloat16_t* x               [[buffer(0)]],
    device int*           dst                [[buffer(1)]],
    constant const int&   K                  [[buffer(2)]],
    constant const float& threshold          [[buffer(3)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tiitg [[thread_index_in_threadgroup]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]]);


///////////////////////////////////////////////////////////////////////////////
// D1: bf16 instantiations for zigzag_qmv_sparse (the combos we actually use
// in Stage 2 integration). Mirrors the existing half instantiations.
///////////////////////////////////////////////////////////////////////////////

#define ZIGZAG_BF16_DENSE(BITS, NSG, TG) \
template [[host_name("zigzag_qmv_dense_bfloat16_t_gs_64_b_" #BITS "_nsg_" #NSG "_tg_" #TG)]] \
[[kernel]] void zigzag_qmv_dense_impl<bfloat16_t, 64, BITS, NSG, TG>( \
    device const uint32_t*      w_zz      [[buffer(0)]], \
    device const bfloat16_t*    x         [[buffer(1)]], \
    device const bfloat16_t*    scales_zz [[buffer(2)]], \
    device const bfloat16_t*    biases_zz [[buffer(3)]], \
    device atomic<float>*       out       [[buffer(4)]], \
    constant const int&         K         [[buffer(5)]], \
    uint3  tid        [[threadgroup_position_in_grid]], \
    uint   simd_gid   [[simdgroup_index_in_threadgroup]], \
    uint   simd_lid   [[thread_index_in_simdgroup]]);

#define ZIGZAG_BF16_SPARSE(BITS, NSG, TG) \
template [[host_name("zigzag_qmv_sparse_bfloat16_t_gs_64_b_" #BITS "_nsg_" #NSG "_tg_" #TG)]] \
[[kernel]] void zigzag_qmv_sparse_impl<bfloat16_t, 64, BITS, NSG, TG>( \
    device const uint32_t*      w_zz      [[buffer(0)]], \
    device const bfloat16_t*    x         [[buffer(1)]], \
    device const bfloat16_t*    scales_zz [[buffer(2)]], \
    device const bfloat16_t*    biases_zz [[buffer(3)]], \
    device const int*           sparse_indices [[buffer(4)]], \
    device atomic<float>*       out       [[buffer(5)]], \
    constant const int&         K         [[buffer(6)]], \
    uint3  tid        [[threadgroup_position_in_grid]], \
    uint   simd_gid   [[simdgroup_index_in_threadgroup]], \
    uint   simd_lid   [[thread_index_in_simdgroup]]);

// bf16 sparse: the combos used by Stage 2 SparseHookedLinear (nsg=2, tg=16)
// plus a few for tuning headroom.
ZIGZAG_BF16_SPARSE(8, 2, 16)
ZIGZAG_BF16_SPARSE(8, 4, 16)
ZIGZAG_BF16_SPARSE(8, 4, 8)
ZIGZAG_BF16_SPARSE(8, 2, 8)
ZIGZAG_BF16_SPARSE(6, 2, 16)
ZIGZAG_BF16_SPARSE(6, 4, 16)
ZIGZAG_BF16_SPARSE(4, 2, 16)
ZIGZAG_BF16_SPARSE(4, 4, 16)

// bf16 dense at the workhorse combo (rarely used, but cheap to add).
ZIGZAG_BF16_DENSE(8, 4, 4)
ZIGZAG_BF16_DENSE(6, 4, 4)
ZIGZAG_BF16_DENSE(4, 4, 4)

#undef ZIGZAG_BF16_DENSE
#undef ZIGZAG_BF16_SPARSE


///////////////////////////////////////////////////////////////////////////////
// D1: bf16 instantiations for mskip_qmv (not currently used directly — the
// FFN block uses fused_silu_mskip_qmv — but kept for completeness).
///////////////////////////////////////////////////////////////////////////////

#define MSKIP_BF16(BITS, NSG) \
template [[host_name("mskip_qmv_bfloat16_t_gs_64_b_" #BITS "_nsg_" #NSG)]] \
[[kernel]] void mskip_qmv_impl<bfloat16_t, 64, BITS, NSG>( \
    const device uint32_t*    w           [[buffer(0)]], \
    const device bfloat16_t*  scales      [[buffer(1)]], \
    const device bfloat16_t*  biases      [[buffer(2)]], \
    const device bfloat16_t*  x           [[buffer(3)]], \
    const device uchar*       mask        [[buffer(4)]], \
    device bfloat16_t*        y           [[buffer(5)]], \
    constant const int&       in_vec_size [[buffer(6)]], \
    constant const int&       out_vec_size [[buffer(7)]], \
    uint3  tid       [[threadgroup_position_in_grid]], \
    uint   simd_gid  [[simdgroup_index_in_threadgroup]], \
    uint   simd_lid  [[thread_index_in_simdgroup]]);

MSKIP_BF16(8, 4)
MSKIP_BF16(8, 8)
MSKIP_BF16(6, 4)
MSKIP_BF16(6, 8)
MSKIP_BF16(4, 4)
MSKIP_BF16(4, 8)

#undef MSKIP_BF16


///////////////////////////////////////////////////////////////////////////////
// D6: zigzag_sparse_indexing_qkv_v2 instantiations.
///////////////////////////////////////////////////////////////////////////////

template [[host_name("zigzag_sparse_indexing_qkv_v2_half")]]
[[kernel]] void zigzag_sparse_indexing_qkv_v2_impl<half>(
    device const half*    x          [[buffer(0)]],
    device int*           dst        [[buffer(1)]],
    constant const int&   K          [[buffer(2)]],
    constant const float& tau_q      [[buffer(3)]],
    constant const float& tau_k      [[buffer(4)]],
    constant const float& tau_v      [[buffer(5)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tiitg [[thread_index_in_threadgroup]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("zigzag_sparse_indexing_qkv_v2_bfloat16_t")]]
[[kernel]] void zigzag_sparse_indexing_qkv_v2_impl<bfloat16_t>(
    device const bfloat16_t* x       [[buffer(0)]],
    device int*           dst        [[buffer(1)]],
    constant const int&   K          [[buffer(2)]],
    constant const float& tau_q      [[buffer(3)]],
    constant const float& tau_k      [[buffer(4)]],
    constant const float& tau_v      [[buffer(5)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    tiitg [[thread_index_in_threadgroup]],
    ushort  tiisg [[thread_index_in_simdgroup]],
    ushort  sgitg [[simdgroup_index_in_threadgroup]]);
