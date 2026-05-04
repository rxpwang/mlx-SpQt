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