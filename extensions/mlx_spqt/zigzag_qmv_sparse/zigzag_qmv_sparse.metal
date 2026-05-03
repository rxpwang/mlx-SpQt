// Copyright © 2023-2025 Apple Inc.

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"   
#include "mlx/backend/metal/kernels/quantized_utils.h"    
#include "mlx/backend/metal/kernels/quantized.h"

// ///////////////////////////////////////////////////////////////////////////////                            
// // Zero-fill kernel — used by eval_gpu to zero-init y before the main kernel
// // (cross-TG atomic-add accumulates from initial value, so initial must be 0)                              
// ///////////////////////////////////////////////////////////////////////////////                            
                                                                                                            
// [[kernel]] void zigzag_zero_fill_float(                                                                    
//     device float* out [[buffer(0)]],
//     uint id [[thread_position_in_grid]]) {                                                                 
//   out[id] = 0.0f;
// }  



template <typename T, int group_size, int bits, int num_simdgroups, int threadgroups_per_band>
[[kernel]] void zigzag_qmv_sparse_impl(
    device const uint32_t*      w_zz      [[buffer(0)]],
    device const T*             x         [[buffer(1)]],
    device const T*             scales_zz [[buffer(2)]],
    device const T*             biases_zz [[buffer(3)]],
    device const int*             sparse_indices [[buffer(4)]],
    device atomic<float>*       out       [[buffer(5)]],
    constant const int&         K         [[buffer(6)]],
    uint3  tid        [[threadgroup_position_in_grid]],
    uint   simd_gid   [[simdgroup_index_in_threadgroup]],
    uint   simd_lid   [[thread_index_in_simdgroup]]) {

    const uint thread_id = simd_gid * 32 + simd_lid;  
    
    constexpr int packs_per_thread = 4; //bits == 2 ? 1 : 2;
    //constexpr int num_simdgroups = num_simdgroups; 
    constexpr int results_per_simdgroup = group_size; // each simdgroup will only work on 1 band of group_size rows final results.
    constexpr int pack_factor = get_pack_factor<bits, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread; // number of values (elements of x) processed by each thread.
    constexpr int block_size = values_per_thread * SIMD_SIZE; // number of values (elements of x) processed by each simdgroup in total.
    constexpr int scale_step_per_thread = group_size / values_per_thread; // scale_step_per_thread also equals to number of threads that share the same scale and bias, since threads in the same simdgroup work on the same group of rows.

    const device uint8_t* ws = (const device uint8_t*)w_zz; // treat weights as bytes for easier indexing with bits.
    const device int* sparse_indices_cur = sparse_indices + 1; // the first element in sparse_indices is the total number of non-zero elements, so we need to offset by 1 to get the actual indices.
    typedef float U;

    // prepare atomic shared memory for accumulation within threadgroup. each threadgroup will compute results for a band of group_size rows.
    constexpr float SCALE_FACTOR = 65536.0f; // float to int
    threadgroup atomic<int> shared_out[group_size]; 

    // initialize shared_out to 0.0f
    for (int i = thread_id; i < group_size; i += num_simdgroups * SIMD_SIZE) {
        atomic_store_explicit(&shared_out[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // ensure shared_out is initialized before

    // local result accumulator in registers for each thread.
    float acc[values_per_thread] = {0.0f}; // each thread accumulates values_per_thread output elements. 
    float bias_total = 0.0f; // accumulate total bias for the thread, since bias is shared across the group of rows that the thread works on, we can accumulate it in the innermost loop together with multiplication to better utilize the accumulated value before writing to shared memory.

    // compute the starting row index for this threadgroup
    int band_id = tid.x; // each threadgroup works on one band of rows, and each band has group_size rows.
    int row_start = band_id * group_size;
    
    // compute the starting group index for this simdgroup. K groups in a row, divided by threadgroups_per_band * num_simdgroups simdgroups.
    int n_total = sparse_indices[0]; // total number of non-zero elements in the sparse vector, stored in the first element of sparse_indices.
    int groups_per_simdgroup = n_total / (threadgroups_per_band * num_simdgroups);
    int groups_start_id = groups_per_simdgroup * (simd_gid + tid.y * num_simdgroups);
    int groups_start_count_thread = groups_start_id + (simd_lid * values_per_thread / group_size);
    int groups_end_count_thread = groups_start_id + groups_per_simdgroup;
    int group_cur_count_thread = groups_start_count_thread;
    // // for each thread, get the x offset, basically the group index the thread will start from.
    // int x_offset = groups_start_id + simd_lid * values_per_thread / group_size; // group count
    // // for each thread, get the w offset, ws uint8. offset = element count * element bitwidth / 8
    // int w_offset = sparse_indices_cur[groups_start_id_thread] * group_size * bits / 8 + row_start * K * bits / 8 + ((simd_lid * values_per_thread % group_size) * bits / 8);

    // x += x_offset; // move the x pointer to the starting offset for this thread.
    // ws += w_offset; // move the weight pointer to the starting offset for this thread.
    // scales_zz += sparse_indices_cur[groups_start_id_thread] + K * band_id; 
    // biases_zz += sparse_indices_cur[groups_start_id_thread] + K * band_id; 



    // calculate the total outer loop step size, essentially total element count processed by each simdgroup, divided by each round of loop processing element count (simdgroup_size * values_per_thread).
    int total_step = (groups_per_simdgroup * group_size + values_per_thread * 32 - 1) / (values_per_thread * 32);
    // outer loop, each simdgroup processes a block of elements from x and corresponding weights, scales, biases, and accumulates partial results to shared_out.
    for (int k = 0; k < total_step; k++) {
        int group_cur_id_thread = sparse_indices_cur[group_cur_count_thread]; // get the actual group index for this thread to work on. since the non-zero elements in x are stored in a sparse format with an index array, we need to get the group index from the sparse_indices array using the thread's group count index.
        if (group_cur_count_thread < groups_end_count_thread) { 
            // load x values for the thread. single value with the offset
            float x_cur = x[group_cur_id_thread];
            int scale_offset = group_cur_id_thread + K * band_id; // calculate the offset for scales and biases. since scales and biases are stored in a dense format, we can directly calculate the offset using the actual group index.
            float x_scale_cur = x_cur * scales_zz[scale_offset]; // apply scale to x. since quantized weight will be multiplied with x, we can apply the scale to x directly.
            float x_bias_cur = x_cur * biases_zz[scale_offset]; // apply bias to x. this is an approximation to adding bias after multiplication, but it allows us to reuse the same accumulation for different groups of rows that share the same scale and bias, which is necessary for efficiency.
            bias_total += x_bias_cur; // accumulate bias for the thread. since bias is shared across the group of rows that the thread works on, we can accumulate it in the innermost loop together with multiplication to better utilize the accumulated value before writing to shared memory.

            // load and unpack weigths. currently 4 bits
            const device uint16_t* w_cur = (const device uint16_t*)(ws + row_start * K * bits / 8 + group_cur_id_thread * group_size * bits / 8 + ((simd_lid * values_per_thread % group_size) * bits / 8)); // load 16 bytes (128 bits) of weights, which corresponds to 32 values for 4-bit quantization.
            for (int i = 0; i < values_per_thread / 4; i++) {
                acc[i*4 + 0] += x_scale_cur * (w_cur[i] & 0x000f); //+ x_bias_cur; // add bias in the innermost loop to better utilize the accumulated value before writing to shared memory, since different groups of rows share the same bias and scale.
                acc[i*4 + 1] += x_scale_cur * ((w_cur[i] & 0x00f0) >> 4); //+ x_bias_cur;
                acc[i*4 + 2] += x_scale_cur * ((w_cur[i] & 0x0f00) >> 8); //+ x_bias_cur;
                acc[i*4 + 3] += x_scale_cur * ((w_cur[i] & 0xf000) >> 12); //+ x_bias_cur;
            }
        }
        // move pointers for the next loop iteration. 
        group_cur_count_thread += values_per_thread * 32 / group_size; 
        // x += values_per_thread * 32 / group_size; 
        // ws += values_per_thread * 32 * bits / 8;
        // scales_zz += values_per_thread * 32 / group_size;
        // biases_zz += values_per_thread * 32 / group_size;
    }

    // apply bias_total to all rows once at the end.
    #pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
        acc[i] += bias_total;
    }

    // write the accumulated results to shared memory atomically.
    for (int i = 0; i < values_per_thread; i++) {
        int idx = simd_lid * values_per_thread % group_size + i;
        atomic_fetch_add_explicit(&shared_out[idx], (int)(acc[i] * SCALE_FACTOR), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // ensure all threads have written their results to shared memory

    // // all the threads write their results to global memory atomically. 
    // for (int i = 0; i < values_per_thread; i++) {
    //     int out_index = row_start + simd_lid * values_per_thread % group_size + i;
    //     atomic_fetch_add_explicit(&out[out_index], acc[i], memory_order_relaxed);
    // }

    // write the shared memory results to global memory in parallel
    for (int i = thread_id; i < group_size; i += num_simdgroups * SIMD_SIZE) {
        float v = atomic_load_explicit(&shared_out[i], memory_order_relaxed) / SCALE_FACTOR; // convert back to float
        atomic_fetch_add_explicit(&out[row_start + i], v, memory_order_relaxed); // write to global memory atomically, since different simdgroups may work on the same group of rows and cause write conflicts.
    }
}

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