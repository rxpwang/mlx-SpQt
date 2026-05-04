// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

    namespace { 
        inline std::string zigzag_kernel_name(
            const std::string& base,
            const std::string& type,
            int group_size,
            int bits,
            int num_simdgroups,
            int threadgroups_per_band) {
            std::string name;
            name.reserve(64);
            concatenate(name, base, "_", type, "_gs_", group_size, "_b_", bits, "_nsg_", num_simdgroups, "_tg_", threadgroups_per_band);
            return name;
        }

        void zero_fill_float_buffer(
            const Stream& s, metal::Device& d, array& out) {
                auto zero_kernel = d.get_kernel("zigzag_zero_fill_float");
                auto& zfe = metal::get_command_encoder(s);
                zfe.set_compute_pipeline_state(zero_kernel);
                zfe.set_output_array(out, 0);
                const size_t n = out.size();
                const size_t tgp = std::min(n, zero_kernel->maxTotalThreadsPerThreadgroup());
                zfe.dispatch_threads(MTL::Size(n, 1, 1), MTL::Size(tgp, 1, 1));
            }
    } // namespace

    void ZigzagQmvDense::eval_gpu(
        const std::vector<array>& inputs,
        array& out) {

        // prepare input and output
        auto& w_zz = inputs[0];
        auto& x = inputs[1];
        auto& scales = inputs[2];
        auto& biases = inputs[3];

        // Get the Metal device and command queue
        auto& s = stream();
        auto& d = metal::device(s.device);

        // allocate output buffer on the device
        out.set_data(allocator::malloc(out.nbytes()));

        constexpr int num_simdgroups = 4;
        constexpr int threadgroups_per_band = 4;
        constexpr int simdgroup_size = 32;

        zero_fill_float_buffer(s, d, out);

        auto kname = zigzag_kernel_name("zigzag_qmv_dense", "half", group_size_, bits_, num_simdgroups, threadgroups_per_band);
        auto kernel = d.get_kernel(kname);

        // set up the encoder + bind buffers
        auto& compute_encoder = metal::get_command_encoder(s);
        compute_encoder.set_compute_pipeline_state(kernel);

        compute_encoder.set_input_array(w_zz, 0);
        compute_encoder.set_input_array(x, 1);
        compute_encoder.set_input_array(scales, 2);
        compute_encoder.set_input_array(biases, 3);
        compute_encoder.set_output_array(out, 4);
        const int K = x.shape(-1);
        compute_encoder.set_bytes(K, 5);

        // constexpr int num_simdgroups = 4;
        // constexpr int threadgroups_per_band = 4;
        // constexpr int simdgroup_size = 32;
        const int M = out.shape(-1);
    
        // dispatch threads
        MTL::Size group_dims = MTL::Size(num_simdgroups * simdgroup_size, 1, 1);
        MTL::Size grid_dims = MTL::Size(M / group_size_, threadgroups_per_band, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    void ZigzagQmvSparse::eval_gpu(
        const std::vector<array>& inputs,
        array& out) {

        // prepare input and output
        auto& w_zz = inputs[0];
        auto& x = inputs[1];
        auto& scales = inputs[2];
        auto& biases = inputs[3];
        auto& sparse_indices = inputs[4];

        // Get the Metal device and command queue
        auto& s = stream();
        auto& d = metal::device(s.device);

        // allocate output buffer on the device
        out.set_data(allocator::malloc(out.nbytes()));

        constexpr int simdgroup_size = 32;

        zero_fill_float_buffer(s, d, out);

        auto kname = zigzag_kernel_name("zigzag_qmv_sparse", "half", group_size_, bits_, num_simdgroups_, threadgroups_per_band_);
        auto kernel = d.get_kernel(kname);

        // set up the encoder + bind buffers
        auto& compute_encoder = metal::get_command_encoder(s);
        compute_encoder.set_compute_pipeline_state(kernel);

        compute_encoder.set_input_array(w_zz, 0);
        compute_encoder.set_input_array(x, 1);
        compute_encoder.set_input_array(scales, 2);
        compute_encoder.set_input_array(biases, 3);
        compute_encoder.set_input_array(sparse_indices, 4);
        compute_encoder.set_output_array(out, 5);
        const int K = x.shape(-1);
        compute_encoder.set_bytes(K, 6);

        // constexpr int num_simdgroups = 4;
        // constexpr int threadgroups_per_band = 4;
        // constexpr int simdgroup_size = 32;
        const int M = out.shape(-1);
    
        // dispatch threads
        MTL::Size group_dims = MTL::Size(num_simdgroups_ * simdgroup_size, 1, 1);
        MTL::Size grid_dims = MTL::Size(M / group_size_, threadgroups_per_band_, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

} // namespace mlx::core