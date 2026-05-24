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

        const char* type_str = (x.dtype() == bfloat16) ? "bfloat16_t" : "half";
        auto kname = zigzag_kernel_name("zigzag_qmv_dense", type_str, group_size_, bits_, num_simdgroups, threadgroups_per_band);
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

    void FusedSiluMskipQmv::eval_gpu(
        const std::vector<array>& inputs,
        array& out) {
        // inputs = {w, scales, biases, x, gate_out}
        auto& w = inputs[0];
        auto& scales = inputs[1];
        auto& biases = inputs[2];
        auto& x = inputs[3];
        auto& gate_out = inputs[4];

        auto& s = stream();
        auto& d = metal::device(s.device);
        out.set_data(allocator::malloc(out.nbytes()));

        constexpr int simdgroup_size = 32;
        const int M = w.shape(0);
        const int K = x.shape(-1);

        // Type tag in kernel name: half or bfloat16_t.
        const char* type_str = (x.dtype() == bfloat16) ? "bfloat16_t" : "half";

        std::string kname;
        kname.reserve(80);
        concatenate(kname, "fused_silu_mskip_qmv_", type_str,
                    "_gs_64_b_", bits_, "_nsg_", num_simdgroups_);
        auto kernel = d.get_kernel(kname);

        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        enc.set_input_array(w, 0);
        enc.set_input_array(scales, 1);
        enc.set_input_array(biases, 2);
        enc.set_input_array(x, 3);
        enc.set_input_array(gate_out, 4);
        enc.set_output_array(out, 5);
        enc.set_bytes(K, 6);
        enc.set_bytes(M, 7);
        enc.set_bytes(threshold_, 8);

        const int tg_count = (M + num_simdgroups_ - 1) / num_simdgroups_;
        MTL::Size group_dims = MTL::Size(num_simdgroups_ * simdgroup_size, 1, 1);
        MTL::Size grid_dims = MTL::Size(1, tg_count, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    void MskipQmv::eval_gpu(
        const std::vector<array>& inputs,
        array& out) {
        // inputs = {w, scales, biases, x, mask}
        auto& w = inputs[0];
        auto& scales = inputs[1];
        auto& biases = inputs[2];
        auto& x = inputs[3];
        auto& mask = inputs[4];

        auto& s = stream();
        auto& d = metal::device(s.device);
        out.set_data(allocator::malloc(out.nbytes()));

        constexpr int simdgroup_size = 32;
        const int M = w.shape(0);
        const int K = x.shape(-1);

        const char* type_str = (x.dtype() == bfloat16) ? "bfloat16_t" : "half";
        std::string kname;
        kname.reserve(64);
        concatenate(kname, "mskip_qmv_", type_str, "_gs_", group_size_, "_b_", bits_, "_nsg_", num_simdgroups_);
        auto kernel = d.get_kernel(kname);

        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        enc.set_input_array(w, 0);
        enc.set_input_array(scales, 1);
        enc.set_input_array(biases, 2);
        enc.set_input_array(x, 3);
        enc.set_input_array(mask, 4);
        enc.set_output_array(out, 5);
        enc.set_bytes(K, 6);
        enc.set_bytes(M, 7);

        // Grid: M / num_simdgroups TGs (each TG has num_simdgroups SGs, each SG handles 1 row).
        const int tg_count = (M + num_simdgroups_ - 1) / num_simdgroups_;
        MTL::Size group_dims = MTL::Size(num_simdgroups_ * simdgroup_size, 1, 1);
        MTL::Size grid_dims = MTL::Size(1, tg_count, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    void ZigzagSparseIndexingQKV::eval_gpu(
        const std::vector<array>& inputs,
        array& out) {
        auto& x = inputs[0];
        auto& s = stream();
        auto& d = metal::device(s.device);

        const int K = x.shape(-1);
        out.set_data(allocator::malloc(out.nbytes()));

        constexpr int ntg = 1024;
        const char* type_suffix = (x.dtype() == bfloat16) ? "bfloat16_t" : "half";
        std::string kname;
        kname.reserve(56);
        concatenate(kname, "zigzag_sparse_indexing_qkv_v2_", type_suffix);
        auto kernel = d.get_kernel(kname);
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        enc.set_input_array(x, 0);
        enc.set_output_array(out, 1);
        enc.set_bytes(K, 2);
        enc.set_bytes(tau_q_, 3);
        enc.set_bytes(tau_k_, 4);
        enc.set_bytes(tau_v_, 5);

        MTL::Size group_dims = MTL::Size(ntg, 1, 1);
        // 3 TGs in parallel — one per mask. Same `x`, different τ, different
        // output slot. Metal scheduler runs them concurrently on M-series.
        MTL::Size grid_dims = MTL::Size(3, 1, 1);
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    void ZigzagSparseIndexing::eval_gpu(
        const std::vector<array>& inputs,
        array& out) {
        auto& x = inputs[0];
        auto& s = stream();
        auto& d = metal::device(s.device);

        const int K = x.shape(-1);

        out.set_data(allocator::malloc(out.nbytes()));

        constexpr int ntg = 1024;
        // Dispatch the dtype-specific kernel (fp16 or bf16).
        const char* type_suffix = (x.dtype() == bfloat16) ? "bfloat16_t" : "half";
        std::string kname;
        kname.reserve(48);
        concatenate(kname, "zigzag_sparse_indexing_v2_", type_suffix);
        auto kernel = d.get_kernel(kname);
        auto& enc = metal::get_command_encoder(s);
        enc.set_compute_pipeline_state(kernel);
        enc.set_input_array(x, 0);
        enc.set_output_array(out, 1);
        enc.set_bytes(K, 2);
        enc.set_bytes(threshold_, 3);

        MTL::Size group_dims = MTL::Size(ntg, 1, 1);
        MTL::Size grid_dims = MTL::Size(1, 1, 1);  // single TG, loops over chunks
        enc.dispatch_threadgroups(grid_dims, group_dims);
    }

    void ZigzagQmvDenseFast::eval_gpu(
        const std::vector<array>& inputs,
        array& out) {

        // Inputs: {w_zz, x, scales, biases}.
        // One TG per band, no cross-TG aggregation, no zero-fill prepass —
        // the kernel writes its results directly to global y.
        auto& w_zz = inputs[0];
        auto& x = inputs[1];
        auto& scales = inputs[2];
        auto& biases = inputs[3];

        auto& s = stream();
        auto& d = metal::device(s.device);

        out.set_data(allocator::malloc(out.nbytes()));

        constexpr int simdgroup_size = 32;

        // Kernel name: zigzag_qmv_fast_half_gs_64_b_{bits}_nsg_{N}_rps_{R}
        std::string kname;
        kname.reserve(64);
        concatenate(kname, "zigzag_qmv_fast_half_gs_", group_size_, "_b_", bits_,
                    "_nsg_", num_simdgroups_, "_rps_", results_per_simdgroup_);
        auto kernel = d.get_kernel(kname);

        auto& compute_encoder = metal::get_command_encoder(s);
        compute_encoder.set_compute_pipeline_state(kernel);

        compute_encoder.set_input_array(w_zz, 0);
        compute_encoder.set_input_array(x, 1);
        compute_encoder.set_input_array(scales, 2);
        compute_encoder.set_input_array(biases, 3);
        compute_encoder.set_output_array(out, 4);
        const int K = x.shape(-1);
        compute_encoder.set_bytes(K, 5);

        const int M = out.shape(-1);
        // One TG per band, num_simdgroups_ simdgroups per TG.
        MTL::Size group_dims = MTL::Size(num_simdgroups_ * simdgroup_size, 1, 1);
        MTL::Size grid_dims = MTL::Size(M / group_size_, 1, 1);
        compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    void ZigzagQmvMskip::eval_gpu(
        const std::vector<array>& inputs,
        array& out) {

        // Inputs: {w_zz, x, scales, biases, active_indices}.
        // First-cut implementation: run the dense GEMV unmodified, then a small
        // post-pass kernel zeroes the M-positions that are not in active_indices.
        // No wall-clock savings vs dense — this exists to lock in the API while
        // the optimized row-walk kernel is built.
        auto& w_zz = inputs[0];
        auto& x = inputs[1];
        auto& scales = inputs[2];
        auto& biases = inputs[3];
        auto& active_indices = inputs[4];

        auto& s = stream();
        auto& d = metal::device(s.device);

        out.set_data(allocator::malloc(out.nbytes()));

        // Same hardcoded dispatch geometry as ZigzagQmvDense.
        constexpr int num_simdgroups = 4;
        constexpr int threadgroups_per_band = 4;
        constexpr int simdgroup_size = 32;

        zero_fill_float_buffer(s, d, out);

        // --- Dense GEMV pass ---
        const char* type_str = (x.dtype() == bfloat16) ? "bfloat16_t" : "half";
        auto kname = zigzag_kernel_name("zigzag_qmv_dense", type_str, group_size_, bits_, num_simdgroups, threadgroups_per_band);
        auto kernel = d.get_kernel(kname);
        {
          auto& compute_encoder = metal::get_command_encoder(s);
          compute_encoder.set_compute_pipeline_state(kernel);
          compute_encoder.set_input_array(w_zz, 0);
          compute_encoder.set_input_array(x, 1);
          compute_encoder.set_input_array(scales, 2);
          compute_encoder.set_input_array(biases, 3);
          compute_encoder.set_output_array(out, 4);
          const int K = x.shape(-1);
          compute_encoder.set_bytes(K, 5);
          const int M = out.shape(-1);
          MTL::Size group_dims = MTL::Size(num_simdgroups * simdgroup_size, 1, 1);
          MTL::Size grid_dims = MTL::Size(M / group_size_, threadgroups_per_band, 1);
          compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
        }

        // --- Zero-inactive post-pass ---
        {
          auto mask_kernel = d.get_kernel("zigzag_mskip_zero_inactive");
          auto& mfe = metal::get_command_encoder(s);
          mfe.set_compute_pipeline_state(mask_kernel);
          mfe.set_output_array(out, 0);
          mfe.set_input_array(active_indices, 1);
          const int M = out.shape(-1);
          mfe.set_bytes(M, 2);
          const size_t tgp = std::min((size_t)M, mask_kernel->maxTotalThreadsPerThreadgroup());
          mfe.dispatch_threads(MTL::Size(M, 1, 1), MTL::Size(tgp, 1, 1));
        }
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

        const char* type_str = (x.dtype() == bfloat16) ? "bfloat16_t" : "half";
        auto kname = zigzag_kernel_name("zigzag_qmv_sparse", type_str, group_size_, bits_, num_simdgroups_, threadgroups_per_band_);
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