// SPDX-License-Identifier: Apache-2.0
// Genesis G4_85: torch custom op wrapping TurboMind's sm80_16816 int4 grouped-MoE
// GEMM so vLLM can replace the slow CUDA-core moe_wna16 at TP=2 for int4 MoE
// where Marlin is structurally rejected (Gemma-4-26B-A4B and the int4-MoE family).
//
// TmInt4MoE holds the prepped per-expert int4 weights (grouped layout) + a
// LlamaLinear, and runs the grouped GEMM. The constructor takes *dequantized*
// fp16 expert weights and re-quantizes via TurboMind's own QuantizeGroupwise
// (guaranteed packed format — the spec's safe weight-prep path).
#include <torch/extension.h>
#include <torch/script.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <functional>
#include <memory>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/data_format.h"
#include "src/turbomind/core/stream.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/quantization.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"

namespace tmext {

using namespace turbomind;

// Copied from testbed_v3.h (int4 / strided-ptr path only): link per-expert
// weights into a batched block view for the fused grouped-MoE GEMM.
static void LinkExperts(std::function<LinearWeight*(int)> experts, int n, LinearWeight& d)
{
    const auto& e0 = *experts(0);
    e0.copy_metadata_to(d);
    d.k_desc.num = d.q_desc.num = n;

    std::vector<std::pair<void*, int>> weights;
    std::vector<std::pair<void*, int>> scales;
    for (int i = 0; i < n; ++i) {
        auto& e = *experts(i);
        weights.emplace_back(e.weight.raw_data(), e.k_desc.ld);
        if (e.scales) {
            scales.emplace_back(e.scales.raw_data(), e.q_desc.ld);
        }
    }
    auto stream = core::Context::stream().handle();
    auto make_strided_ptr = [&](const auto& ptrs) {
        return std::shared_ptr<void>{gemm::MakeStridedPtrs(ptrs, stream), [](auto p) { cudaFree(p); }};
    };
    d.weight = core::Tensor{make_strided_ptr(weights), {n}, d.weight_format.dtype, kDEVICE};
    if (e0.scales) {
        d.scales = core::Tensor{make_strided_ptr(scales), {n}, e0.scales.dtype(), kDEVICE};
    }
    d.k_desc.ld = d.q_desc.ld = 0;
}

// --- process-lifetime TurboMind context (stream + allocators) -----------------
// Mirror test_gemm_v2 main(): establish ONE ContextGuard that is never popped,
// so the stream + allocators stay on the context stack for the whole process
// (and weights allocated through the device allocator stay valid). The guard is
// intentionally leaked (process lifetime). ScopedCtx just guarantees it exists.
struct ScopedCtx {
    ScopedCtx()
    {
        static core::Stream        s    = core::Stream::create();
        static core::Allocator     host = core::Allocator{kCPU};
        static core::Allocator     dev  = core::Allocator{s, false};
        static core::ContextGuard* g    = new core::ContextGuard{s, host, dev};  // never deleted
        (void)g;
    }
};

static core::LinearConfig make_cfg(int K, int N, DataType wt, DataType dt, int group_size)
{
    core::LinearConfig cfg;
    cfg.input_dim  = K;
    cfg.output_dim = N;
    cfg.data_type  = dt;
    // Trivial float weight -> block_in 1; quantized weight -> group_size.
    cfg.format   = ResolveLinearWeightFormat(dt, wt, (wt == dt) ? 1 : group_size, 1);
    cfg.has_bias = false;
    return cfg;
}

// Wrap a contiguous CUDA half torch tensor as a TurboMind core::Tensor (no copy).
static core::Tensor as_tm_half(const torch::Tensor& t, int rows, int cols)
{
    return core::Tensor{(half_t*)t.data_ptr(), core::Layout{{rows, cols}}, kDEVICE};
}

class TmInt4MoE: public torch::CustomClassHolder {
public:
    int64_t E_, K_, N_, group_size_;
    std::vector<std::unique_ptr<LinearWeight>> experts_;
    std::unique_ptr<LinearWeight>              batched_;
    LlamaLinear                               linear_;
    std::vector<torch::Tensor>                dq_list_;    // diag: int4-dequantized weights
    std::vector<torch::Tensor>                orig_list_;  // diag: weight as seen by the quantizer

    // w_fp16: (E, K, N) dequantized expert weights, input-major (K=input, N=output).
    TmInt4MoE(torch::Tensor w_fp16, int64_t group_size)
    {
        TORCH_CHECK(w_fp16.dim() == 3, "w_fp16 must be (E,K,N)");
        TORCH_CHECK(w_fp16.scalar_type() == torch::kHalf, "w_fp16 must be fp16");
        TORCH_CHECK(w_fp16.is_cuda() && w_fp16.is_contiguous(), "w_fp16 cuda+contiguous");

        ScopedCtx ctx;
        E_          = w_fp16.size(0);
        K_          = w_fp16.size(1);
        N_          = w_fp16.size(2);
        group_size_ = group_size;
        const DataType dt = kHalf;

        auto stream = core::Context::stream().handle();

        for (int e = 0; e < E_; ++e) {
            auto src = w_fp16[e].contiguous();  // (K,N) half

            auto orig = std::make_unique<LinearWeight>(make_cfg(K_, N_, dt, dt, group_size_));
            orig->param("weight").alloc({(size_t)K_, (size_t)N_}, dt);
            Copy(as_tm_half(src, K_, N_), orig->weight);

            // diag: read back what the quantizer will actually see
            auto origt = torch::empty({(int64_t)K_, (int64_t)N_},
                                      torch::dtype(torch::kHalf).device(torch::kCUDA));
            Copy(orig->weight, as_tm_half(origt, K_, N_));
            orig_list_.push_back(origt);

            auto qw = std::make_unique<LinearWeight>(make_cfg(K_, N_, kUint4, dt, group_size_));
            qw->param("weight").alloc({(size_t)K_, (size_t)N_}, kUint4);
            qw->param("scales").alloc({(size_t)(K_ / group_size_), (size_t)N_}, dt);
            qw->param("zeros").alloc({(size_t)(K_ / group_size_), (size_t)N_}, dt);

            auto dq = std::make_unique<LinearWeight>(make_cfg(K_, N_, dt, dt, group_size_));
            dq->param("weight").alloc({(size_t)K_, (size_t)N_}, dt);

            // (M,N) storage; quantizer needs K-major -> .t()
            QuantizeGroupwise(qw->weight.t(), qw->scales.t(), qw->zeros.t(),
                              dq->weight.t(), orig->weight.t(), {}, group_size_);

            // diag: stash the int4-dequantized weight (before prepare repacks qw)
            auto dqt = torch::empty({(int64_t)K_, (int64_t)N_},
                                    torch::dtype(torch::kHalf).device(torch::kCUDA));
            Copy(dq->weight, as_tm_half(dqt, K_, N_));
            dq_list_.push_back(dqt);

            qw->set_grouped(true);
            qw->prepare();
            experts_.push_back(std::move(qw));
        }

        batched_ = std::make_unique<LinearWeight>();
        LinkExperts([&](int i) { return experts_[i].get(); }, (int)E_, *batched_);
        cudaStreamSynchronize(stream);
    }

    // Raw grouped int4 GEMM (no epilogue): out[r] = x[f2n[r]] @ W[expert(r)].
    //   x       : (M, K) fp16
    //   f2n     : (R,)  int32 (routed slot -> source token)
    //   offsets : (E+1) int32 prefix sums
    // returns out: (R, N) fp16
    torch::Tensor forward_w1w3(torch::Tensor x, torch::Tensor f2n, torch::Tensor offsets)
    {
        ScopedCtx ctx;
        const int M = x.size(0);
        const int R = f2n.size(0);
        TORCH_CHECK(x.size(1) == K_, "x K mismatch");

        auto out = torch::empty({(int64_t)R, N_},
                                torch::dtype(torch::kHalf).device(x.device()));

        core::Tensor x_tm  = as_tm_half(x.contiguous(), M, K_);
        core::Tensor o_tm  = as_tm_half(out, R, N_);
        Buffer_<int> f2n_b{(int*)f2n.contiguous().data_ptr(), (size_t)R, kDEVICE};
        Buffer_<int> off_b{(int*)offsets.contiguous().data_ptr(), (size_t)(E_ + 1), kDEVICE};

        linear_.Forward(x_tm, *batched_, f2n_b, off_b, o_tm);
        cudaStreamSynchronize(core::Context::stream().handle());
        return out;
    }

    int64_t num_experts() const { return E_; }

    // diag: int4-dequantized weight of expert e, (K,N) fp16.
    torch::Tensor get_dequant(int64_t e) { return dq_list_[e]; }
    torch::Tensor get_orig(int64_t e) { return orig_list_[e]; }
};

}  // namespace tmext

TORCH_LIBRARY(genesis_tm, m)
{
    m.def("tm_probe() -> int");
    m.class_<tmext::TmInt4MoE>("TmInt4MoE")
        .def(torch::init<torch::Tensor, int64_t>())
        .def("forward_w1w3", &tmext::TmInt4MoE::forward_w1w3)
        .def("num_experts", &tmext::TmInt4MoE::num_experts)
        .def("get_dequant", &tmext::TmInt4MoE::get_dequant)
        .def("get_orig", &tmext::TmInt4MoE::get_orig);
}

namespace tmext {
int64_t tm_probe()
{
    ScopedCtx   ctx;
    LlamaLinear linear;
    cudaStreamSynchronize(core::Context::stream().handle());
    return 1;
}
}  // namespace tmext

TORCH_LIBRARY_IMPL(genesis_tm, CompositeExplicitAutograd, m)
{
    m.impl("tm_probe", &tmext::tm_probe);
}
