"""Test if tl.dot(input_precision='ieee') eliminates drift vs default.

Compares two kernel variants on long-prior workloads:
- v7.33 (current): tl.dot default — uses Ampere tensor cores (TF32 effectively)
- ieee variant: tl.dot(input_precision='ieee') — software MMA, full IEEE-754

If 'ieee' matches reference better at long prior, the Ampere TF32 / tensor-core
rounding is responsible for the drift, NOT summation-order differences.
"""
import sys, math, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p67_test_wrapper_e2e import build_realistic_cache, reference_attention


def build_kernel(input_precision):
    """Build a P67-style kernel with given input_precision setting."""
    import triton
    import triton.language as tl

    @triton.jit
    def _p67_test_kernel(
        Q_ptr, KV_cache_ptr, Block_table_ptr, Seq_lens_ptr,
        K_chunk_ptr, V_chunk_ptr, O_ptr,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_cache_block, stride_cache_pos, stride_cache_head,
        stride_bt_b,
        stride_kkb, stride_kkt, stride_kkh, stride_kkd,
        stride_vkb, stride_vkt, stride_vkh, stride_vkd,
        stride_ob, stride_ot, stride_oh, stride_od,
        SCALE: tl.constexpr,
        K_PLUS_1: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        Hq_TOTAL: tl.constexpr,
        KPS: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        FP8_E4B15: tl.constexpr = 0,
        IEEE: tl.constexpr = 0,
    ):
        bid = tl.program_id(0)
        kv_head = tl.program_id(1)
        BLOCK_M: tl.constexpr = K_PLUS_1 * HEADS_PER_KV

        offs_m = tl.arange(0, BLOCK_M)
        q_t = offs_m // HEADS_PER_KV
        head_in_group = offs_m % HEADS_PER_KV
        abs_head = kv_head * HEADS_PER_KV + head_in_group
        head_mask = abs_head < Hq_TOTAL

        total_seq_len = tl.load(Seq_lens_ptr + bid)
        prior_seq_len = total_seq_len - K_PLUS_1
        q_abs_pos = prior_seq_len + q_t

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM
        offs_kv = tl.arange(0, BLOCK_KV)
        vb_idx = offs_d // 2
        vb_shift = (offs_d % 2) * 4

        q_addrs = (
            bid * stride_qb
            + q_t[:, None] * stride_qt
            + abs_head[:, None] * stride_qh
            + offs_d[None, :] * stride_qd
        )
        Q = tl.load(
            Q_ptr + q_addrs,
            mask=head_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        M_state = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        L_state = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        bt_base = bid * stride_bt_b

        for start_n in range(0, total_seq_len, BLOCK_KV):
            seq_offset = start_n + offs_kv
            tile_mask = seq_offset < total_seq_len

            page_idx = seq_offset // BLOCK_SIZE
            page_off = seq_offset % BLOCK_SIZE
            physical_block = tl.load(
                Block_table_ptr + bt_base + page_idx,
                mask=tile_mask, other=0,
            ).to(tl.int64)
            slot_bases = (
                physical_block * stride_cache_block
                + page_off.to(tl.int64) * stride_cache_pos
                + tl.cast(kv_head, tl.int64) * stride_cache_head
            )

            k_addrs = slot_bases[None, :] + offs_d[:, None]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=d_mask[:, None] & tile_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            k_safe = tl.where(k_float == k_float, k_float, 0.0)
            k_safe = tl.minimum(tl.maximum(k_safe, -0.9375), 0.9375)

            if IEEE:
                S = SCALE * tl.dot(Q, k_safe.to(Q.dtype), out_dtype=tl.float32, input_precision="ieee")
            else:
                S = SCALE * tl.dot(Q, k_safe.to(Q.dtype), out_dtype=tl.float32)

            causal = q_abs_pos[:, None] >= seq_offset[None, :]
            valid = head_mask[:, None] & tile_mask[None, :] & causal
            S = tl.where(valid, S, float("-inf"))

            M_new = tl.maximum(tl.max(S, axis=1), M_state)
            alpha = tl.exp(M_state - M_new)
            P = tl.exp(S - M_new[:, None])
            L_state = L_state * alpha + tl.sum(P, axis=1)
            acc = acc * alpha[:, None]

            val_bases = slot_bases + KPS
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=tile_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=tile_mask, other=0).to(tl.uint16)
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=tile_mask, other=0).to(tl.uint16)
            v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=tile_mask, other=0).to(tl.uint16)
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=tile_mask, other=0).to(tl.uint16)
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            V_dequant = v_idx * v_scales[:, None] + v_zeros[:, None]
            V_safe = tl.where(V_dequant == V_dequant, V_dequant, 0.0)
            V_safe = tl.minimum(tl.maximum(V_safe, -32.0), 32.0)

            if IEEE:
                acc += tl.dot(P.to(Q.dtype), V_safe.to(Q.dtype), out_dtype=tl.float32, input_precision="ieee")
            else:
                acc += tl.dot(P.to(Q.dtype), V_safe.to(Q.dtype), out_dtype=tl.float32)
            M_state = M_new

        safe_L = tl.where(L_state > 0.0, L_state, 1.0)
        out = acc / safe_L[:, None]

        o_addrs = (
            bid * stride_ob
            + q_t[:, None] * stride_ot
            + abs_head[:, None] * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(
            O_ptr + o_addrs, out.to(Q.dtype),
            mask=head_mask[:, None] & d_mask[None, :],
        )
    return _p67_test_kernel


def call_kernel(kernel, q, kv_cache, block_table, seq_lens, k_chunk, v_chunk,
                scale, block_size, kps, val_data_bytes, fp8_e4b15, ieee):
    import torch as t
    import triton
    B, K_PLUS_1, Hq, D = q.shape
    Hk = k_chunk.shape[2]
    heads_per_kv = Hq // Hk
    BLOCK_D = triton.next_power_of_2(D)
    output = t.empty_like(q)
    grid = (B, Hk, 1)
    kernel[grid](
        q, kv_cache, block_table, seq_lens, k_chunk, v_chunk, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        k_chunk.stride(0), k_chunk.stride(1), k_chunk.stride(2), k_chunk.stride(3),
        v_chunk.stride(0), v_chunk.stride(1), v_chunk.stride(2), v_chunk.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        SCALE=scale,
        K_PLUS_1=K_PLUS_1, BLOCK_D=BLOCK_D, HEAD_DIM=D,
        BLOCK_SIZE=block_size, BLOCK_KV=32,
        HEADS_PER_KV=heads_per_kv, Hq_TOTAL=Hq,
        KPS=kps, VAL_DATA_BYTES=val_data_bytes,
        FP8_E4B15=fp8_e4b15, IEEE=ieee,
        num_warps=4, num_stages=2,
    )
    return output


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    cap = torch.cuda.get_device_capability()
    fp8_e4b15 = 1 if cap < (8, 9) else 0
    print(f"GPU SM={cap}")

    Hq, Hk, D, K1, BS = 8, 1, 256, 4, 2832
    test_cases = [
        (1, 256,  "S256"),
        (1, 1024, "S1K"),
        (1, 4096, "S4K"),
        (1, 8192, "S8K"),
    ]

    kernel_default = build_kernel(input_precision=None)  # not used as marker
    kernel_ieee = build_kernel(input_precision="ieee")

    print(f"\n{'name':6s}  {'prior':>6s}  {'mode':>10s}  {'rel_avg':>10s}  {'max_abs':>11s}")
    for B, prior_S, name in test_cases:
        g = torch.Generator().manual_seed(42)
        q_in = torch.randn((B, K1, Hq, D), generator=g, dtype=torch.float16) * 0.5
        kv_cache, k_dq, v_dq, bt, layout = build_realistic_cache(
            B, prior_S, K1, Hk, D, BS, fp8_e4b15=bool(fp8_e4b15)
        )
        k_chunk = k_dq[:, prior_S:prior_S + K1].to(torch.float16)
        v_chunk = v_dq[:, prior_S:prior_S + K1].to(torch.float16)
        scale = 1.0 / math.sqrt(D)
        ref = reference_attention(q_in, k_dq, v_dq, prior_S, K1, scale, Hq // Hk)
        seq_lens = torch.tensor([prior_S + K1] * B, dtype=torch.int32)

        for mode_name, ieee_flag, kernel in [
            ("default", 0, kernel_default),
            ("ieee", 1, kernel_ieee),
        ]:
            out = call_kernel(
                kernel, q_in.cuda(), kv_cache.cuda(), bt.cuda(),
                seq_lens.cuda(), k_chunk.cuda(), v_chunk.cuda(),
                scale, BS, layout["KPS"], layout["VDB"],
                fp8_e4b15, ieee_flag,
            )
            out_cpu = out.cpu().to(torch.float32)
            diff = (out_cpu - ref).abs()
            rel_avg = float(diff.mean() / max(ref.abs().mean().item(), 1e-6))
            max_abs = float(diff.max())
            print(f"{name:6s}  {prior_S:6d}  {mode_name:>10s}  {rel_avg:10.5f}  {max_abs:11.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
