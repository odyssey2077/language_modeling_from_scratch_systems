import torch
import torch.nn as nn
import math
from einops import einsum
import triton
import triton.language as tl

QUERY_TILE_SIZE = 128
KEY_TILE_SIZE = 128

# • To debug, we suggest comparing the results of each Triton operation you perform with the
# tiled PyTorch implementation you wrote in part (a).
# • Your launch grid should be set as (Tq , batch_size), meaning each Triton program instance
# will load only elements from a single batch index, and only read/write to a single query tile
# of Q, O, and L.
# • The kernel should only have a single loop, which will iterate key tiles 1 ≤ j ≤ Tk.
# • Advance block pointers at the end of the loop.
# • Use the function declaration below (using the block pointer we give you, you should be
# able to infer the setup of the rest of the pointers)
# You can use print statements in Triton with tl.device_print to debug: https://triton-lang.
# org/main/python-api/generated/triton.language.device_print.html. There is a setting
# TRITON_INTERPRET=1 to run the Triton interpreter on CPU, though we have found it buggy.
# • When defining block pointers, make sure they have the correct offsets, and that block offsets are
# multiplied by the appropriate tile sizes.
# • The launch grid of thread blocks is set with
# kernel_fn[(launch_grid_d1, launch_grid_d2, ...)](...arguments...)
# in the methods of the torch.autograd.Function subclass, as we saw in the weighted sum exam-
# ple.
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Q_i = Q_i.to(tl.float32)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_i = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        K_i = K_i.to(tl.float32)
        V_i = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_i = V_i.to(tl.float32)
        S_i = tl.dot(Q_i, tl.trans(K_i)) * scale
        last_m = m
        m = tl.maximum(m, tl.max(S_i, axis=-1))
        P_i = tl.exp(S_i - tl.expand_dims(m, axis=-1)).to(V_i.dtype)
        l = tl.exp(last_m - m) * l + tl.sum(P_i, axis=-1)
        output = output * tl.expand_dims(tl.exp(last_m - m), axis=-1) + tl.dot(P_i, V_i)
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    output = output * tl.expand_dims(1 / l, axis=-1)
    output = output.to(O_block_ptr.type.element_ty)
    l = m + tl.log(l)
    l = l.to(L_block_ptr.dtype.element_ty)
    tl.store(O_block_ptr, output, boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))



class FlashAttention(torch.autograd.Function):
    # input n_q / n_k dimension is expected to be pow of 2 and >= 16
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        n_q, n_k, d = Q.shape[-2], K.shape[-2], K.shape[-1]
        splited_Q = torch.split(Q, QUERY_TILE_SIZE, -2)
        splited_K = torch.split(K, KEY_TILE_SIZE, -2)
        splited_V = torch.split(V, KEY_TILE_SIZE, -2)
        global_O = torch.zeros(Q.shape, device=Q.device)
        global_L = torch.zeros(Q.shape[:-1], device=Q.device)
        # expect n_q % QUERY_TILE_SIZE == 0, n_k % KEY_TILE_SIZE == 0
        for i in range(n_q // QUERY_TILE_SIZE):
            O_i = torch.zeros_like(splited_Q[i], device=Q.device)
            l_i = torch.zeros(splited_Q[i].shape[:-1], device=Q.device)
            m_i = torch.full(splited_Q[i].shape[:-1], -torch.inf, device=Q.device)
            for j in range(n_k // KEY_TILE_SIZE):
                S_i = einsum(splited_Q[i], splited_K[j], '... q_tile d, ... k_tile d -> ... q_tile k_tile') / math.sqrt(d)
                previous_m_i = m_i
                local_max, _ = torch.max(S_i, dim=-1)
                m_i = torch.max(m_i, local_max)
                P_i = torch.exp(S_i - m_i.unsqueeze(-1))
                l_i = torch.exp(previous_m_i - m_i) * l_i + torch.sum(P_i, dim=-1)
                O_i = einsum(torch.exp(previous_m_i - m_i), O_i, "... q_tile, ... q_tile d -> ... q_tile d") + einsum(P_i, splited_V[j], "... q_tile k_tile, ... k_tile d -> ... q_tile d")
            O_i = einsum(1 / l_i, O_i, "... q_tile, ... q_tile d -> ... q_tile d")
            l_i = m_i + torch.log(l_i)
            start_idx = i * QUERY_TILE_SIZE
            end_idx = start_idx + QUERY_TILE_SIZE            
            global_O[..., start_idx:end_idx, :] = O_i
            global_L[..., start_idx:end_idx] = l_i           

        ctx.save_for_backward(global_L)
        return global_O


    def backward(ctx, gradient_out):
        raise NotImplementedError


class FlashAttentionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, n_q, d = Q.shape
        _, n_k, _ = K.shape
        
        scale = 1.0 / math.sqrt(d)
        
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, n_q), device=Q.device, dtype=torch.float32)
        
        grid = (triton.cdiv(n_q, QUERY_TILE_SIZE), batch_size)
        
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_q, n_k,
            scale,
            D=d,
            Q_TILE_SIZE=QUERY_TILE_SIZE,
            K_TILE_SIZE=KEY_TILE_SIZE,
        )
        
        ctx.save_for_backward(L)
        return O
    
    @staticmethod
    def backward(ctx, gradient_out):
        raise NotImplementedError