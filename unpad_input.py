import torch
import torch.nn.functional as F
from flash_attn.bert_padding import index_first_axis
# 输入张量 (batch_size=2, sequence_length=5, hidden_dim=3)
hidden_states = torch.tensor([
    [[1, 1, 1], [2, 2, 2], [3, 3, 3], [0, 0, 0], [0, 0, 0]],
    [[4, 4, 4], [5, 5, 5], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
])

hidden_states = torch.tensor([
    [[1], [2], [3], [0], [0]],
    [[4], [5], [0], [0], [0]]
])

# 掩码张量 (batch_size=2, sequence_length=5)
attention_mask = torch.tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0]
])


from einops import rearrange

# def index_first_axis(input, indices):
#     return input[indices]

def unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    print("seqlens_in_batch:", seqlens_in_batch)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    print("indices:", indices)
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    print("cu_seqlens:", cu_seqlens)
    print("hidden_states shape:", hidden_states.shape)
    print("hidden_states:", hidden_states)
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# 调用函数
hidden_states_unpadded, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)

print("Unpadded hidden states shape:", hidden_states_unpadded.shape)
print("Unpadded hidden states:", hidden_states_unpadded)
print("Indices of non-masked tokens:", indices)
print("Cumulative sequence lengths:", cu_seqlens)
print("Max sequence length in batch:", max_seqlen_in_batch)