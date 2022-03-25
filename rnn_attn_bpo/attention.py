import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

  def __init__(self):
    super(Attention, self).__init__()

  def forward(self, h, mems=None):
    # [seq_len, bsz, nhid] -> [seq_len, bsz, nhid]

    if mems is not None:
      c = torch.cat([mems, h], 0)
    else:
      c = h

    qlen = h.size(0)
    mlen = mems[0].size(0) if mems is not None else 0
    klen = mlen + qlen
    attn_mask = torch.triu(torch.ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None].cuda()

    #attn_score = torch.einsum('ibd,jbd->ijb', (h, c))
    h_permuted = h.permute(1, 0, 2).contiguous()
    c_permuted = c.permute(1, 2, 0).contiguous()
    attn_score = torch.bmm(h_permuted, c_permuted).permute(1, 2, 0).contiguous()

    if attn_mask is not None and attn_mask.any():
      attn_score.masked_fill_(torch.autograd.Variable(attn_mask[:,:,:]), -1000)

    attn_prob = F.softmax(attn_score, dim=1)

    #attn_vec = torch.einsum('ijb,jbd->ibd', (attn_prob, c))
    attn_prob_permuted = attn_prob.permute(2, 0, 1).contiguous()
    c_permuted = c.permute(1, 0, 2).contiguous()
    attn_vec = torch.bmm(attn_prob_permuted, c_permuted).permute(1, 0, 2).contiguous()

    output = attn_vec

    return output
