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
      attn_score.masked_fill_(torch.autograd.Variable(attn_mask[:,:,:]), -1e6)

    attn_prob = F.softmax(attn_score, dim=1)

    #attn_vec = torch.einsum('ijb,jbd->ibd', (attn_prob, c))
    attn_prob_permuted = attn_prob.permute(2, 0, 1).contiguous()
    c_permuted = c.permute(1, 0, 2).contiguous()
    attn_vec = torch.bmm(attn_prob_permuted, c_permuted).permute(1, 0, 2).contiguous()

    output = attn_vec

    return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0, dropatt=0):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.scale = 1 / (d_head ** 0.5)

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [seq_len, bsz, nhid] -> [seq_len, bsz, nhid]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        qlen = h.size(0)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        attn_mask = torch.triu(torch.ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None].cuda()

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.contiguous().view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.contiguous().view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        #attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score = torch.matmul(head_q.permute(1, 2, 0, 3), head_k.permute(1, 2, 3, 0)).permute(2, 3, 0, 1).contiguous()
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any():
            if attn_mask.dim() == 2:
                # attn_score.masked_fill_(attn_mask[None,:,:,None], -1000000.0)
                attn_score.masked_fill_(torch.autograd.Variable(attn_mask[None,:,:,None]), -1e6)
            elif attn_mask.dim() == 3:
                # attn_score.masked_fill_(attn_mask[:,:,:,None], -1000000.0)
                attn_score.masked_fill_(torch.autograd.Variable(attn_mask[:,:,:,None]), -1e6)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        #attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = torch.matmul(attn_prob.permute(2, 3, 0, 1), head_v.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        output = self.o_net(attn_vec)
        #output = self.drop(output)

        ##### residual connection
        #output = h + output

        return output