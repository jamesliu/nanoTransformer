import torch
import torch.nn.functional as F

def dot_product_attention(q, K, V):
    logits = torch.einsum('k,mk->m', q, K)
    weights = F.softmax(logits, dim=-1)
    return torch.einsum('m,mv->v', weights, V)

def multihead_attention(q, K, V, P_q, P_k, P_v, P_o):
    q = torch.einsum('d,hdk->hk', q, P_q)
    K = torch.einsum('md,hdk->hmk', K, P_k)
    V = torch.einsum('md,hdv->hmv', V, P_v)
    logits = torch.einsum('hk,hmk->hm', q, K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('hm,hmv->hv', weights, V)
    y = torch.einsum('hv,hdv->d', o, P_o)
    return y

def multihead_attention_batched(X, M, mask, P_q, P_k, P_v, P_o):
    Q = torch.einsum('bnd,hdk->bhnk', X, P_q)
    K = torch.einsum('bmd,hdk->bhmk', M, P_k)
    V = torch.einsum('bmd,hdv->bhmv', M, P_v)
    logits = torch.einsum('bhnk,bhmk->bhnm', Q, K)
    
    #print("Logits shape:", logits.shape)
    #print("Mask shape:", mask.shape)

    # Add extra dimensions to the mask to match the logits' shape
    if mask.dim() == 2:
        #print("The mask is 2D.", mask.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)  # Correcting to [batch_size, 1, 1, seq_len]
        #print("New mask shape:", mask.shape)
    

    logits = logits + mask
    weights = F.softmax(logits, dim=-1)

    O = torch.einsum('bhnm,bhmv->bhnv', weights, V)
    Y = torch.einsum('bhnv,hdv->bnd', O, P_o)
    return Y

def multihead_self_attention_incremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
    q = torch.einsum('bd,hdk->bhk', x, P_q)
    new_K = torch.cat([prev_K, torch.einsum('bd,hdk->bhk', x.unsqueeze(2), P_k)], dim=2)
    new_V = torch.cat([prev_V, torch.einsum('bd,hdv->bhv', x.unsqueeze(2), P_v)], dim=2)

    logits = torch.einsum('bhk,bhmk->bhm', q, new_K)
    weights = F.softmax(logits, dim=-1)
    o = torch.einsum('bhm,bhmv->bhv', weights, new_V)
    y = torch.einsum('bhv,hdv->bd', o, P_o)

    return y, new_K, new_V

def multiquery_attention_batched(X, M, mask, P_q, P_k, P_v, P_o):
    Q = torch.einsum('bnd,hdk->bhnk', X, P_q)
    K = torch.einsum('bmd,dk->bmk', M, P_k)
    V = torch.einsum('bmd,dv->bmv', M, P_v)

    logits = torch.einsum('bhnk,bmk->bhnm', Q, K)
    logits = logits + mask
    weights = F.softmax(logits, dim=-1)

    O = torch.einsum('bhnm,bmv->bhnv', weights, V)
    Y = torch.einsum('bhnv,hdv->bnd', O, P_o)
    return Y

# You would continue with the other function conversions similarly.
