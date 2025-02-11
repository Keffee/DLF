import torch
from torch import Tensor, nn
from fuxictr.pytorch.layers import ScaledDotProductAttention
import torch.nn.functional as F

class DLF_MLP(nn.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(dim_in, dim_hidden))

            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_hidden))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        if dim_out:
            layers.append(nn.Linear(dim_in, dim_out))
        else:
            layers.append(nn.Linear(dim_in, dim_hidden))

        super().__init__(*layers)
        
class DLF_LinearCompressBlock(nn.Module):
    def __init__(self, num_emb_in: int, num_emb_out: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty((num_emb_in, num_emb_out)))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.weight

        # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.permute(0, 2, 1)

        return outputs

class DLF_FactorizationMachineBlock(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        self.weight = nn.Parameter(torch.empty((num_emb_in, rank)))
        self.norm = nn.LayerNorm(num_emb_in * rank)
        self.mlp = DLF_MLP(
            dim_in=num_emb_in * rank,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = outputs @ self.weight

        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = torch.bmm(inputs, outputs)

        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        outputs = outputs.view(-1, self.num_emb_in * self.rank)

        # (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        outputs = self.mlp(self.norm(outputs))

        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.view(-1, self.num_emb_out, self.dim_emb)

        return outputs

class DLF_FactorizationMachineBlock_x0(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        num_emb_out: int,
        dim_emb: int,
        rank: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_emb_in = num_emb_in
        self.num_emb_out = num_emb_out
        self.dim_emb = dim_emb
        self.rank = rank

        self.weight = nn.Parameter(torch.empty((num_emb_in, rank)))
        self.norm = nn.LayerNorm(num_emb_in * rank)
        self.mlp = DLF_MLP(
            dim_in=num_emb_in * rank,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=num_emb_out * dim_emb,
            dropout=dropout,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor, x0: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, rank) -> (bs, dim_emb, rank)
        outputs = outputs @ self.weight

        # (bs, num_emb_in, dim_emb) @ (bs, dim_emb, rank) -> (bs, num_emb_in, rank)
        outputs = torch.bmm(x0, outputs)

        # (bs, num_emb_in, rank) -> (bs, num_emb_in * rank)
        outputs = outputs.view(-1, self.num_emb_in * self.rank)

        # (bs, num_emb_in * rank) -> (bs, num_emb_out * dim_emb)
        outputs = self.mlp(self.norm(outputs))

        # (bs, num_emb_out * dim_emb) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.view(-1, self.num_emb_out, self.dim_emb)

        return outputs


class DLF_ResidualProjection(nn.Module):
    def __init__(self, num_emb_in: int, num_emb_out: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty((num_emb_in, num_emb_out)))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, dim_emb, num_emb_in)
        outputs = inputs.permute(0, 2, 1)

        # (bs, dim_emb, num_emb_in) @ (num_emb_in, num_emb_out) -> (bs, dim_emb, num_emb_out)
        outputs = outputs @ self.weight

        # # (bs, dim_emb, num_emb_out) -> (bs, num_emb_out, dim_emb)
        outputs = outputs.permute(0, 2, 1)

        return outputs
    
class DLF_MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_num, att_num, num_hidden, dim_hidden, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., scale=False):
        super(DLF_MultiHeadSelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.head_dim = attention_dim // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
        self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.scale = scale
        self.mlp = DLF_MLP(
            dim_in=input_num*attention_dim,
            num_hidden=num_hidden,
            dim_hidden=dim_hidden,
            dim_out=att_num*attention_dim,
            dropout=dropout_rate,
        )
        self.norm = nn.LayerNorm(input_num*attention_dim)

    def forward(self, X, mask=None):
        residual = X
        
        # linear projection
        query = self.W_q(X)
        key = self.W_k(X)
        value = self.W_v(X)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if mask:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # scaled dot product attention
        output, attention = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, -1)
        output = self.mlp(self.norm(output))
        output = output.view(batch_size, self.num_heads * self.head_dim, -1)
        output = output.transpose(1, 2)
        return output
    
    
class DLFLayer(nn.Module):
    def __init__(
        self,
        base_emb_in: int,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.lcb = DLF_LinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = DLF_FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        self.fmb_x0 = DLF_FactorizationMachineBlock_x0(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        self.norm = nn.LayerNorm(dim_emb)

        if num_emb_in != num_emb_lcb + num_emb_fmb*2:
            self.residual_projection = DLF_ResidualProjection(num_emb_in, num_emb_lcb + num_emb_fmb*2)
        else:
            self.residual_projection = nn.Identity()
        if base_emb_in != num_emb_in:
            self.x0_change_projection = DLF_ResidualProjection(base_emb_in, num_emb_in)
        else:
            self.x0_change_projection = nn.Identity()

    def forward(self, inputs: Tensor, x0: Tensor) -> Tensor:
        x0 = self.x0_change_projection(x0)
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)
        fmb_x0 = self.fmb_x0(inputs, x0)

        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = torch.concat((fmb, fmb_x0, lcb), dim=1)

        # (bs, num_emb_lcb + num_emb_fmb, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb, dim_emb)
        outputs = self.norm(outputs + self.residual_projection(inputs))

        return outputs
    
    
class DLFLayer_att_cross(nn.Module):
    def __init__(
        self,
        base_emb_in: int,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        num_emb_att: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        num_heads: int,
        scale: bool,
    ) -> None:
        super().__init__()

        self.lcb = DLF_LinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = DLF_FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        self.fmb_x0 = DLF_FactorizationMachineBlock_x0(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        self.att = DLF_MultiHeadSelfAttention(
            num_emb_in,
            num_emb_att,
            num_hidden,
            dim_hidden,
            dim_emb,
            attention_dim=dim_emb, 
            num_heads=num_heads, 
            dropout_rate=dropout,
            scale=scale,
        )
        self.norm = nn.LayerNorm(dim_emb)

        if num_emb_in != num_emb_lcb + num_emb_fmb*2 + num_emb_att:
            self.residual_projection = DLF_ResidualProjection(num_emb_in, num_emb_lcb + num_emb_fmb*2 + num_emb_att)
        else:
            self.residual_projection = nn.Identity()
        if base_emb_in != num_emb_in:
            self.x0_change_projection = DLF_ResidualProjection(base_emb_in, num_emb_in)
        else:
            self.x0_change_projection = nn.Identity()

    def forward(self, inputs: Tensor, x0: Tensor) -> Tensor:
        x0 = self.x0_change_projection(x0)
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)
        fmb_x0 = self.fmb_x0(inputs, x0)
        
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_att, dim_emb)
        att = self.att(inputs)

        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb), (bs, num_emb_att, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb)
        # print(lcb.shape, fmb.shape, att.shape)
        outputs = torch.concat((fmb, fmb_x0, lcb, att), dim=1)

        # (bs, num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb)
        outputs = self.norm(outputs + self.residual_projection(inputs))

        return outputs
    
# 这里的self代表的意思是，我的attention token只在自己内部运行，也就是说我不会像cross版一样利用来自lcb和fmb
class DLFLayer_att_self(nn.Module):
    def __init__(
        self,
        base_emb_in: int,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        num_emb_att: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        num_heads: int,
        scale: bool,
    ) -> None:
        super().__init__()

        self.lcb = DLF_LinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = DLF_FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        self.fmb_x0 = DLF_FactorizationMachineBlock_x0(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        if num_emb_in == num_emb_lcb + num_emb_fmb + num_emb_att:
            self.att = DLF_MultiHeadSelfAttention(
                num_emb_att,
                num_emb_att,
                num_hidden,
                dim_hidden,
                dim_emb,
                attention_dim=dim_emb, 
                num_heads=num_heads, 
                dropout_rate=dropout,
                scale=scale,
            )
        else:
            self.att = DLF_MultiHeadSelfAttention(
                num_emb_in,
                num_emb_att,
                num_hidden,
                dim_hidden,
                dim_emb,
                attention_dim=dim_emb, 
                num_heads=num_heads, 
                dropout_rate=dropout,
                scale=scale,
            )
        self.num_emb_att = num_emb_att
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_fmb = num_emb_fmb
        self.num_emb_in = num_emb_in
        self.norm = nn.LayerNorm(dim_emb)

        if num_emb_in != num_emb_lcb + num_emb_fmb*2 + num_emb_att:
            self.residual_projection = DLF_ResidualProjection(num_emb_in, num_emb_lcb + num_emb_fmb*2 + num_emb_att)
        else:
            self.residual_projection = nn.Identity()
        if base_emb_in != num_emb_in:
            self.x0_change_projection = DLF_ResidualProjection(base_emb_in, num_emb_in)
        else:
            self.x0_change_projection = nn.Identity()

    def forward(self, inputs: Tensor, x0: Tensor) -> Tensor:
        x0 = self.x0_change_projection(x0)
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)
        fmb_x0 = self.fmb_x0(inputs, x0)
        
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_att, dim_emb)
        if self.num_emb_in == self.num_emb_lcb + self.num_emb_fmb + self.num_emb_att:
            att = self.att(inputs[:,-self.num_emb_att:,:])
        else:
            att = self.att(inputs)

        # (bs, num_emb_lcb, dim_emb), (bs, num_emb_fmb, dim_emb), (bs, num_emb_att, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb)
        outputs = torch.concat((fmb, fmb_x0, lcb, att), dim=1)

        # (bs, num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb)
        outputs = self.norm(outputs + self.residual_projection(inputs))

        return outputs

    
class DLFLayer_att_self_gate(nn.Module):
    def __init__(
        self,
        num_emb_in: int,
        dim_emb: int,
        num_emb_lcb: int,
        num_emb_fmb: int,
        num_emb_att: int,
        rank_fmb: int,
        num_hidden: int,
        dim_hidden: int,
        dropout: float,
        num_heads: int,
        scale: bool,
    ) -> None:
        super().__init__()

        self.lcb = DLF_LinearCompressBlock(num_emb_in, num_emb_lcb)
        self.fmb = DLF_FactorizationMachineBlock(
            num_emb_in,
            num_emb_fmb,
            dim_emb,
            rank_fmb,
            num_hidden,
            dim_hidden,
            dropout,
        )
        self.att = DLF_MultiHeadSelfAttention(
            num_emb_att,
            num_emb_att,
            num_hidden,
            dim_hidden,
            dim_emb,
            attention_dim=dim_emb, 
            num_heads=num_heads, 
            dropout_rate=dropout,
            scale=scale,
        )
        self.num_emb_fmb = num_emb_fmb
        self.num_emb_lcb = num_emb_lcb
        self.num_emb_att = num_emb_att
        self.num_emb_out = num_emb_fmb + num_emb_lcb + num_emb_att
        self.norm = nn.LayerNorm(dim_emb)

        if num_emb_in != num_emb_lcb + num_emb_fmb + num_emb_att:
            self.residual_projection = DLF_ResidualProjection(num_emb_in, num_emb_lcb + num_emb_fmb + num_emb_att)
        else:
            self.residual_projection = nn.Identity()
                # Learnable query vectors for output channels
                
        self.output_queries = nn.Parameter(torch.randn(num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb))  # [num_emb_out, emb_dim]

        # Gate weights for mixing lcb, fmb, and att embeddings
        self.gate_fc = nn.Linear(dim_emb * 3, 3)  # Project concatenated embeddings to 3 gates

    def forward(self, inputs: Tensor) -> Tensor:
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_lcb, dim_emb)
        lcb = self.lcb(inputs)

        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_fmb, dim_emb)
        fmb = self.fmb(inputs)
        
        # (bs, num_emb_in, dim_emb) -> (bs, num_emb_att, dim_emb)
        att = self.att(inputs[:,-self.num_emb_att:,:])

        # Step 2: Combine all embeddings into a single tensor
        all_emb = torch.cat([lcb, fmb, att], dim=1)  # [batch_size, num_emb_lcb + num_emb_fmb + num_emb_att, emb_dim]

        # Step 3: Use learnable queries to generate output channels
        batch_size = inputs.size(0)
        queries = self.output_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_emb_out, emb_dim]

        # Step 4: Compute gate weights for each query and mix embeddings

        # Step 4.1: Compute attention weights for all queries
        attn_weights = torch.matmul(queries, all_emb.transpose(1, 2))  # [batch_size, num_emb_out, num_emb_lcb + num_emb_fmb + num_emb_att]
        attn_weights = F.softmax(attn_weights, dim=-1)  # Normalize attention weights along the last dimension

        # Step 4.2: Compute weighted sum of all embeddings for all queries
        attended_emb = torch.bmm(attn_weights, all_emb)  # [batch_size, num_emb_out, emb_dim]

        # Step 4.3: Compute gate values (shared across all queries)
        gate_inputs = torch.cat([lcb.mean(dim=1), fmb.mean(dim=1), att.mean(dim=1)], dim=-1)  # [batch_size, emb_dim * 3]
        gates = F.softmax(self.gate_fc(gate_inputs), dim=-1)  # [batch_size, 3]

        # Step 4.4: Compute mixed embeddings using gates
        mixed_emb = (
            gates[:, 0].unsqueeze(-1) * lcb.mean(dim=1) +  # [batch_size, emb_dim]
            gates[:, 1].unsqueeze(-1) * fmb.mean(dim=1) +  # [batch_size, emb_dim]
            gates[:, 2].unsqueeze(-1) * att.mean(dim=1)    # [batch_size, emb_dim]
        )  # [batch_size, emb_dim]

        # Step 4.5: Add the mixed embeddings to each attended embedding
        # Expand mixed_emb to match attended_emb's shape
        mixed_emb = mixed_emb.unsqueeze(1).expand(-1, self.num_emb_out, -1)  # [batch_size, num_emb_out, emb_dim]
        output_emb = attended_emb + mixed_emb  # [batch_size, num_emb_out, emb_dim]

        # (bs, num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb) -> (bs, num_emb_lcb + num_emb_fmb + num_emb_att, dim_emb)
        outputs = self.norm(output_emb + self.residual_projection(inputs))

        return outputs