class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # (N, L, D)
        a = self.attn(x)  # (N, L, 1)
        x = (x * a).sum(dim=1)  # (N, D)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, dim, hd_size):
        super().__init__()
        self.attn1 = SelfAttentionEncoder(dim)
        self.biLSTM = nn.LSTM(
            input_size=dim,
            hidden_size=hd_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(hd_size * 2, dim)
        self.attn2 = SelfAttentionEncoder(hd_size * 2)

    def forward(self, x):
        # (N, S, L, D) batch_size,max_sent_num,max_sent_len,embedding_size
        x = x.permute(1, 0, 2, 3)  # (S, N, L, D) max_sent_num,batch_size,max_sent_len,embedding_size
        # 第一层注意力处理
        x = torch.cat([self.attn1(_).unsqueeze(0) for _ in x])  # (S, N, D)
        s = x.permute(1, 0, 2)  # (N, S, D)
        c = self.biLSTM(s)[0]  # (N, S, D)
        # c = self.linear(c)  # 将 hidden_size*2 维度映射到 D=768
        g = self.attn2(c)  # (N, D)
        return s, g
