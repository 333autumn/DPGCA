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
        a = self.attn(x)  
        x = (x * a).sum(dim=1)  
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
        x = x.permute(1, 0, 2, 3)  
        x = torch.cat([self.attn1(_).unsqueeze(0) for _ in x])  # (S, N, D)
        s = x.permute(1, 0, 2)  
        c = self.biLSTM(s)[0]  
        # c = self.linear(c)
        g = self.attn2(c)  
        return s, g
