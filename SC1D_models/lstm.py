import torch.nn as nn

class LSTM(nn.Module) :
    def __init__(self, input_size, hidden_size, seq_length, num_classes, bidirectional=True):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional)
        expand_ratio = 2 if bidirectional else 1
        self.linear = nn.Linear(expand_ratio * hidden_size * seq_length, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        batch_size, seq_len, _ = x.size()
        out, _ = self.lstm(x)
        out = self.linear(out.contiguous().view(batch_size,-1))

        return out