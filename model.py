import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.distributions.binomial as binomial

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)

class CooperativeGame(nn.Module):
    def __init__(self, args):
        super(CooperativeGame, self).__init__()
        self.args = args


        self.embedding_layer = Embedding(args.vocab_size, args.embedding_dim, args.pretrained_embedding)


        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.z_dim = 2
        self.dropout = nn.Dropout(args.dropout)

        if args.share == 0:

            self.player1 = nn.Sequential(
            nn.GRU(input_size=args.embedding_dim,
                   hidden_size=args.hidden_dim // 2,
                   num_layers=args.num_layers,
                   batch_first=True,
                   bidirectional=True),
            SelectItem(0),
            nn.LayerNorm(args.hidden_dim),
            self.dropout,
            nn.Linear(args.hidden_dim, self.z_dim)).to('cuda:{}'.format(args.gpu))

            self.player2 = nn.Sequential(
                nn.GRU(input_size=args.embedding_dim,
                       hidden_size=args.hidden_dim // 2,
                       num_layers=args.num_layers,
                       batch_first=True,
                       bidirectional=True),
                SelectItem(0),
                nn.LayerNorm(args.hidden_dim),
                self.dropout,
                nn.Linear(args.hidden_dim, self.z_dim)).to('cuda:{}'.format(args.gpu))

            self.gen_list = [self.player1, self.player2]

        elif args.share == 1:

            self.shared_gru = nn.GRU(input_size=args.embedding_dim,
                                     hidden_size=args.hidden_dim // 2,
                                     num_layers=args.num_layers,
                                     batch_first=True,
                                     bidirectional=True)

            self.player1 = nn.Sequential(
                self.shared_gru,
                SelectItem(0),
                nn.LayerNorm(args.hidden_dim),
                self.dropout,
                nn.Linear(args.hidden_dim, self.z_dim)).to('cuda:{}'.format(args.gpu))

            self.player2 = nn.Sequential(
                self.shared_gru,
                SelectItem(0),
                nn.LayerNorm(args.hidden_dim),
                self.dropout,
                nn.Linear(args.hidden_dim, self.z_dim)).to('cuda:{}'.format(args.gpu))

            self.gen_list = [self.player1, self.player2]
            self.gen = self.shared_gru

    def _independent_soft_sampling(self, rationale_logits):
        z = torch.softmax(rationale_logits, dim=-1)
        return z

    def independent_straight_through_sampling(self, rationale_logits):
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)


        player1_logits = self.player1(embedding)
        player2_logits = self.player2(embedding)


        z_player1 = self.independent_straight_through_sampling(player1_logits)
        z_player2= self.independent_straight_through_sampling(player2_logits)


        cls_logits_list = []



        cls_embedding_player1 = embedding * (z_player1[:, :, 1].unsqueeze(-1))
        cls_outputs_player1, _ = self.cls(cls_embedding_player1)
        cls_outputs_player1 = cls_outputs_player1 * masks_ + (1. - masks_) * (-1e6)
        cls_outputs_player1 = torch.transpose(cls_outputs_player1, 1, 2)
        cls_outputs_player1, _ = torch.max(cls_outputs_player1, axis=2)
        cls_logits_player1 = self.cls_fc(self.dropout(cls_outputs_player1))
        cls_logits_list.append(cls_logits_player1)


        cls_embedding_player2 = embedding * (z_player2[:, :, 1].unsqueeze(-1))
        cls_outputs_player2, _ = self.cls(cls_embedding_player2)
        cls_outputs_player2 = cls_outputs_player2 * masks_ + (1. - masks_) * (-1e6)
        cls_outputs_player2 = torch.transpose(cls_outputs_player2, 1, 2)
        cls_outputs_player2, _ = torch.max(cls_outputs_player2, axis=2)
        cls_logits_player2 = self.cls_fc(self.dropout(cls_outputs_player2))
        cls_logits_list.append(cls_logits_player2)

        return [z_player1, z_player2], cls_logits_list

# Check if the second dimension of cls_logits matches the number of classes,if not, raise an AssertionError with the expected and actual shapes
        assert cls_logits.shape[
                   1] == self.args.num_class, f"Expected cls_logits shape (batch_size, {self.args.num_class}), got {cls_logits.shape}"

        return z, cls_logits

    def test(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)

        player1_logits = torch.softmax(self.player1(embedding), dim=-1)
        player2_logits = torch.softmax(self.player2(embedding), dim=-1)
        mean_logits = (player1_logits + player2_logits) / 2
        z_distribution = binomial.Binomial(1, mean_logits)
        z = z_distribution.sample()

        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))
        cls_outputs, _ = self.cls(cls_embedding)
        cls_outputs = cls_outputs * masks_ + (1 - masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        return z, cls_logits

    def test_one_head(self, inputs, masks):
        head_1 = self.player1
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)
        gen_logits = head_1(embedding)

        z = self.independent_straight_through_sampling(gen_logits)
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))
        cls_outputs, _ = self.cls(cls_embedding)
        cls_outputs = cls_outputs * masks_ + (1 - masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        return z, cls_logits

    def get_cls_param(self):
        layers = [self.cls, self.cls_fc]
        params = []
        for layer in layers:
            params.extend([param for param in layer.parameters() if param.requires_grad])
        return params

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.cls_fc(self.dropout(outputs))
        return logits
