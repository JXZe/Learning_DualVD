import torch
from torch import nn


class MemNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        fusion_size = config["img_feature_size"] + config["lstm_hidden_size"]
        self.fusion_img_ques = nn.Linear(fusion_size, config["lstm_hidden_size"])

        self.q2h_att_mn = nn.Linear(
            config["lstm_hidden_size"],
            1
        )

        self.hist_mn_layer = nn.Linear(
            config["lstm_hidden_size"],
            config["lstm_hidden_size"]
        )

        self.dropout = nn.Dropout(p=config["dropout"])




        # initialization
        nn.init.kaiming_uniform_(self.fusion_img_ques.weight)
        nn.init.constant_(self.fusion_img_ques.bias, 0)

        nn.init.kaiming_uniform_(self.q2h_att_mn.weight)
        nn.init.constant_(self.q2h_att_mn.bias, 0)

        nn.init.kaiming_uniform_(self.hist_mn_layer.weight)
        nn.init.constant_(self.hist_mn_layer.bias, 0)

    def forward(self, img, ques, hist, batch_size, num_rounds):

        #############################################################################
        ##                          Image Question Fusion                          ##
        #############################################################################
        fused_vector = torch.cat((img, ques), 1)
        fused_vector = self.dropout(fused_vector)
        fused_embedding = torch.tanh(self.fusion_img_ques(fused_vector))

        #############################################################################
        ##                          Memory Network Fusion                          ##
        #############################################################################
        dim = fused_embedding.size(-1)
        ques_embed_2_hist_mn = fused_embedding.view(batch_size, num_rounds, -1)
        ques_embed_2_hist_mn = ques_embed_2_hist_mn.repeat(1, 1, num_rounds).view(batch_size, -1, dim)
        ques_embed_2_hist_mn = ques_embed_2_hist_mn.view(-1, dim)
        hist_new = hist.view(batch_size * num_rounds * num_rounds, -1)

        hist_w_mn = hist_new * ques_embed_2_hist_mn
        hist_w_mn = self.dropout(hist_w_mn)
        hist_w_mn = self.q2h_att_mn(hist_w_mn)
        hist_w_mn = hist_w_mn.view(batch_size * num_rounds, num_rounds)
        hist_w_mn = torch.softmax(hist_w_mn, 1).view(batch_size * num_rounds, num_rounds, 1)
        hist_new = hist_new.view(batch_size * num_rounds, num_rounds, -1)
        hist_embed = torch.sum((hist_w_mn * hist_new), 1)

        hist_embed = self.dropout(hist_embed)
        hist_embed = torch.tanh(self.hist_mn_layer(hist_embed))
        fused_embedding = fused_embedding + hist_embed
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)

        return fused_embedding