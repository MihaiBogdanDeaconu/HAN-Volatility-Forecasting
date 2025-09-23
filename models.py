import torch
import torch.nn as nn


class ScaleSpecificEncoder(nn.Module):
    """A standard Transformer encoder block dedicated to a single time scale."""
    def __init__(self, seq_len, input_dim, embed_dim, num_layers, num_heads, ff_dim, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        b = x.shape[0]
        x = self.embedding(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        return x[:, 0, :]

class HANTransformer(nn.Module):
    """The proposed Hierarchical Attention Network, with a stabilized regression head."""
    def __init__(self, config):
        super().__init__()
        self.short_encoder = ScaleSpecificEncoder(
            config.SEQ_LEN_SHORT, config.INPUT_DIM_SHORT, config.EMBED_DIM_SHORT,
            config.NUM_LAYERS_SHORT, config.NUM_HEADS_SHORT, config.FF_DIM_SHORT, config.DROPOUT
        )
        self.mid_encoder = ScaleSpecificEncoder(
            config.SEQ_LEN_MID, config.INPUT_DIM_MID, config.EMBED_DIM_MID,
            config.NUM_LAYERS_MID, config.NUM_HEADS_MID, config.FF_DIM_MID, config.DROPOUT
        )
        self.long_encoder = ScaleSpecificEncoder(
            config.SEQ_LEN_LONG, config.INPUT_DIM_LONG, config.EMBED_DIM_LONG,
            config.NUM_LAYERS_LONG, config.NUM_HEADS_LONG, config.FF_DIM_LONG, config.DROPOUT
        )

        self.proj_short = nn.Linear(config.EMBED_DIM_SHORT, config.FUSER_COMMON_DIM)
        self.proj_mid = nn.Linear(config.EMBED_DIM_MID, config.FUSER_COMMON_DIM)
        self.proj_long = nn.Linear(config.EMBED_DIM_LONG, config.FUSER_COMMON_DIM)
        self.norm = nn.LayerNorm(config.FUSER_COMMON_DIM)

        fuser_layer = nn.TransformerEncoderLayer(
            d_model=config.FUSER_COMMON_DIM, nhead=config.FUSER_NUM_HEADS,
            dim_feedforward=config.FUSER_COMMON_DIM * 4, dropout=config.DROPOUT,
            activation='gelu', batch_first=True
        )
        self.fuser_encoder = nn.TransformerEncoder(fuser_layer, num_layers=config.FUSER_NUM_LAYERS)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(config.FUSER_COMMON_DIM),
            nn.Linear(config.FUSER_COMMON_DIM, config.REG_HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.REG_HEAD_HIDDEN_DIM, config.REG_HEAD_HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Linear(config.REG_HEAD_HIDDEN_DIM // 2, 1),
        )

    def forward(self, x_short, x_mid, x_long, return_embedding=False):
        h_short = self.short_encoder(x_short)
        h_mid = self.mid_encoder(x_mid)
        h_long = self.long_encoder(x_long)

        h_short_proj = self.proj_short(h_short)
        h_mid_proj = self.proj_mid(h_mid)
        h_long_proj = self.proj_long(h_long)

        fuser_input = torch.stack([h_short_proj, h_mid_proj, h_long_proj], dim=1)
        fuser_input = self.norm(fuser_input)
        fused_output = self.fuser_encoder(fuser_input)

        final_representation = fused_output.mean(dim=1)

        if return_embedding:
            return final_representation

        prediction = self.regression_head(final_representation)
        return prediction

class FlatTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_short = nn.Linear(config.INPUT_DIM_SHORT, config.FLAT_EMBED_DIM)
        self.embed_mid = nn.Linear(config.INPUT_DIM_MID, config.FLAT_EMBED_DIM)
        self.embed_long = nn.Linear(config.INPUT_DIM_LONG, config.FLAT_EMBED_DIM)
        total_seq_len = config.SEQ_LEN_SHORT + config.SEQ_LEN_MID + config.SEQ_LEN_LONG
        self.pos_embedding = nn.Parameter(torch.randn(1, total_seq_len + 1, config.FLAT_EMBED_DIM))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.FLAT_EMBED_DIM))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.FLAT_EMBED_DIM, nhead=config.FLAT_NUM_HEADS,
            dim_feedforward=config.FLAT_FF_DIM, dropout=config.DROPOUT,
            activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.FLAT_NUM_LAYERS)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(config.FLAT_EMBED_DIM),
            nn.Linear(config.FLAT_EMBED_DIM, 1)
        )

    def forward(self, x_short, x_mid, x_long):
        b = x_short.shape[0]
        x_short_emb = self.embed_short(x_short)
        x_mid_emb = self.embed_mid(x_mid)
        x_long_emb = self.embed_long(x_long)
        x_flat_seq = torch.cat([x_short_emb, x_mid_emb, x_long_emb], dim=1)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x_flat_seq), dim=1)
        x = x + self.pos_embedding
        encoded = self.transformer_encoder(x)
        prediction = self.regression_head(encoded[:, 0, :])
        return prediction