import torch
import torch.nn as nn

class RefinedOmicsTransformer(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=256, num_heads=8,
                 num_layers=6, dropout=0.1, use_batch_norm=True, use_input_norm=True,
                 activation_fn="gelu", use_modality_embedding=True, pooling_type="attention"):
        super(RefinedOmicsTransformer, self).__init__()

        self.modalities = list(input_dims.keys())
        self.hidden_dim = hidden_dim
        self.use_batch_norm = use_batch_norm
        self.use_input_norm = use_input_norm
        self.use_modality_embedding = use_modality_embedding
        self.pooling_type = pooling_type
        self.activation_fn = self._get_activation(activation_fn)

        # Optional input normalization (per modality)
        self.input_norms = nn.ModuleDict()
        for omics, input_dim in input_dims.items():
            self.input_norms[omics] = nn.LayerNorm(input_dim) if use_input_norm else nn.Identity()

        # Embedding layers with optional normalization
        self.embeddings = nn.ModuleDict()
        for omics, input_dim in input_dims.items():
            layers = [nn.Linear(input_dim, hidden_dim)]
            layers.append(self._norm_layer(hidden_dim))
            layers.extend([
                self.activation_fn,
                nn.Dropout(dropout * 0.5)
            ])
            self.embeddings[omics] = nn.Sequential(*layers)

        self.modality_embeddings = nn.Parameter(
            torch.randn(len(input_dims), hidden_dim) * 0.02
        ) if use_modality_embedding else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation=activation_fn,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.pooling_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            self._norm_layer(hidden_dim * 2),
            self.activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            self._norm_layer(hidden_dim),
            self.activation_fn,
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, output_dim)
        )

        self.feature_importance = nn.Parameter(torch.ones(len(input_dims)))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _norm_layer(self, dim):
        if self.use_batch_norm:
            return nn.BatchNorm1d(dim)
        elif self.use_input_norm:
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()

    def _get_activation(self, name):
        if isinstance(name, str):
            name = name.lower()
            if name == "gelu":
                return nn.GELU()
            elif name == "relu":
                return nn.ReLU()
            elif name == "identity":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown activation: {name}")
        return name

    def forward(self, x_dict):
        batch_size = next(iter(x_dict.values())).size(0)

        modality_embeddings = []
        feature_weights = torch.softmax(self.feature_importance, dim=0)

        for i, omics in enumerate(self.modalities):
            x = self.input_norms[omics](x_dict[omics])
            x_embedded = self.embeddings[omics](x)

            if self.use_modality_embedding:
                modality_embed = self.modality_embeddings[i].unsqueeze(0).expand(batch_size, -1)
                x_embedded = x_embedded + modality_embed

            x_embedded = x_embedded * feature_weights[i]
            modality_embeddings.append(x_embedded.unsqueeze(1))

        x = torch.cat(modality_embeddings, dim=1)
        x_transformed = self.transformer(x)

        if self.pooling_type == "attention":
            query = self.pooling_query.expand(batch_size, -1, -1)
            pooled, attention_weights = self.attention_pooling(query, x_transformed, x_transformed)
            pooled = pooled.squeeze(1)
        elif self.pooling_type == "mean":
            pooled = x_transformed.mean(dim=1)
            attention_weights = torch.ones(batch_size, len(self.modalities)) / len(self.modalities)
        elif self.pooling_type == "max":
            pooled, _ = x_transformed.max(dim=1)
            attention_weights = torch.ones(batch_size, len(self.modalities)) / len(self.modalities)
        elif self.pooling_type == "concat":
            pooled = x_transformed.view(batch_size, -1)
            attention_weights = torch.ones(batch_size, len(self.modalities)) / len(self.modalities)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        output = self.prediction_head(pooled)

        return output, attention_weights.squeeze(1)
