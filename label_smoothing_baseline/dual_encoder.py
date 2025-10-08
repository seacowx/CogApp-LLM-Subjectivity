import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.models.deberta.modeling_deberta import ContextPooler


class DualEncoder(nn.Module):
    def __init__(
        self, 
        hidden_size=768,
        dropout_prob=0.1
    ):
        super().__init__()
        
        # Single encoder instead of two
        self.encoder = AutoModel.from_pretrained((
            '/scratch/prj/charnu/seacow_hf_cache/models--microsoft--deberta-v3-large/' 
            'snapshots/64a8c8eab3e352a784c658aef62be1662607476f'
        ))

        self.pooler = ContextPooler(self.encoder.config)
        
        # Get correct hidden size from pooler
        combined_hidden_size = self.pooler.output_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_hidden_size, 5),
        )

    def forward(self, context_ids, context_mask, auxiliary_ids, auxiliary_mask):
            
        # Encode context
        context_outputs = self.encoder(
            input_ids=context_ids,
            attention_mask=context_mask,
        )
        context_encoded = self.pooler(context_outputs[0])
        
        # Encode auxiliary info using same encoder
        auxiliary_outputs = self.encoder(
            input_ids=auxiliary_ids,
            attention_mask=auxiliary_mask
        )
        auxiliary_encoded = self.pooler(auxiliary_outputs[0])
        
        # Concatenate embeddings
        combined = torch.cat([context_encoded, auxiliary_encoded], dim=1)

        out = self.classifier(combined)

        return out
        
