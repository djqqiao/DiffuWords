from doctest import OutputChecker
from transformers import AutoConfig, BertConfig
# from transformers import BertEncoder

# from diffuwords.modeling_bart import BartModel
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
from diffuwords.BasicTransformers import BasicTransformerBlock
import torch
from transformers import (
    BertModel,
    BertConfig,
)

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        #added param
        # fix_encoder=False       
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size
        self.init_pretrained = 'no'

        cfg = BertConfig.from_pretrained(config_name)
        cfg.num_hidden_layers = 6
        self.input_transformers = BertModel.from_pretrained(config_name, config=cfg)

        config = BertConfig.from_pretrained(config_name)
        config.hidden_dropout_prob = self.dropout
        print(config)




        #embedding layer
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        # position embedding
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        
        # share weight between lm_head and word_embedding
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight
        
        # time embedding layer
        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        # input transform
        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(
                # #self_conditon
                nn.Linear(input_dims * 2, config.hidden_size),
                # nn.Linear(input_dims , config.hidden_size),
                nn.Tanh(), 
                nn.Linear(config.hidden_size, config.hidden_size)
            )


        # if self.init_pretrained == 'bert':
        #     print('initializing from pretrained bert...')
        #     print(config)
        #     temp_bert = BertModel.from_pretrained(config_name, config=config)

        #     self.word_embedding = temp_bert.embeddings.word_embeddings
        #     with th.no_grad():
        #         self.lm_head.weight = self.word_embedding.weight
        #     # self.lm_head.weight.requires_grad = False
        #     # self.word_embedding.weight.requires_grad = False
            
        #     self.input_transformers = temp_bert.encoder

        #     # position embedding
        #     self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        #     self.position_embeddings = temp_bert.embeddings.position_embeddings

        #     self.LayerNorm = temp_bert.embeddings.LayerNorm

        #     del temp_bert.embeddings
        #     del temp_bert.pooler

        # elif init_pretrained == 'no':

        #     self.input_transformers = BertEncoder(config)

        #     self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        #     self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #     self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # else:
        #     assert False, "invalid type of init_pretrained"


        # Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        
        config.num_hidden_layers = 6
        # define transformer model(6 layer)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.hidden_size // config.num_attention_heads,
                    dropout=config.hidden_dropout_prob,
                    cross_attention_dim=config.hidden_size,
                    activation_fn="geglu",
                )
                for d in range(config.num_hidden_layers)
            ]
        )


        # output transform
        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(), 
                nn.Linear(config.hidden_size, self.output_dims)
            )

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps,input_id_x, input_x_attention_mask,self_conditions = None):
        # 
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.            torch.Size([B, seq_len, embedding_dim])
        :param timesteps: a 1-D batch of timesteps.             torch.Size([B])
        :return: an [N x C x ...] Tensor of outputs.
        """
        if x.device != input_id_x.device:
            input_id_x = input_id_x.to(x.device)

        if x.device != input_x_attention_mask.device:
            input_x_attention_mask = input_x_attention_mask.to(x.device)        
        
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        #self-condition:
        if self_conditions is not None:
            
            x = th.concat((x, self_conditions), dim = -1)

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        

        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        decoder_hidden_state = self.dropout(self.LayerNorm(emb_inputs))

        # decoder_hidden_state = emb_inputs


        # with torch.no_grad():
        #     # input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        #     input_trans_hidden_states = self.input_transformers(input_ids=input_id_x,attention_mask=input_x_attention_mask).last_hidden_state


        # out = self.input_transformers.encoder(decoder_hidden_state)      
        out = self.input_transformers(input_ids=input_id_x,attention_mask=input_x_attention_mask)
        input_trans_hidden_states = out.last_hidden_state + 0 * out.pooler_output.unsqueeze(1)        


        # input_trans_hidden_states = out.last_hidden_state                        

        for block in self.transformer_blocks:
            decoder_hidden_state = block(decoder_hidden_state, input_trans_hidden_states)

        

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(decoder_hidden_state)
            # encoder_h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states

        h = h.type(x.dtype)


        return h


