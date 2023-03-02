# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.distributions.categorical import Categorical
from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
    transformer,
)

@register_strategy('denoise_predict')
class DenoisePredict(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        self.n_unroll_step = args.n_unroll_step
        self.temperature = args.temperature
    
    def generate(self, model, encoder_out, tgt_tokens, tgt_dict):
        vocab_size = model.encoder.embed_tokens.weight.shape[0]
        bsz, seq_len = tgt_tokens.size() # tgt_tokens: 4-Real, 1-Pad
        pad_mask = tgt_tokens.eq(tgt_dict.pad()) #tgt_dict.pad()=1, pad_mask=(need pad)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs = self.generate_non_autoregressive(model, encoder_out, tgt_tokens)
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
        #assign_single_value_byte(token_probs, pad_mask, 1.0)
        #print("Initialization: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        for counter in range(1, iterations):
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()
            
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens = self.corrupt_text(tgt_tokens, mask_ind, vocab_size)
            
            #assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
            #assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())

            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[0]))
            for i in range(0, self.n_unroll_step - 1):
                decoder_out = model.decoder(tgt_tokens, encoder_out)
                tgt_tokens = Categorical(logits=decoder_out[0] / self.temperature).sample().detach()
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            
            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            
            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs
    
    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        decoder_out = model.decoder(tgt_tokens, encoder_out)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)
    
    def get_random_text(self, shape, vocab_size):
        # --max-tokens
        return torch.randint(vocab_size, shape)
    
    def random_corrupt_text_mask(self, batched_text, ):
        corruption_prob_per_sequence = torch.rand((batched_text[0], 1)) 
        rand = torch.rand(batched_text.shape) # From a uniform distribution [0, 1)
        mask = (rand < corruption_prob_per_sequence).to(batched_text.device)
        return mask
        
    def corrupt_text(self, batched_text, mask_ind, vocab_size):
        
        mask = mask_ind.new(mask_ind.shape).fill_(0)
        mask.scatter_(-1, mask_ind, 1)
        random_text = self.get_random_text(batched_text.shape, vocab_size).to(batched_text.device)
        return mask * random_text + (1 - mask) * batched_text
        
    def logits_fn(self, batched_text, mask, vocab_size, n_unrolled_steps, enable_sampling):
        seq_len = len(batched_text[0])
        
        def unrolled_fn(batched_text, mask):
            #samples = self.corrupt_text(batched_text, mask)
            samples = batched_text
            all_logits = []
            for _ in range(n_unrolled_steps):
                logits = self.fn(samples, 'Transformer')
                samples = Categorical(logits=logits).sample().detach()
                all_logits += [logits]
            final_logits = torch.cat(all_logits, axis=0)
            return final_logits
        
        if enable_sampling:
            return self.fn(batched_text, 'linear')
        else:
            return unrolled_fn(batched_text, mask)
    
    def sampling_fn(self, logits_fn, steps, batch_size, seq_len, vocab_size, temperature=0.8):
        batched_text = self.get_random_text([batch_size, seq_len], vocab_size) 
        for _ in range(steps):
            logits = logits_fn(batched_text)
            samples = Categorical(logits=logits / temperature).sample()
            batched_text = samples
            return batched_text
        
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        return self.w_2(self.dropout(F.relu(self.w_1(x))))