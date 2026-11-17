import torch
from . import transformer
from . import utils
import numpy as np
import math


class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention_backend = cfg.get("attention_backend", "sdpa")
        self.tok_emb = torch.nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = torch.nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = torch.nn.Dropout(0)
        self.trf_blocks = torch.nn.Sequential(
            *[transformer.TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = torch.nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )
    
    def forward(self, in_idx, attention_mask=None, document_ids=None, position_ids=None):
        # Keep master parameters in FP32; autocast covers training, eval and generation.
        if self.attention_backend == "fa4":
            if in_idx.device.type != "cuda":
                raise ValueError("The fa4 backend requires a CUDA device")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._forward(in_idx, attention_mask, document_ids, position_ids)
        return self._forward(in_idx, attention_mask, document_ids, position_ids)

    def _forward(self, in_idx, attention_mask=None, document_ids=None, position_ids=None):
        batch_size, seq_len = in_idx.shape
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError("Input sequence exceeds the model context length")
        if attention_mask is not None:
            if attention_mask.shape != in_idx.shape:
                raise ValueError("attention_mask must have the same shape as input IDs")
            attention_mask = attention_mask.to(device=in_idx.device, dtype=torch.bool)
            if not attention_mask[:, 0].all():
                raise ValueError("Use nonempty, right-padded sequences")
        if document_ids is not None:
            document_ids = document_ids.to(in_idx.device)
            if document_ids.shape != in_idx.shape:
                raise ValueError("document_ids must have the same shape as input IDs")
            valid = attention_mask if attention_mask is not None else torch.ones_like(in_idx, dtype=torch.bool)
            if (document_ids[valid] < 0).any() or ((document_ids[:, 1:] < document_ids[:, :-1]) & valid[:, 1:]).any():
                raise ValueError("Valid document_ids must be nonnegative and nondecreasing")
            offsets = torch.arange(seq_len, device=in_idx.device).expand_as(in_idx)
            starts = torch.cat((torch.ones_like(in_idx[:, :1], dtype=torch.bool),
                                document_ids[:, 1:] != document_ids[:, :-1]), dim=1)
            expected = offsets - torch.where(starts, offsets, 0).cummax(1).values
            expected = expected.masked_fill(~valid, 0)
            if position_ids is not None and not torch.equal(position_ids.to(in_idx.device), expected):
                raise ValueError("position_ids must reset at each visible document segment")
            position_ids = expected
        elif position_ids is None:
            position_ids = torch.arange(seq_len, device=in_idx.device)
        position_ids = position_ids.to(in_idx.device)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(position_ids)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, attention_mask=attention_mask, document_ids=document_ids)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-8
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift
    

class GELU(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2/torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            torch.nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)
    

def train_model_simple_function(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, val_perplexities, track_tokens_seen = [], [], [], []
    train_loss_by_pos_list, val_loss_by_pos_list = [], []
    val_accuracies = []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_batch, target_batch, attention_mask, document_ids, position_ids = utils.unpack_batch(batch)
            if not (target_batch != -100).any():
                continue
            optimizer.zero_grad()
            loss = utils.calc_loss_batch(input_batch, target_batch, model, device, attention_mask,
                                         document_ids=document_ids, position_ids=position_ids)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
                error_if_nonfinite=True
            )

            optimizer.step()
            tokens_seen += (target_batch != -100).sum().item()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, train_loss_by_pos, val_loss, val_loss_by_pos, val_perplexity, val_accuracy = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                val_accuracies.append(val_accuracy)

                track_tokens_seen.append(tokens_seen)

                train_loss_by_pos_list.append(train_loss_by_pos.detach().cpu())

                val_loss_by_pos_list.append(val_loss_by_pos.detach().cpu())

                print(f'epoch {epoch+1} (step {global_step:06d})\ntrain_loss: {train_loss:.3f} val_loss: {val_loss:.3f} val_perplexity: {val_perplexity:.2f} val_accuracy: {val_accuracy:.2%} grad_norm: {grad_norm:.2f}')
    
    return train_losses, train_loss_by_pos_list, val_losses, val_loss_by_pos_list, val_perplexities, val_accuracies, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = utils.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_metrics = utils.calc_metrics_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()

    train_loss_by_pos = train_loss['loss_by_pos']
    train_loss = train_loss['loss']

    val_loss = val_metrics["loss"]
    val_loss_by_pos = val_metrics['loss_by_pos']
    val_perplexity = math.exp(val_loss)
    val_accuracy = val_metrics["accuracy"]

    return train_loss, train_loss_by_pos, val_loss, val_loss_by_pos, val_perplexity, val_accuracy

def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=15, top_k=25, temperature=1):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = utils.text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = utils.generate_text_simple(model, idx=encoded, max_new_tokens=max_new_tokens, context_size=context_size, top_k=top_k, temperature=temperature)
        decoded_text = utils.token_ids_to_text(token_ids, tokenizer)
        #print(decoded_text.replace('\n', ' '))
    model.train()
    t = decoded_text.replace('\n', ' ')
    return t

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])