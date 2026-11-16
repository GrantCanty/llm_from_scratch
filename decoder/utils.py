import torch


def unpack_batch(batch):
    """Accept notebook dictionaries and existing (inputs, targets) loaders."""
    if isinstance(batch, dict):
        return batch["input_ids"], batch["labels"], batch.get("attention_mask")
    input_batch, target_batch = batch
    return input_batch, target_batch, None


def calc_loss_batch(input_batch, target_batch, model, device, attention_mask=None, return_logits=False):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        logits = model(input_batch, attention_mask=attention_mask)
    else:
        logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100)
    return (loss, logits) if return_logits else loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    return calc_metrics_loader(data_loader, model, device, num_batches)


@torch.no_grad()
def calc_metrics_loader(data_loader, model, device, num_batches=None):
    """Calculate token-weighted loss and accuracy; caller selects eval mode."""
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for i, batch in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        input_batch, target_batch, attention_mask = unpack_batch(batch)
        targets = target_batch.to(device)
        valid = targets != -100
        count = valid.sum().item()
        if count == 0:
            continue
        loss, logits = calc_loss_batch(
            input_batch, targets, model, device, attention_mask, return_logits=True
        )
        total_loss += loss.item() * count
        total_correct += ((logits.argmax(dim=-1) == targets) & valid).sum().item()
        total_tokens += count
    return {
        "loss": total_loss / total_tokens if total_tokens else float('nan'),
        "accuracy": total_correct / total_tokens if total_tokens else float('nan'),
    }

def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    start_idx_len = len(idx.squeeze(0))
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logs, _ = torch.topk(logits, top_k)
            min_val = top_logs[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
            
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            #dx = torch.cat((idx, idx_next), dim=-1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    #idx = idx.squeeze(0)[start_idx_len:].unsqueeze(0)
    idx = idx[:, start_idx_len:]
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(tokens, tokenizer):
    flat = tokens.squeeze(0)
    return tokenizer.decode(flat.tolist())
