def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def top_decode(model, tokenizer, min_length=1, max_length=40, temperature=0.9, top_k=100, top_p=0.9):
    # start decoding. Start from BOS
    current_output = [tokenizer.bos_token_id]
    for i in range(max_length):
        # TODO: Prepare input_ids to feed into transformer LM model.
        # Shape: (B=1, N), where N is a variable = len(current_output)
        # 1 line of code
        input_ids = torch.Tensor(current_output).to(device).reshape(1, -1)

        # TODO: feed input_ids into model, producing logits
        # 1 line of code
        logits = model(input_ids)[0]
        # Scale the logits by temperature to "flatten" the probs
        # Apply top_filtering on logits
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)

        # TODO: Calculate probs from the adjusted logits
        # 1 line of code
        probs = F.softmax(logits, dim=-1)

        # TODO: randomly sample 1 word from probs
        # 1 line of code
        prev = torch.multinomial(probs, 1)  # Use multinomial to sample according to distribution

        # Detect pre-mature end of sentence
        if i < min_length and prev.item() == tokenizer.eos_token_id:
            while prev.item() == tokenizer.eos_token_id:
                if probs.max().item() == 1:
                    break
                # TODO: re-sample 1 word from probs
                # 1 line of code
                prev = torch.multinomial(probs, 1)

        # TODO: handle exit condition
        # ~2 line of code
        if prev.item() == tokenizer.eos_token_id:
            current_output.append(prev.item())
            break
        # TODO: add the sampled word ID into buffer
        # ~1 line of code
        current_output.append(prev.item())
    return current_output


# generate 5 sentences randomly
for j in range(5):
    print(tokenizer.decode(top_decode(model, tokenizer)))

!date
