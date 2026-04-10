import torch
import torch.nn.functional as F

def infonce_loss(query, keys, temperature):
    """
    Calculates InfoNCE loss using PyTorch operations.

    Args:
        query (torch.Tensor): Query embeddings (B, D)
        keys (torch.Tensor): Key embeddings (B, D). keys[i] is positive for query[i].
        temperature (float): Temperature scaling factor.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    B, D = query.shape

    # Normalize embeddings (important for cosine similarity via dot product)
    query = F.normalize(query, p=2, dim=-1)
    keys = F.normalize(keys, p=2, dim=-1)

    # Calculate similarity logits
    # (B, D) @ (D, B) -> (B, B)
    logits = torch.matmul(query, keys.T)

    # Scale by temperature
    logits /= temperature

    # Create labels: target index is 'i' for query 'i' (diagonal)
    labels = torch.arange(B, device=query.device)

    # Calculate cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss