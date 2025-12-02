from torch.nn import functional as F
import torch

R = 6371.0  # Earth radius in km
def haversine_loss(pred, target):
    # pred, target: [B, 2] -> [lat, lon] in degrees
    pred_lat = torch.deg2rad(pred[:, 0])
    pred_lon = torch.deg2rad(pred[:, 1])
    true_lat = torch.deg2rad(target[:, 0])
    true_lon = torch.deg2rad(target[:, 1])

    dlat = pred_lat - true_lat
    dlon = pred_lon - true_lon

    a = torch.sin(dlat/2)**2 + torch.cos(true_lat) * torch.cos(pred_lat) * torch.sin(dlon/2)**2
    c = 2 * torch.arcsin(torch.clamp(torch.sqrt(a), 0.0, 1.0))

    d = R * c  # [B] distance in km
    return d.mean()

def multitask_loss(outputs, batch, weights=None):
    """
    outputs: dict from model.forward
    batch:   dict from DataLoader
    weights: optional dict of loss weights
    """
    if weights is None:
        weights = {
            "s3": 1.0,
            "s16": 1.0,
            "s365": 1.0,
        }

    loss_total = 0.0

    # 1) Geo-cell classification losses
    s3_logits = outputs["logits_s3"]
    s16_logits = outputs["logits_s16"]
    s365_logits = outputs["logits_s365"]

    s3 = batch["s3_label"].long().to(s3_logits.device)
    s16 = batch["s16_label"].long().to(s16_logits.device)
    s365 = batch["s365_label"].long().to(s365_logits.device)

    loss_s3 = F.cross_entropy(s3_logits, s3)
    loss_s16 = F.cross_entropy(s16_logits, s16)
    loss_s365 = F.cross_entropy(s365_logits, s365)

    loss_total += weights["s3"] * loss_s3
    loss_total += weights["s16"] * loss_s16
    loss_total += weights["s365"] * loss_s365

    # For logging if you want:
    aux_losses = {
        "loss_s3": loss_s3.detach(),
        "loss_s16": loss_s16.detach(),
        "loss_s365": loss_s365.detach(),
    }

    return loss_total, aux_losses
