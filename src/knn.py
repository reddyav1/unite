import torch
import torch.nn.functional as F

def evaluate_knn_accuracy(model, source_loader, target_loader, device, args):

    src_embedding = torch.zeros([len(source_loader.dataset), 768]).to(device)
    src_labels = torch.zeros(len(source_loader.dataset)).long().to(device)
    model.eval()
    model_select = model.module if args.distributed else model
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(source_loader):
            data = data.to(device)
            latent_s, _, ids_restore_s = model_select.forward_encoder(data, mask_ratio=0.0)            
            emb = latent_s[:, 0, :]
            
            src_embedding[batch_idx*args.batch_size:batch_idx*args.batch_size+data.size(0), :] = emb
            src_labels[batch_idx*args.batch_size:batch_idx*args.batch_size+data.size(0)] = target.to(device)

    K = 7
    total, top1 = 0, 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(target_loader):            
            inputs = inputs.to(device)
            batchSize = inputs.size(0)
            latent_t, _, ids_restore_t = model_select.forward_encoder(inputs, mask_ratio=0.0)
            emb = latent_t[:, 0, :]
            dist = -torch.cdist(emb, src_embedding)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            pred_labels, _ = torch.mode(src_labels[yi], dim=-1)
            correct = pred_labels.eq(targets.to(device))
            top1 = top1 + correct.sum().item()
            total += targets.size(0)
    top1_acc = top1*100. / total

    print('tgt CD-kNN acc. on {} tgt examples: Top-1={:.2f}%'.format(total, top1_acc))
    return top1_acc

def compute_ece(softmaxes, labels, n_bins=10):
    """
    Compute the expected calibration error (ECE) of a model.
    
    Args:
    - softmaxes (torch.Tensor): The tensor of softmax distributions from the model.
      Shape should be (N, C) where N is the number of samples and C is the number of classes.
    - labels (torch.Tensor): The tensor of true labels. Shape should be (N,).
    - n_bins (int): The number of bins to use for calibration.
    
    Returns:
    - ece (float): The expected calibration error.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=softmaxes.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Masks for data points that fall into the current bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # Proportion of data in this bin

        if prop_in_bin.item() > 0:
            # Average accuracy of data points in the bin
            accuracy_in_bin = accuracies[in_bin].float().mean()
            # Average confidence of data points in the bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            # ECE contribution for this bin is the absolute difference times the proportion of data in the bin
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()