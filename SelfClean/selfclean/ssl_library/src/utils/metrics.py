import torch


def get_seg_metrics(
    mask: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    mode: str = "binary",
):
    import segmentation_models_pytorch as smp

    tp, fp, fn, tn = smp.metrics.get_stats(
        output=mask.int() if mode == "binary" else mask.argmax(dim=1).int(),
        target=target.int(),
        mode=mode,
        num_classes=n_classes,
        threshold=0.5 if mode == "binary" else None,
    )
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
    return iou_score, f1_score


def calculate_embedding_entropy(embeddings: torch.Tensor):
    embeddings = (embeddings - torch.min(embeddings, dim=0).values) + 1e-7
    embedding_dist = embeddings / torch.sum(embeddings, dim=0)
    entropy_mat = torch.sum((embedding_dist * torch.log(embedding_dist)), dim=0)
    ent_avg = -torch.mean(entropy_mat)
    ent_min = -torch.min(entropy_mat)
    ent_max = -torch.max(entropy_mat)
    ent_med = -torch.median(entropy_mat)
    ent_std = torch.std(entropy_mat)
    return ent_avg, ent_min, ent_max, ent_std, ent_med


def calculate_student_teacher_acc(teacher_output, student_output, n_g_crops):
    # check if the outputs are tuples or not
    # if yes, use the first element (iBOT)
    if type(teacher_output) == tuple and type(student_output) == tuple:
        probs1 = teacher_output[0].chunk(n_g_crops)
        probs2 = student_output[0].chunk(n_g_crops)
    # DINO
    else:
        probs1 = teacher_output.chunk(n_g_crops)
        probs2 = student_output.chunk(n_g_crops)
    pred1 = probs1[0].max(dim=1)[1]
    pred2 = probs2[1].max(dim=1)[1]
    acc = (pred1 == pred2).sum() / pred1.size(0)
    return acc
