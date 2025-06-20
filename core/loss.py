import chainer.functions as F

EPS = 2e-05

def focial_loss(pred, gt, alpha=2, beta=4, comm=None):
    th = 1
    pos_indices = gt >= th
    neg_indices = gt < th

    neg_weights = (1 - gt) ** beta

    pos_loss = F.log(pred + EPS) * (1 - pred) ** alpha * pos_indices
    neg_loss = F.log(1 - pred + EPS) * pred ** alpha * neg_weights * neg_indices

    num_pos = (gt >= 1).sum()
    pos_loss = F.sum(pos_loss)
    neg_loss = F.sum(neg_loss)

    loss = 0
    if comm is not None:
        num_pos = comm.allreduce_obj(num_pos)

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss, pos_loss, neg_loss

def reg_loss(output, mask, target, comm=None):
    ae = F.absolute_error(output, target)
    n_pos = mask.sum()

    if comm is not None:
        n_pos = comm.allreduce_obj(n_pos)

    return F.sum(ae * mask) / (n_pos + EPS)

def center_detection_loss(outputs, gts, hm_weight, wh_weight, offset_weight, focial_loss_alpha=2, focial_loss_beta=4, comm=None):
    hm_loss, wh_loss, offset_loss = 0, 0, 0
    hm_pos_loss, hm_neg_loss = 0, 0
    for output in outputs:
        output['hm'] = F.sigmoid(output['hm'])

        t_hm, t_pos, t_neg = focial_loss(output['hm'], gts['hm'], alpha=focial_loss_alpha, beta=focial_loss_beta, comm=comm)
        hm_loss += t_hm / len(outputs)
        hm_pos_loss += t_pos / len(outputs)
        hm_neg_loss += t_neg / len(outputs)

        if wh_weight > 0.0:
            wh_loss += reg_loss(output['wh'], gts['dense_mask'], gts['dense_wh'], comm=comm) / len(outputs)

        if offset_weight > 0.0:
            offset_loss += reg_loss(output['offset'], gts['dense_mask'], gts['dense_offset'], comm=comm) / len(outputs)

    loss = hm_weight * hm_loss + wh_weight * wh_loss + offset_weight * offset_loss

    return loss, hm_loss, wh_loss, offset_loss, {
        'hm_pos_loss': hm_pos_loss,
        'hm_neg_loss': hm_neg_loss,
    }
