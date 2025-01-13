import torch


def get_sparsity_loss(z, mask, level):
    """
    Exact sparsity loss in a batchwise sense.
    Inputs:
        z -- (batch_size, sequence_length)
        mask -- (batch_size, seq_length)
        level -- sparsity level
    """
    sparsity = torch.sum(z) / torch.sum(mask)
    # print('get_sparsity={}'.format(sparsity.cpu().item()))
    return torch.abs(sparsity - level)


def get_continuity_loss(z):
    """
    Compute the continuity loss.
    Inputs:
        z -- (batch_size, sequence_length)
    """
    return torch.mean(torch.abs(z[:, 1:] - z[:, :-1]))


def compute_micro_stats(labels, predictions):
    """
    Inputs:
        labels binary sequence indicates the if it is rationale
        predicitions -- sequence indicates the probability of being rationale

        labels -- (batch_size, sequence_length)
        predictions -- (batch_size, sequence_length) in soft probability

    Outputs:
        Number of true positive among predicition (True positive)
        Number of predicted positive (True pos + false pos)
        Number of real positive in the labels (true pos + false neg)
    """

    # threshold predictions
    predictions = (predictions > 0.5).long()

    # cal precision, recall
    num_true_pos = torch.sum(labels * predictions)
    num_predicted_pos = torch.sum(predictions)
    num_real_pos = torch.sum(labels)

    return num_true_pos, num_predicted_pos, num_real_pos


#def compute_detail_micro_stats(labels, predictions):
#   """
#    Inputs:
#        labels binary sequence indicates the if it is rationale
#        predicitions -- sequence indicates the probability of being rationale
#        labels -- (batch_size, sequence_length)
#        predictions -- (batch_size, sequence_length) in soft probability

#    Outputs:
#        Number of true positive among predicition (True positive) in each examples
#        Number of predicted positive (True pos + false pos) in each examples
#        Number of real positive in the labels (true pos + false neg) in each examples
#    """
#    labels = tf.cast(labels, tf.float32)
#    predictions = tf.cast(predictions, tf.float32)

#    threshold predictions
#    predictions = tf.cast(tf.greater_equal(predictions, 0.5), tf.float32)

#   (batch_size, )
#    true_pos = tf.reduce_sum(labels * predictions, axis=-1)
#    predicted_pos = tf.reduce_sum(predictions, axis=-1)
#    real_pos = tf.reduce_sum(labels, axis=-1)
#    return true_pos, predicted_pos, real_pos

def computer_pre_rec(pred, target):

    TP, TN, FN, FP = 0, 0, 0, 0
    TP += ((pred == 1) & (target == 1)).cpu().sum()

    TN += ((pred == 0) & (target == 0)).cpu().sum()

    FN += ((pred == 0) & (target == 1)).cpu().sum()

    FP += ((pred == 1) & (target == 0)).cpu().sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy