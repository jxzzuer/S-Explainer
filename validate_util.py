import torch

from metric import compute_micro_stats
import torch.nn.functional as F


def validate(generator, classifier, annotation_loader, device):
    """
        Validate the performance of the generator and classifier on the annotation dataset.

        Args:
            generator: The generator model.
            classifier: The classifier model.
            annotation_loader: DataLoader providing batches of inputs, masks, labels, and annotations.
            device: The device (CPU or GPU) to run the models on.

        Returns:
            Tuple containing sparsity, micro precision, micro recall, and micro F1 score.
        """
    # Initialize counters for micro statistics
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    # Initialize counters for classification metrics
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        rationales = generator(inputs, masks)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales[:, :, 1])

        # cls
        logits = classifier(inputs, masks, z=rationales[:, :, -1])

        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)


        TP += ((pred == 1) & (labels == 1)).cpu().sum()

        TN += ((pred == 0) & (labels == 0)).cpu().sum()

        FN += ((pred == 0) & (labels == 1)).cpu().sum()

        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

    # Calculate micro precision, recall, F1 score, and sparsity
    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # Calculate classification precision, recall, F1 score, and accuracy
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f}".format(recall, precision,
                                                                                       f1_score))
    return sparsity, micro_precision, micro_recall, micro_f1


def validate_att(model, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        att_score, rationales, cls_logits = model(inputs, masks)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)


        TP += ((pred == 1) & (labels == 1)).cpu().sum()

        TN += ((pred == 0) & (labels == 0)).cpu().sum()

        FN += ((pred == 0) & (labels == 1)).cpu().sum()

        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)

    print("annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f}".format(recall, precision,
                                                                                       f1_score))
    return sparsity, micro_precision, micro_recall, micro_f1


def validate_share(model, annotation_loader, device):
    """
        Validate the performance of the shared model on the annotation dataset.

        Args:
            model: The shared model.
            annotation_loader: DataLoader providing batches of inputs, masks, labels, and annotations.
            device: The device (CPU or GPU) to run the models on.

        Returns:
            Tuple containing sparsity, micro precision, micro recall, and micro F1 score.
        """
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        rationales, cls_logits = model.test(inputs, masks)

        # Compute micro statistics for the generated rationales
        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales[:, :, 1])

        # Compute softmax probabilities and predictions
        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

        # Update classification metrics

        TP += ((pred == 1) & (labels == 1)).cpu().sum()

        TN += ((pred == 0) & (labels == 0)).cpu().sum()

        FN += ((pred == 0) & (labels == 1)).cpu().sum()

        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        # Update micro statistics
        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

    # Calculate micro precision, recall, F1 score, and sparsity
    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # Calculate classification precision, recall, F1 score, and accuracy
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(
        "annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))
    return sparsity, micro_precision, micro_recall, micro_f1

def validate_onehead(model, annotation_loader, device):
    num_true_pos = 0.
    num_predicted_pos = 0.
    num_real_pos = 0.
    num_words = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        rationales, cls_logits = model.test_one_head(inputs, masks)

        num_true_pos_, num_predicted_pos_, num_real_pos_ = compute_micro_stats(
            annotations, rationales[:, :, 1])

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)


        TP += ((pred == 1) & (labels == 1)).cpu().sum()

        TN += ((pred == 0) & (labels == 0)).cpu().sum()

        FN += ((pred == 0) & (labels == 1)).cpu().sum()

        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        num_true_pos += num_true_pos_
        num_predicted_pos += num_predicted_pos_
        num_real_pos += num_real_pos_
        num_words += torch.sum(masks)

    micro_precision = num_true_pos / num_predicted_pos
    micro_recall = num_true_pos / num_real_pos
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision +
                                                       micro_recall)
    sparsity = num_predicted_pos / num_words

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(
        "annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))
    return sparsity, micro_precision, micro_recall, micro_f1



def validate_annotation_sentence(model, annotation_loader, device):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(device), annotations.to(
            device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)


        TP += ((pred == 1) & (labels == 1)).cpu().sum()

        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        #
        FN += ((pred == 0) & (labels == 1)).cpu().sum()

        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(
        "annotation dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                     f1_score,
                                                                                                     accuracy))


def validate_dev_sentence(model, dev_loader, device,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    writer,epoch=writer_epoch
    for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales -- (batch_size, seq_length, 2)
        cls_logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(cls_logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)


        TP += ((pred == 1) & (labels == 1)).cpu().sum()

        TN += ((pred == 0) & (labels == 0)).cpu().sum()

        FN += ((pred == 0) & (labels == 1)).cpu().sum()

        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("dev dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                f1_score, accuracy))
    writer.add_scalar('./sent_acc',accuracy,epoch)
    return f1_score,accuracy


def validate_rationales(model, annotation_loader, device,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    writer,epoch=writer_epoch
    for (batch, (inputs, masks, labels,
                 annotations)) in enumerate(annotation_loader):
        inputs, masks, labels, annotations = inputs.to(device), masks.to(device), labels.to(
            device), annotations.to(
            device)

        masks = annotations
        logits = model.train_one_step(inputs, masks)

        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)


        TP += ((pred == 1) & (labels == 1)).cpu().sum()

        TN += ((pred == 0) & (labels == 0)).cpu().sum()

        FN += ((pred == 0) & (labels == 1)).cpu().sum()

        FP += ((pred == 1) & (labels == 0)).cpu().sum()

    # cls
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer.add_scalar('rat_acc',accuracy,epoch)
    print("rationale dataset : recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(recall, precision,
                                                                                                      f1_score,
                                                                                                      accuracy))
