import torch
import torch.nn.functional as F

def acc_two_class(output, target, thredshold=0.5):
    """
    :param output:model classification output
    :param target:ground truth label
    :param threshold:正样本的阈值
    return 返回这个batch size的准确率
    """
    correct = 0
    for i in range(len(output)):
        _o = output[i]
        _t = target[i]
        label = 1 if _o > thredshold else 0
        if label == int(_t):
            correct += 1
    return 100 * correct / len(output)

def acc_binary_class(output, target, thredshold=0.5):
    """
    :param output:model classification output
    :param target:ground truth label
    :param threshold:正样本的阈值
    return 返回这个batch size的准确率,total,pos,neg
    """
    total_pos = 0
    total_neg = 0
    correct_pos = 0
    correct_neg = 0
    for i in range(len(output)):
        _o = output[i]
        _t = target[i]
        pred_label = 1 if _o > thredshold else 0
        if int(_t) == 1:
            total_pos += 1
            if pred_label == 1:
                correct_pos += 1
        else:
            total_neg += 1
            if pred_label == 0:
                correct_neg += 1
    total = total_pos + total_neg
    total_pos = 1 if total_pos== 0 else total_pos
    total_neg = 1 if total_neg== 0 else total_neg
    correct_total = correct_pos + correct_neg
    return (100 * correct_total / total), (100 * correct_pos / total_pos), (100 * correct_neg / total_neg)



def acc_two_class_image(output, target, path_list, thredshold=0.5):
    """
    返回每个batch size的评测结果
    :param output:model classification output
    :param target:ground truth label
    :param path_list：本次batch的图片列表
    :param threshold:正样本的阈值
    :return acc_image_list
    """
    acc_image_list = []
    correct = 0
    for i in range(len(target)):
        _o = float(output[i])  # 不可直接饮用outpu，会导致显存不能释放
        _t = target[i]
        pred_label = 1 if _o > thredshold else 0
        per_image = {'path': path_list[i],
                     "gt_label": int(target[i]),
                     "pred_label": pred_label,  # 预测的top k类别
                     "pred_score": _o,  # 每一类的预测值
                     "is_correct": True if pred_label == int(_t) else False}
        acc_image_list.append(per_image)
    return acc_image_list

def handle_binary_classification(output, target, path_list, acc_image_list, thredshold=0.5):
    """
    返回每个batch size的评测结果
    :param output:model classification output
    :param target:ground truth label
    :param path_list：本次batch的图片列表
    :param threshold:正样本的阈值
    :return acc_image_list
    """
    correct = 0
    for i in range(len(target)):
        _o = float(output[i])  # 不可直接饮用outpu，会导致显存不能释放
        _t = target[i]
        pred_label = 1 if _o > thredshold else 0
        per_image = {'path': path_list[i],
                     "gt_label": int(target[i]),
                     "pred_label": pred_label,  # 预测的top k类别
                     "pred_score": _o,  # 每一类的预测值
                     "is_correct": True if pred_label == int(_t) else False}
        acc_image_list.append(per_image)


def topk(output, target, top=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    :param output:model classification output
    :param target:ground truth
    :param top:计算前几的准确率
    :return:res[top_1, top_2, top_3]
    """
    with torch.no_grad():  # 将下面的tensor从计算图求导中排除
        maxk = max(top)
        batch_size = target.size(0)

        sorted_val, pred = output.topk(maxk, 1, True, True)  # 取前k个值
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def topk_with_class(output, target, path_list, topk=1):
    """
    获取每一类的topk准确率
    :param output: 模型的输出结果
    :param target: 训练集的分类标注
    :param path_list：本次batch的图片列表
    :param topk：topk准确率
    :return：
    acc_image_list：包含
    """
    acc_image_list = []

    with torch.no_grad():  # 将下面的tensor从计算图求导中排除
        sorted_val, pred = output.topk(output.shape[1], 1, True, True)  # 取前k个值

        for i in range(len(target)):
            per_image = {'path': path_list[i],
                         "label": int(target[i]),
                         "pred": pred[i],  # 预测的top k类别
                         "pred_score": sorted_val[i],  # 每一类的预测值
                         "is_correct": True if target[i] in pred[i][:topk] else False}
            acc_image_list.append(per_image)

        # 增加类别的
        return acc_image_list


def neighbork(output, target, k=10):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    :param output:model classification output
    :param target:ground truth
    :param top:计算前几的准确率
    :return:res[top_1, top_2, top_3]
    """
    with torch.no_grad():  # 将下面的tensor从计算图求导中排除
        batch_size = target.size(0)
        sorted_val, pred = output.topk(k, 1, True, True)  # 取前k个值
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
