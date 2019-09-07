
# make modifies and fixed bug from https://zhuanlan.zhihu.com/p/56710152


```python
#reference
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
```

# VGG模型


```python
class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.features = self._make_layers(cfg)
        self._rpn_model()

        size = (7, 7)
        self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(size[0], size[1])
        self.roi_classifier()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        # layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)]
        return nn.Sequential(*layers)
        # return layers

    def _rpn_model(self, mid_channels=512, in_channels=512, n_anchor=9):
        self.rpn_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        # conv sliding layer
        self.rpn_conv.weight.data.normal_(0, 0.01)
        self.rpn_conv.bias.data.zero_()

        # Regression layer
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()

        # classification layer
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()

    def forward(self, data):
        out_map = self.features(data)
        # for layer in self.features:
        #     # print layer
        #     data = layer(data)
        #     # print data.data.shape
        #
        # # out = data.view(data.size(0), -1)
        x = self.rpn_conv(out_map)
        pred_anchor_locs = self.reg_layer(x)  # 回归层，计算有效anchor转为目标框的四个系数
        pred_cls_scores = self.cls_layer(x)  # 分类层，判断该anchor是否可以捕获目标

        return out_map, pred_anchor_locs, pred_cls_scores

    def roi_classifier(self, class_num=20):  # 假设为VOC数据集，共20分类
        # 分类层
        self.roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096),
                                                   nn.ReLU(),
                                                   nn.Linear(4096, 4096),
                                                   nn.ReLU()])
        self.cls_loc = nn.Linear(4096, (class_num+1) * 4)  # (VOC 20 classes + 1 background. Each will have 4 co-ordinates)
        self.cls_loc.weight.data.normal_(0, 0.01)
        self.cls_loc.bias.data.zero_()


        self.score = nn.Linear(4096, class_num+1)  # (VOC 20 classes + 1 background)

    def rpn_loss(self, rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_label, weight=10.0):
        # 对与classification我们使用Cross Entropy损失
        gt_rpn_label = torch.autograd.Variable(gt_rpn_label.long())
        rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        # print(rpn_cls_loss)  # Variable containing: 0.6931

        # 对于 Regression 我们使用smooth L1 损失
        pos = gt_rpn_label.data > 0  # Regression 损失也被应用在有正标签的边界区域中
        mask = pos.unsqueeze(1).expand_as(rpn_loc)
        # print(mask.shape)  # (22500L, 4L)

        # 现在取有正数标签的边界区域
        mask_pred_loc = rpn_loc[mask].view(-1, 4)
        mask_target_loc = gt_rpn_loc[mask].view(-1, 4)
        print(mask_pred_loc.shape, mask_target_loc.shape)  # ((18L, 4L), (18L, 4L))

        # regression损失应用如下
        rpn_loc_loss = torch.nn.functional.smooth_l1_loss(mask_pred_loc.float(), mask_target_loc.float(), 
                                                          reduction='mean')
        rpn_loss = rpn_cls_loss + (weight * rpn_loc_loss.data)
        # print("rpn_loss: {}".format(rpn_loss))  # 1.33919757605
        return rpn_loss

    def roi_loss(self, pre_loc, pre_conf, target_loc, target_conf, weight=10.0):
        # 分类损失
        target_loc = target_loc.to(device)
        target_conf = target_conf.to(device)
        target_conf = torch.autograd.Variable(target_conf.long())
        pred_conf_loss = torch.nn.functional.cross_entropy(pre_conf, target_conf, ignore_index=-1)
        # print(pred_conf_loss)  # Variable containing:  3.0515

        #  对于 Regression 我们使用smooth L1 损失
        # 用计算RPN网络回归损失的方法计算回归损失
        # pre_loc_loss = REGLoss(pre_loc, target_loc)
        pos = target_conf.data > 0  # Regression 损失也被应用在有正标签的边界区域中
        mask = pos.unsqueeze(1).expand_as(pre_loc)  # (128, 4L)

        # 现在取有正数标签的边界区域
        mask_pred_loc = pre_loc[mask].view(-1, 4)
        mask_target_loc = target_loc[mask].view(-1, 4)
        # print(mask_pred_loc.shape, mask_target_loc.shape)  # ((19L, 4L), (19L, 4L))
        pre_loc_loss = torch.nn.functional.smooth_l1_loss(mask_pred_loc.float(), mask_target_loc.to(device).float(), 
                                                          reduction='mean')
        # print pre_loc_loss  # 0.077294916
        # pre_loc_loss = torch.autograd.Variable(torch.from_numpy(pre_loc_loss))
        # 损失总和
        pred_conf_loss = np.squeeze(pred_conf_loss)
        total_loss = pred_conf_loss + (weight * pre_loc_loss)

        return total_loss
```

# utils


```python
def init_anchor(img_size=800,sub_sample=16):
    ratios = [0.5,1,2]
    anchor_scales = [8,16,32]
    feature_size = img_size//sub_sample #50
    ctr_x = np.arange(sub_sample,(feature_size+1)*sub_sample,sub_sample)
    ctr_y = np.arange(sub_sample,(feature_size+1)*sub_sample,sub_sample)
    index = 0
    ctr = dict()
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index] = [-1,-1]
            ctr[index][1] = ctr_x[x]-sub_sample/2
            ctr[index][0] = ctr_y[y]-sub_sample/2
            index+=1
            
    anchors = np.zeros(((feature_size * feature_size * 9), 4))  # (22500, 4)
    index = 0
    
    for c in ctr:
        ctr_y,ctr_x = ctr[c]
        for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
                length = anchor_scales[j]*sub_sample
                h = length*np.sqrt(ratios[i])
                w = length*np.sqrt(1/ratios[i])
                anchors[index, 0] = ctr_y - h / 2.
                anchors[index, 1] = ctr_x - w / 2.
                anchors[index, 2] = ctr_y + h / 2.
                anchors[index, 3] = ctr_x + w / 2.
                index += 1
    valid_anchor_index = np.where((anchors[:,0]>=0)&
                                 (anchors[:,1]>=0)&
                                 (anchors[:,2]<=800)&
                                 (anchors[:,3]<=800))[0]
    valid_anchor_boxes = anchors[valid_anchor_index]
    return anchors,valid_anchor_boxes,valid_anchor_index
```


```python
def compute_iou(valid_anchor_boxes,gts):
    ious = np.empty((len(valid_anchor_boxes),gts.shape[0]),dtype = np.float32)
    ious.fill(0)
    for i,box in enumerate(valid_anchor_boxes):
        ya1,xa1,ya2,xa2 = box
        anchor_area = (ya2-ya1)*(xa2-xa1)
        for j,gt in enumerate(gts):
            yb1,xb1,yb2,xb2 = gt
            box_area = (yb2-yb1)*(xb2-xb1)
            inter_y1 = max([ya1,yb1])
            inter_x1 = max([xa1,xb1])
            inter_y2 = min([ya2,yb2])
            inter_x2 = min([xa2,xb2])
            if(inter_x1<inter_x2) and (inter_y1<inter_y2):
                inter_area = (inter_x2-inter_x1)*(inter_y2-inter_y1)
                iou = inter_area/(anchor_area+box_area-inter_area)
            else:
                iou = 0
            ious[i,j]=iou
    return ious                          
```


```python
def get_pos_neg_sample(ious, valid_anchor_len, pos_iou_threshold=0.7,neg_iou_threshold=0.3, pos_ratio=0.5, n_sample=256):
    gt_argmax_ious = ious.argmax(axis=0)
    gt_max_ious = ious[gt_argmax_ious,np.arange(ious.shape[1])]
    argmax_ious = ious.argmax(axis=1)
    max_ious = ious[np.arange(ious.shape[0]),argmax_ious]
    
    gt_argmax_ious=np.where(ious==gt_max_ious)[0]
    label = np.empty((ious.shape[0],),dtype = np.int8)
    label.fill(-1)
    label[max_ious<neg_iou_threshold]=0
    label[gt_argmax_ious] = 1 
    label[max_ious >= pos_iou_threshold] = 1 
    n_pos = pos_ratio * n_sample 
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1
        
    n_neg = n_sample - np.sum(label == 1)
    neg_index = np.where(label == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
        label[disable_index] = -1
    return label, argmax_ious
```


```python
def get_coefficient(anchor, bbox):
    # 根据上面得到的预测框和与之对应的目标框，计算4维参数（平移参数：dy, dx； 缩放参数：dh, dw）
    height = anchor[:, 2] - anchor[:, 0]
    width = anchor[:, 3] - anchor[:, 1]
    ctr_y = anchor[:, 0] + 0.5 * height
    ctr_x = anchor[:, 1] + 0.5 * width
    base_height = bbox[:, 2] - bbox[:, 0]
    base_width = bbox[:, 3] - bbox[:, 1]
    base_ctr_y = bbox[:, 0] + 0.5 * base_height
    base_ctr_x = bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()
    # print(gt_roi_locs.shape)

    return gt_roi_locs
```


```python
def get_predict_bbox(anchors, pred_anchor_locs, objectness_score, n_train_pre_nms=12000, min_size=16):
    #将anchor值转换为中心点和宽高值，后续与预测的偏移系数做偏移值
    anc_height = anchors[:, 2] - anchors[:, 0]
    anc_width = anchors[:, 3] - anchors[:, 1]
    anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
    anc_ctr_x = anchors[:, 1] + 0.5 * anc_width
    pred_anchor_locs = pred_anchor_locs[0]
    objectness_score = objectness_score[0]
    dy = pred_anchor_locs[:, 0::4]
    dx = pred_anchor_locs[:, 1::4]
    dh = pred_anchor_locs[:, 2::4]
    dw = pred_anchor_locs[:, 3::4]
    anc_height = anc_height.unsqueeze(-1)
    anc_width = anc_width.unsqueeze(-1)
    anc_ctr_y = anc_ctr_y.unsqueeze(-1)
    anc_ctr_x = anc_ctr_x.unsqueeze(-1)
    ctr_y = dy * anc_height + anc_ctr_y
    ctr_x = dx * anc_width+ anc_ctr_x
    h = torch.exp(dh)*anc_height
    w = torch.exp(dw)*anc_width
    
    roi = torch.zeros(pred_anchor_locs.shape, dtype=pred_anchor_locs.dtype)
    roi[:, 0::4] = ctr_y - 0.5 * h
    roi[:, 1::4] = ctr_x - 0.5 * w
    roi[:, 2::4] = ctr_y + 0.5 * h
    roi[:, 3::4] = ctr_x + 0.5 * w
    
    img_size = (800, 800) 
    #固定到0-800之间
    roi[:, slice(0, 4, 2)] = torch.clamp(roi[:, slice(0, 4, 2)], 0, img_size[0])
    roi[:, slice(1, 4, 2)] = torch.clamp(roi[:, slice(1, 4, 2)], 0, img_size[1])
    #删除小鱼阀值的预测框结果
    hs = roi[:, 2] - roi[:, 0]
    ws = roi[:, 3] - roi[:, 1]
    keep = torch.where((hs >= min_size) & (ws >= min_size))[0]
    roi = roi[keep, :]
    score = objectness_score[keep]
    order = torch.argsort(score,descending = True)   # (22500,)
    # 取前几个预测框pre_nms_topN(如训练时12000，测试时300)
    order = order[:n_train_pre_nms]

    return roi, score, order
```


```python
def nms(roi, score, order, nms_thresh=0.7, n_train_post_nms=2000):
    roi = roi.cpu().detach().numpy()
    score = score.cpu().detach().numpy()
    order = order.cpu().detach().numpy()
    roi = roi[order, :]  # (12000, 4)
    score = score[order]
    y1 = np.ascontiguousarray(roi[:, 0])
    x1 = np.ascontiguousarray(roi[:, 1])
    y2 = np.ascontiguousarray(roi[:, 2])
    x2 = np.ascontiguousarray(roi[:, 3])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = score.argsort()[::-1]
    # print score
    # print order
    keep = []
    while order.size > 0:
        # print order
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # print ovr
        inds = np.where(ovr <= nms_thresh)[0]
        # print inds
        order = order[inds + 1]  # 这里加1是因为在计算IOU时，把序列的第一个忽略了（如上面的order[1:]）

    keep = keep[:n_train_post_nms]  # while training/testing , use accordingly
    roi = roi[keep]  # the final region proposals（region proposals表示预测目标框）
    # print roi.shape  # (1758, 4)
    return roi
```


```python
def get_propose_target(roi, bbox, labels, n_sample=128, pos_ratio=0.25,
                       pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo = 0.0):
    # Proposal targets
    # 找到每个ground-truth目标（真实目标框bbox）与region proposal（预测目标框roi）的iou
    ious = compute_iou(roi, bbox)
    # print(ious.shape)  # (1758, 2)

    # 找到与每个region proposal具有较高IoU的ground truth，并且找到最大的IoU
    gt_assignment = ious.argmax(axis=1)
    max_iou = ious.max(axis=1)
    # print(gt_assignment)  # [0 0 1 ... 0 0 0]
    # print(max_iou)  # [0.17802152 0.17926688 0.04676317 ... 0.         0.         0.        ]

    # 为每个proposal分配标签：
    gt_roi_label = labels[gt_assignment]
    # print(gt_roi_label)  # [6 6 8 ... 6 6 6]

    # 希望只保留n_sample*pos_ratio（128*0.25=32）个前景样本，因此如果只得到少于32个正样本，保持原状。
    # 如果得到多余32个前景目标，从中采样32个样本
    pos_roi_per_image = n_sample*pos_ratio
    pos_index = np.where(max_iou >= pos_iou_thresh)[0]
    pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
    # print(pos_roi_per_this_image)
    # print(pos_index)  # 19

    # 针对负[背景]region proposal进行相似处理
    neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
    neg_roi_per_this_image = n_sample - pos_roi_per_this_image
    neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
    # print(neg_roi_per_this_image)
    # print(neg_index)  # 109

    keep_index = np.append(pos_index, neg_index)
    gt_roi_labels = gt_roi_label[keep_index]
    gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0
    sample_roi = roi[keep_index]  # 预测框
    # print(sample_roi.shape)  # (128, 4)
    return sample_roi, keep_index, gt_assignment, gt_roi_labels
```

# train


```python
ground_truth = np.asarray([[20,30,400,500],[300,400,500,600]],dtype=np.float32)
true_label = np.asarray([6,8],dtype=np.int8)
```


```python
img_tensor = torch.zeros((1,3,800,800)).float()
img_var = torch.autograd.Variable(img_tensor)
```


```python
anchors, valid_anchor_boxes, valid_anchor_index = init_anchor()
print(anchors.shape,valid_anchor_boxes.shape,valid_anchor_index.shape)
```

    (22500, 4) (8940, 4) (8940,)



```python
ious = compute_iou(valid_anchor_boxes, ground_truth)
print(ious.shape)
```

    (8940, 2)



```python
valid_anchor_len = len(valid_anchor_boxes)
print(valid_anchor_len)
```

    8940



```python
label, argmax_ious = get_pos_neg_sample(ious, valid_anchor_len, pos_iou_threshold=0.7,neg_iou_threshold=0.3, pos_ratio=0.5, n_sample=256)
```


```python
max_iou_ground_truth = ground_truth[argmax_ious]
```


```python
anchor_locs = get_coefficient(valid_anchor_boxes, max_iou_ground_truth)#合法的anchior与之对应的groundtruth直接调整系数值
```


```python
anchor_conf = np.empty((len(anchors),), dtype=label.dtype)
anchor_conf.fill(-1)
anchor_conf[valid_anchor_index] = label
print (anchor_conf.shape)
```

    (22500,)



```python
anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[valid_anchor_index, :] = anchor_locs
print(anchor_locations.shape)
```

    (22500, 4)


# 至此已经得到所有的anchor对应的训练数据，anchor标签和系数


```python
vgg = VGG()
cudnn.benchmark = True
device = 'cuda'
vgg.to(device)
```




    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (9): ReLU(inplace=True)
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (12): ReLU(inplace=True)
        (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (16): ReLU(inplace=True)
        (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (19): ReLU(inplace=True)
        (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (26): ReLU(inplace=True)
        (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (29): ReLU(inplace=True)
        (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (32): ReLU(inplace=True)
        (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (36): ReLU(inplace=True)
        (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (39): ReLU(inplace=True)
        (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (42): ReLU(inplace=True)
      )
      (rpn_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (reg_layer): Conv2d(512, 36, kernel_size=(1, 1), stride=(1, 1))
      (cls_layer): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
      (adaptive_max_pool): AdaptiveMaxPool2d(output_size=7)
      (roi_head_classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU()
        (2): Linear(in_features=4096, out_features=4096, bias=True)
        (3): ReLU()
      )
      (cls_loc): Linear(in_features=4096, out_features=84, bias=True)
      (score): Linear(in_features=4096, out_features=21, bias=True)
    )




```python
img_var =img_var.to(device)
out_map, pred_anchor_locs, pred_anchor_conf = vgg.forward(img_var)#pred_anchor_conf回归层
print(pred_anchor_locs.data.shape)
```

    torch.Size([1, 36, 50, 50])



```python
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
print(pred_anchor_locs.shape)
```

    torch.Size([1, 22500, 4])



```python
pred_anchor_conf = pred_anchor_conf.permute(0, 2, 3, 1).contiguous()
print(pred_anchor_conf.shape)  
objectness_score = pred_anchor_conf.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
print(objectness_score.shape) 
pred_anchor_conf = pred_anchor_conf.view(1, -1, 2)
print(pred_anchor_conf.shape) 
```

    torch.Size([1, 50, 50, 18])
    torch.Size([1, 22500])
    torch.Size([1, 22500, 2])


# 计算RPN损失


```python
rpn_anchor_loc = pred_anchor_locs[0]
rpn_anchor_conf = pred_anchor_conf[0]
print(rpn_anchor_loc.shape)
print(rpn_anchor_conf.shape)
```

    torch.Size([22500, 4])
    torch.Size([22500, 2])



```python
anchor_locations = torch.from_numpy(anchor_locations)
anchor_conf = torch.from_numpy(anchor_conf)
```


```python
anchor_locations  =anchor_locations.to(device)
anchor_conf = anchor_conf.to(device)
rpn_loss = vgg.rpn_loss(rpn_anchor_loc, rpn_anchor_conf, anchor_locations, anchor_conf, weight=10.0)
print("rpn_loss: {}".format(rpn_loss))  # 1.33919
```

    torch.Size([18, 4]) torch.Size([18, 4])
    rpn_loss: 0.8424026966094971



```python
roi, score, order =get_predict_bbox(torch.tensor(anchors).to(device).float(), pred_anchor_locs, objectness_score,
                                           n_train_pre_nms=12000, min_size=16)
```


```python
roi = nms(roi, score, order, nms_thresh=0.7, n_train_post_nms=2000)
```


```python
print(roi.shape)
```

    (2000, 4)



```python
sample_roi, keep_index, gt_assignment, roi_labels = get_propose_target(roi, ground_truth, true_label,
                                                                                n_sample=128,
                                                                                pos_ratio=0.25,
                                                                                pos_iou_thresh=0.5,
                                                                                neg_iou_thresh_hi=0.5,
                                                                                neg_iou_thresh_lo=0.0)
```


```python
print(sample_roi.shape,keep_index.shape,gt_assignment.shape,roi_labels.shape)
```

    (128, 4) (128,) (2000,) (128,)



```python
bbox_for_sampled_roi = ground_truth[gt_assignment[keep_index]]  # 目标框
print(bbox_for_sampled_roi.shape) 
```

    (128, 4)



```python
roi_locs = get_coefficient(sample_roi, bbox_for_sampled_roi)
```

# ROIpooling


```python
rois = torch.from_numpy(sample_roi).float()
```


```python
roi_indices = np.zeros((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()
```


```python
indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1) 
```


```python
print(indices_and_rois.shape)
```

    torch.Size([128, 5])



```python
output = []
rois = indices_and_rois.float()
rois[:, 1:].mul_(1/16.0)
rois = rois.long()
num_rois = rois.size(0)
for i in range(num_rois):
    roi =rois[i]
    im_idx = roi[0]
    out_map = out_map.narrow(0, im_idx, 1)#相当于out_map(0,:,:,:,:)
    im = out_map[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
    output.append(vgg.adaptive_max_pool(im)[0].data)
```


```python
output = torch.cat(output,0)
print(output.shape)
```

    torch.Size([128, 512, 7, 7])



```python
k = output.view(output.size(0), -1)
k = torch.autograd.Variable(k)
k = vgg.roi_head_classifier(k) 
# torch.Size([128, 84])  84 ==> (20+1)*4,表示每个框有20个候选类别和一个置信度（假设为VOC数据集，共20分类），4表示坐标信息
pred_roi_locs = vgg.cls_loc(k)
# pred_roi_labels： [128, 21] 表示每个框的类别和置信度
pred_roi_labels = vgg.score(k)
print(pred_roi_locs.data.shape, pred_roi_labels.data.shape) 
```

    torch.Size([128, 84]) torch.Size([128, 21])


# 分类损失


```python
# 预测框的坐标系数(roi_locs)：  (128, 4)
# 预测框的所属类别(roi_labels)：(128, )

# 从上面step_6中，我们得到了预测框转为目标框的预测信息：
# 预测框的坐标系数：pred_roi_locs  (128, 84)
# 预测框的所属类别和置信度: pred_roi_labels  (128, 21)
```


```python
gt_roi_loc = torch.from_numpy(roi_locs)
gt_roi_label = torch.from_numpy(np.float32(roi_labels)).long()
```


```python
n_sample = pred_roi_locs.shape[0]
roi_loc = pred_roi_locs.view(n_sample, -1, 4)
print(roi_loc.shape)
roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label] 
```

    torch.Size([128, 21, 4])



```python
roi_loss = vgg.roi_loss(roi_loc, pred_roi_labels, gt_roi_loc, gt_roi_label, weight=10.0)
print(roi_loss)
```

    tensor(3.2642, device='cuda:0', grad_fn=<AddBackward0>)



```python
total_loss = rpn_loss + roi_loss
print(total_loss)
```

    tensor(4.1066, device='cuda:0', grad_fn=<AddBackward0>)



```python

```
