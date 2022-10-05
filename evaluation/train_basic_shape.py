import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../3rd_party/pointnet_pointnet2_pytorch/'))

import numpy as np
import torch

from BasicShapeDataloaderInterface import PointcloudDataset
from pointnet2_sem_seg import PointNet2, PointNet2Loss


# # see https://github.com/pyg-team/pytorch_geometric/issues/366
# # PLATFORM = 'windows'
# PLATFORM = 'ubuntu'  # {'windows', 'ubuntu'}
# NUM_WORKERS = 0 if PLATFORM == 'windows' else 10
NUM_WORKERS = 10


NUM_CLASSES = 3
NUM_EPOCH = 128
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DECAY_RATE = 1e-4
LEARNING_RATE_CLIP = 1e-5
LEARNING_RATE_DECAY = 0.7
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = 10

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    train_root_path = cur_path + '/../data/basic_shapes/dataset/train/'
    train_set = PointcloudDataset(train_root_path, NUM_CLASSES, 'train')
    test_root_path = cur_path + '/../data/basic_shapes/dataset/test/'
    test_set = PointcloudDataset(test_root_path, NUM_CLASSES, 'test')

    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True, 
        drop_last=True
    )

    label_weights = torch.Tensor(train_set.labelweights).to(DEVICE)


    classifier = PointNet2(NUM_CLASSES).to(DEVICE)
    classifier.apply(inplace_relu)
    # classifier = classifier.apply(weights_init)
    criterion = PointNet2Loss().to(DEVICE)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=DECAY_RATE
    )

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in range(NUM_EPOCH):
        lr = max(LEARNING_RATE * (LEARNING_RATE_DECAY ** (epoch // MOMENTUM_DECCAY_STEP)), LEARNING_RATE_CLIP)

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        num_batches = len(train_data_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()
        
        for i, (points, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()

            points = points.float().to(DEVICE)
            labels = labels.long().to(DEVICE)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = labels.view(-1, 1)[:, 0].cpu().data.numpy()
            target = labels.view(-1, 1)[:, 0]
            # print(seg_pred.size())
            # print(target.size())
            # print(label_weights)
            loss = criterion(seg_pred, target, trans_feat, label_weights)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * train_set.num_points)
            loss_sum += loss

            train_loss_list.append(loss_sum / num_batches)
            train_acc_list.append(total_correct / float(total_seen))

            print('Training mean loss: %f' % (loss_sum / num_batches))
            print('Training accuracy: %f' % (total_correct / float(total_seen)))

        with torch.no_grad():
            num_batches = len(test_data_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            for i, (points, labels) in enumerate(test_data_loader):
                points = points.float().to(DEVICE)
                labels = labels.long().to(DEVICE)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = labels.cpu().data.numpy()
                labels = labels.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, labels, trans_feat, label_weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * train_set.num_points)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))

            test_loss_list.append(loss_sum / float(num_batches))
            test_acc_list.append(total_correct / float(total_seen))

            print('eval mean loss: %f' % (loss_sum / float(num_batches)))
            print('eval point avg class IoU: %f' % (mIoU))
            print('eval point accuracy: %f' % (total_correct / float(total_seen)))
            print('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

    


if __name__ == '__main__':
    main()