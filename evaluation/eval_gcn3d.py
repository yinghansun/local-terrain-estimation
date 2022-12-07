import datetime
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../3rd_party/gcn3d/'))

import numpy as np
import torch

from BasicShapeDataloaderInterface import PointcloudDataset
from gcn3d_utils import *
from hyper_paras import *
from model_gcn3d import GCN3D


def record(info):
    print(info)
    # if record_file:
    #     record_file.write(info + '\n')

class IouTable():
    def __init__(self):
        self.obj_miou = {}
        
    def add_obj_miou(self, category: str, miou: float):
        if category not in self.obj_miou:
            self.obj_miou[category] = [miou]
        else:
            self.obj_miou[category].append(miou)

    def get_category_miou(self):
        """
        Return: moiu table of each category
        """
        category_miou = {}
        for c, mious in self.obj_miou.items():
            category_miou[c] = np.mean(mious)
        return category_miou

    def get_mean_category_miou(self):
        category_miou = []
        for c, mious in self.obj_miou.items():
            c_miou = np.mean(mious)
            category_miou.append(c_miou)
        return np.mean(category_miou)
    
    def get_mean_instance_miou(self):
        object_miou = []
        for c, mious in self.obj_miou.items():
            object_miou += mious
        return np.mean(object_miou)

    def get_string(self):
        mean_c_miou = self.get_mean_category_miou()
        mean_i_miou = self.get_mean_instance_miou()
        first_row  = "| {:5} | {:5} ||".format("Avg_c", "Avg_i")
        second_row = "| {:.3f} | {:.3f} ||".format(mean_c_miou, mean_i_miou)
        
        categories = list(self.obj_miou.keys())
        categories.sort()
        cate_miou = self.get_category_miou()

        for c in categories:
            miou = cate_miou[c]
            first_row  += " {:5} |".format(c[:3])
            second_row += " {:.3f} |".format(miou)
        
        string = first_row + "\n" + second_row
        return string 


cur_path = os.path.dirname(os.path.abspath(__file__))
result_root_path = cur_path + '/results/'
date = datetime.date.today()
model = 'gcn3d'
num_experiments = '1'
folder_name = str(date) + '-' + model + '-' + num_experiments
save_path = result_root_path + folder_name

if not os.path.exists(result_root_path):
    os.mkdir(result_root_path)
if not os.path.exists(save_path):
    os.mkdir(save_path)

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

    model = GCN3D(class_num=NUM_CLASSES, support_num=1, neighbor_num=50).to(DEVICE)
    # classifier.apply(inplace_relu)
    # classifier = classifier.apply(weights_init)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        # betas=(0.9, 0.999),
        # eps=1e-08,
        # weight_decay=DECAY_RATE
    )
    loss_function = torch.nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    eval_idx = 0

    record("*****************************************")
    # record("Hyper-parameters: {}".format(self.args_info))
    record("Model parameter number: {}".format(parameter_number(model)))
    record("Model structure: \n{}".format(model.__str__()))
    record("*****************************************")

    for epoch in range(NUM_EPOCH):
        record('epoch {} / {}:'.format(epoch+1, NUM_EPOCH))
        model.train()
        train_loss = 0
        train_iou_table = IouTable()

        num_batches = len(train_data_loader)
        total_correct = 0
        total_seen = 0

        for i, (points, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()

            points = points.float().to(DEVICE)
            labels = torch.squeeze(labels.long(), dim=1).to(DEVICE)
            # print(points.shape)
            # print(labels.shape)
            out = model(points)

            optimizer.zero_grad()
            loss = loss_function(out.reshape(-1, out.size(-1)), labels.view(-1,))
            loss.backward()
            optimizer.step()

            out = out.contiguous().view(-1, NUM_CLASSES)

            train_loss += loss.item()
            record('training loss: {}'.format(loss.item()))
            # pred = torch.max(out, 2)[1].contiguous().view(-1, NUM_CLASSES)
            # print(out.shape)
            pred_choice = out.cpu().data.max(1)[1].numpy()

            batch_label = labels.view(-1, 1)[:, 0].cpu().data.numpy()
            # pred_choice = pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * train_set.num_points)

            record('Training accuracy: %f' % (total_correct / float(total_seen)))

        with torch.no_grad():
            num_batches = len(test_data_loader)
            total_correct = 0
            total_seen = 0
            model = model.eval()

            for i, (points, labels) in enumerate(train_data_loader):
                points = points.float().to(DEVICE)
                labels = torch.squeeze(labels.long(), dim=1).to(DEVICE)
                out = model(points)

                out = out.contiguous().view(-1, NUM_CLASSES)
                pred_choice = out.cpu().data.max(1)[1].numpy()
                batch_label = labels.view(-1, 1)[:, 0].cpu().data.numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE * train_set.num_points)

            record('test accuracy: %f' % (total_correct / float(total_seen)))

            torch.save(model.state_dict(), save_path + '/para_dic' + str(eval_idx) + '.pth')
            

if __name__ == '__main__':
    main()