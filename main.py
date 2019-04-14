#_*_coding:utf-8_*_

from config import opt
import models
import fire
from utils.dataset import ZhihuData
from torch.utils import data
import tqdm
from utils import get_score

def main(**kwargs):
    model = getattr(models,opt.model)(opt).cuda()
    pre_loss = 1.0
    lr,lr2 = opt.lr,opt.lr2
    loss_function = getattr(models,opt.loss)()
    best_score = 0
    dataset = ZhihuData(opt.train_data_path,opt.labels_path,type_=opt.type_,augument=opt.augument)
    dataloader = data.DataLoader(dataset=dataset,batch_size=opt.batch_size,shuffle=opt.shuffle,num_workers=opt.num_workers,
                                 pin_memory=True)
    optimizer = model.get_optimizer(lr,opt.lr2,opt.weight_decay)

    for epoch in range(opt.max_epoch):
        for i,((title,content),label) in tqdm.tqdm(enumerate(dataloader)):
            title,content,label = title.cuda(),content.cuda(),label.cuda()
            optimizer.zero_grad()
            score = model(title,content)
            loss = loss_function(score,opt.weight * label.float())
            loss.backward()
            optimizer.step()

            predict = score.data.topk(5,dim=1)[1].cpu().tolist()
            true_target = label.data.float().cpu().topk(5,dim=1)
            true_label = true_target[0][:,:5]
            true_index = true_target[1][:,:5]
            predict_label_and_marked_label_list = []
            for j in range(label.size(0)):
                true_index_ = true_index[j]
                true_label_ = true_label[j]
                true = true_index_[true_label_ > 0]
                predict_label_and_marked_label_list.append(predict[j],true.tolist())
        score_,prec_,recall_,ss= get_score(predict_label_and_marked_label_list)

    scores, prec_, recall_, _ss = val(model, dataset)
    if scores > best_score:
        best_score = scores
        best_path = model.save(name=str(scores), new=True)

    if scores < best_score:
        model.load(best_path, change_opt=False)
        lr = lr * opt.lr_decay
        lr2 = 2e-4 if lr2 == 0 else lr2 * 0.8
        optimizer = model.get_optimizer(lr, lr2, 0)

def val(model, dataset):
    '''
    计算模型在验证集上的分数
    '''

    dataset.train(False)
    model.eval()

    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers,
                                 pin_memory=True
                                 )

    predict_label_and_marked_label_list = []
    for ii, ((title, content), label) in tqdm.tqdm(enumerate(dataloader)):
        title, content, label = title.cuda(),content.cuda(), label.cuda()
        score = model(title, content)
        # !TODO: 优化此处代码
        #       1. append
        #       2. for循环
        #       3. topk 代替sort

        predict = score.data.topk(5, dim=1)[1].cpu().tolist()
        true_target = label.data.float().topk(5, dim=1)
        true_index = true_target[1][:, :5]
        true_label = true_target[0][:, :5]
        tmp = []

        for jj in range(label.size(0)):
            true_index_ = true_index[jj]
            true_label_ = true_label[jj]
            true = true_index_[true_label_ > 0]
            tmp.append((predict[jj], true.tolist()))

        predict_label_and_marked_label_list.extend(tmp)
    del score

    dataset.train(True)
    model.train()

    scores, prec_, recall_, _ss = get_score(predict_label_and_marked_label_list)
    return (scores, prec_, recall_, _ss)

if __name__ == '__main__':
    fire.Fire()