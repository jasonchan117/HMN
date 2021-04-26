import torch
from torchvision import transforms
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import numpy as np
from utils.datasets import *
import Make_Law_Label
from sklearn.metrics import confusion_matrix
import datetime
from tqdm import tqdm
def cal_precision_recall(parent,y_true, y_pred):
    macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
    macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
    print("===== parent is {}, ma precision is {}, ma recall is {}".format(parent, macro_precision, macro_recall))

def cal_metric(y_true, y_pred):
    ma_p, ma_r, ma_f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
    # mi_p, mi_r, mi_f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')
    acc = metrics.accuracy_score(y_true,y_pred)
    jaccard = metrics.jaccard_similarity_score(y_true, y_pred)
    hamming_loss = metrics.hamming_loss(y_true, y_pred)
    # average_f1 = (ma_f1 + mi_f1)/2 * 100
    return [(ma_p, ma_r, ma_f1), acc, jaccard, hamming_loss]

def cal_metrics(y_batch, y_predictions, loss):

    f1_score_macro = metrics.f1_score(np.array(y_batch), y_predictions, average='macro')
    macro_precision = metrics.precision_score(np.array(y_batch), y_predictions, average='macro')
    macro_recall = metrics.recall_score(np.array(y_batch), y_predictions, average='macro')
    # metrics.auc()
    f1_score_micro = metrics.f1_score(np.array(y_batch), y_predictions, average='micro')
    micro_precision = metrics.precision_score(np.array(y_batch), y_predictions, average='micro')
    micro_recall = metrics.recall_score(np.array(y_batch), y_predictions, average='micro')
    average_f1 = (f1_score_macro + f1_score_micro)/2 * 100

    time_str = datetime.datetime.now().isoformat()
    print("the time is : {}. the loss is: {}. the average f1 score is : {}".format(time_str, loss.data[0], average_f1))
    print("macro precision is: {}. macro recall is: {}.micro precision is: {}. micro recall is: {}.".format(macro_precision, macro_recall, micro_precision, micro_recall))
    return average_f1

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        a = param_group['lr']
    print('lr:', a)
def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    if args.sep_nln == True and args.nln == True:
        optimizer_sep = torch.optim.Adam([{'params':[ param for name, param in model.named_parameters() if 'NLN_child' in name or 'NLN_parent' in name or 'trans' in name or 'p_trans' in name]}], lr = args.sep_lr)
        optimizer = torch.optim.Adam([{'params':[ param for name, param in model.named_parameters() if 'NLN_child' not in name or 'NLN_parent' not in name or 'trans' not in name or 'p_trans' not in name]}], lr = args.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    parent_size = [[0, 17], [17, 71], [71, 91], [91, 104], [104, 159], [159, 162], [162, 176], [176, 183]]
    law_text, law_length, law_order, parent2law = Make_Law_Label.makelaw()
    steps = 0
    best_f1_score = 0.0
    sum_loss = 0
    model.train()
    best_f1 = -1
    for epoch in range(1, args.epoch + 1):
        start_test_time = datetime.datetime.now()
        print("==================== epoch:{} ====================".format(epoch))
        for batch in tqdm(train_iter):
            # Label1: parent label(One hot), Label2: child label(One hot), law: origin child label index.
            text, text_lens, label1, label2, law, law_num, parent_num = batch

            text, label1, label2= Variable(text),Variable(label1), Variable(label2)
            law_num, parent_num = Variable(law_num), Variable(parent_num)
            article_text, article_len = Variable(law_text), Variable(law_length)
            if args.cuda:
                text, label1 = text.cuda(), label1.cuda()
                label2 = label2.cuda()
                text_lens = text_lens.cuda()
                law_num, parent_num = law_num.cuda(), parent_num.cuda()
                # law_text = law_text.cuda()
                article_text, article_len = article_text.cuda(), article_len.cuda()

            # we have parent classifier and sub classifier,separate input by parent class and train
            # Generate non-zero index
            parent_index = torch.nonzero(label1) # tensor([ 0, -1,  1,  1, -1,  0,  1, -1, -1, -1]) -> tensor([[1],[2], [3],[4],[6],[7],[8], [9]])
            classify = [[] for i in range(8)] # [[], [], [], [], [], [], [], []]
            label2_list = []
            # 2-d list classify recode the case index in one batch that violate corresponding parent label.
            for index in parent_index:
                classify[index[1]] = classify[index[1]] + [index[0]]

            # classify[5]= [i for i in range(len(text))]
            classify = [torch.LongTensor(item) for item in classify]
            for i, item in enumerate(classify):
                if(len(item)==1):
                    classify[i] =classify[i].repeat(2)
                    item = item.repeat(2)
                label2_part = label2[item]  # label2 size:64x183, select those rows
                if len(label2_part) > 0:
                    label2_part = label2_part[:, parent_size[i][0]: parent_size[i][1]]
                label2_list.append(label2_part)# label2_list is 8xnxm matrix(8: parent classes, n: how many cases, m:184), each row contains the one-hot of label2

            optimizer.zero_grad()

            # label_des, all_list= model(label_inputs=article_text, label_inputs_length=article_len,epoch=epoch,step=steps)
            # The article_text and article_len here are the child label text and len after encoding. Using Dynamic GRU, and the return value label des and all_list are (183,128) and (8 x m x 128)(allocate to parent label)
            label_des, all_list = model(label_inputs=article_text, label_inputs_length=article_len)
            # (183,128), (8, m, 128)
            if args.nln == True:
                logits, logits_list, logits_child_num, logits_parent_num = model(inputs=text, inputs_length=text_lens, label_des=label_des,
                               all_list=all_list, classify=classify, flag=0)
            else:
                logits, logits_list = model(inputs=text, inputs_length=text_lens, label_des=label_des,
                               all_list=all_list, classify=classify, flag=0)
            # logits :: label1
            # logits_list :: label2

            # print(steps)
            # loss1 is the loss of parent label
            loss1 = torch.nn.functional.binary_cross_entropy_with_logits(logits, label1)
            loss2 = 0
            # If the the number of cases in each parent label is greater than 0, then compute the child label loss
            if len(label2_list[0]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[0], label2_list[0])
            if len(label2_list[1]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[1], label2_list[1])
            if len(label2_list[2]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[2], label2_list[2])
            if len(label2_list[3]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[3], label2_list[3])
            if len(label2_list[4]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[4], label2_list[4])
            if len(label2_list[5]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[5], label2_list[5])
            if len(label2_list[6]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[6], label2_list[6])
            if len(label2_list[7]) > 0:
                loss2 += torch.nn.functional.binary_cross_entropy_with_logits(logits_list[7], label2_list[7])

            # c_sample = [110905, 28075, 9631, 4028, 1423, 394, 96, 29, 8, 1, 1, 1]
            # p_sample = [138516, 15641, 413, 21, 1]
            # p_labels = []
            # c_labels = []
            # for i in range(args.p_num):
            #     p_labels += [i] * p_sample[i]
            # for i in range(args.c_num):
            #     c_labels += [i] * c_sample[i]
            # #Compute weights
            # p_class_wts = compute_class_weight('balanced', range(args.p_num), p_labels)
            # c_class_wts = compute_class_weight('balanced', range(args.c_num), c_labels)

            # Loss of label prediction
            # Use NLN , train separately
            if args.nln == True and args.sep_nln == True:
                loss_child_num = torch.nn.functional.binary_cross_entropy_with_logits(logits_child_num, law_num)
                loss_parent_num = torch.nn.functional.binary_cross_entropy_with_logits(logits_parent_num, parent_num)
                loss_NLN = loss_child_num + loss_parent_num
                loss_NLN.backward()
                optimizer_sep.step()
                loss = loss_NLN
            # Use NLN, train together
            elif args.nln == True:
                loss_child_num = torch.nn.functional.binary_cross_entropy_with_logits(logits_child_num, law_num)
                loss_parent_num = torch.nn.functional.binary_cross_entropy_with_logits(logits_parent_num, parent_num)
                loss = loss1 + loss2 + loss_child_num + loss_parent_num
                loss.backward()
                optimizer.step()
            # Do not use NLN.
            else:
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()


            sum_loss = sum_loss + loss.data
            steps = steps + 1
            if steps % args.print_freq == 0:
                print("##################### step is : {} ########################".format(steps))
                for i in range(8):
                    if len(logits_list[i])>0:
                        logits_numpy = (F.sigmoid(logits_list[i]).cpu().data.numpy() > 0.5).astype('int')
                        label_numpy = label2_list[i].cpu().data.numpy()
                        cal_precision_recall(i+1,label_numpy, logits_numpy)

            # if steps % args.test_freq == 0:
            #     eval(dev_iter, model, args, label_des,all_list)

            #     if valid_average_f1 > best_f1_score:
            #         best_f1_score = valid_average_f1
            #         last_step = steps
            #         if args.save_best:
            #             save(model, args.save_dir, args.save_dir.split("/")[0] + "_best", steps)
            # if steps % args.save_interval == 0:
            #     save(model, args.save_dir, args.save_dir.split("/")[0], steps)
        end_test_time = datetime.datetime.now()
        print("Train : epoch {}, time cost {}".format(epoch , end_test_time - start_test_time))
        print("Train : sum loss {}, average loss {}".format(sum_loss, sum_loss / (steps)))
        sum_loss = 0
        steps = 0
        if epoch % int(args.valid_fre) == 0:
            f1 = eval(dev_iter, model, args,label_des, all_list )
            if f1 > best_f1 :
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(args.ckpt, ''.join([args.id, '_', str(epoch),'_', str(f1), '.pt'])))
        if (epoch) % 5 == 0:
            if args.sep_nln == True:
                adjust_learning_rate(optimizer_sep)
            else:
                adjust_learning_rate(optimizer)
            print("lr dec 5")

def eval(dev_iter, model, args,label_des,all_list):
    model.eval()
    avg_loss = 0.0
    avg_f1 = 0.0
    batch_num = 0
    pre_label1_list = []
    label1_list = []
    pre_label2_list = []
    label2_list = []
    c_g = []
    c_p = []
    p_g = []
    p_p = []
    start_test_time = datetime.datetime.now()
    print("======================== Evaluation =====================")
    for batch in tqdm(dev_iter):
        batch_num = batch_num + 1

        text, text_lens, label1, label2, law, law_num, parent_num = batch
        text, label2 = Variable(text), Variable(label2)

        if args.cuda:
            text, label2 = text.cuda(), label2.cuda()
            label1 = label1.cuda()
            text_lens = text_lens.cuda()

        logits, logits2, child_num, par_num = model(inputs=text, inputs_length=text_lens, label_des=label_des,all_list=all_list,flag=1,label1=label1)
        c_g.append(law_num.max(1)[1] + 1)
        p_g.append(parent_num.max(1)[1] + 1)
        if args.nln == True:
            c_p.append(child_num.cpu())
            p_p.append(par_num.cpu())
        else:
            c_p.append(child_num)
            p_p.append(par_num)

        pre_numpy1 = logits.cpu().data.numpy().astype('int')
        label1_numpy = label1.cpu().data.numpy()
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(logits2, label2)
        logits_numpy = logits2.cpu().data.numpy().astype('int')
        label_numpy = label2.cpu().data.numpy()

        pre_label1_list.append(pre_numpy1)
        label1_list.append(label1_numpy)

        pre_label2_list.append(logits_numpy)
        label2_list.append(label_numpy)

    c_g = np.array(c_g).astype('int')
    p_g = np.array(p_g).astype('int')
    c_p = np.array(c_p).astype('int')
    p_p = np.array(p_p).astype('int')
    pre_label1_list = np.concatenate(pre_label1_list)
    pre_label2_list = np.concatenate(pre_label2_list)
    label1_list = np.concatenate(label1_list)
    label2_list = np.concatenate(label2_list)

    pre_sumlist = np.concatenate((pre_label1_list,pre_label2_list),1)
    label_sumlist = np.concatenate((label1_list,label2_list),1)

    (c_p, c_r, c_f1), c_acc, c_jaccard, c_hamming_loss = cal_metric(c_g, c_p)
    (p_p, p_r, p_f1), p_acc, p_jaccard, p_hamming_loss = cal_metric(p_g, p_p)
    print("Parent label number : macro precision: {} macro recall: {}  ma f1 {}".format(p_p, p_r, p_f1))
    print("Parent label number : Acc is {}".format(p_acc))
    print("Parent label number : hamming is {}".format(p_hamming_loss))
    print("Parent label number : jaccard is {} ".format(p_jaccard))

    print("Child label number : macro precision: {} macro recall: {}  ma f1 {}".format(c_p, c_r, c_f1))
    print("Child label number : Acc is {}".format(c_acc))
    print("Child label number : hamming is {}".format(c_hamming_loss))
    print("Child label number : jaccard is {} ".format(c_jaccard))
    print('----------------------')
    parent_size = [[0, 17], [17, 71], [71, 91], [91, 104], [104, 159], [159, 162], [162, 176], [176, 183]]
    for j,item in enumerate(parent_size):
        cal_precision_recall(j+1,label2_list[:,item[0]:item[1]], pre_label2_list[:,item[0]:item[1]])
    print('----------------------')
    (pma_p, pma_r, pma_f1), pacc, pjaccard, phamming_loss = cal_metric(label1_list, pre_label1_list)
    (ma_p, ma_r, ma_f1), acc, jaccard, hamming_loss = cal_metric(label2_list,pre_label2_list)

    (sma_p, sma_r, sma_f1), sacc, sjaccard, shamming_loss = cal_metric(label_sumlist, pre_sumlist)
    model.train()

    end_test_time = datetime.datetime.now()
    print("TestP: time cost {}".format(end_test_time - start_test_time))
    print("TestP: macro precision: {} macro recall: {}  ma f1 {}".format(pma_p, pma_r, pma_f1))
    print("TestP: Acc is {}".format(pacc))
    print("TestP: hamming is {}".format(phamming_loss))
    print("TestP: jaccard is {} ".format(pjaccard))

    # print("Test : time cost {}".format(end_test_time - start_test_time))
    print("TestC : macro precision: {} macro recall: {}  ma f1 {}".format(ma_p, ma_r, ma_f1))
    print("TestC : Acc is {}".format(acc))
    print("TestC : hamming is {}".format(hamming_loss))
    print("TestC : jaccard is {} ".format(jaccard))

    print("TestS : macro precision: {} macro recall: {}  ma f1 {}".format(sma_p, sma_r, sma_f1))
    print("TestS : Acc is {}".format(sacc))
    print("TestS : hamming is {}".format(shamming_loss))
    print("TestS : jaccard is {} ".format(sjaccard))
    model.train()
    # print("average loss is {}, average f1 is {}".format(avg_loss, avg_f1))
    return sma_f1
def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    print(1)