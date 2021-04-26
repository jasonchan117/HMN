import argparse

import datetime
from utils.datasets import *
import torch
from model.model import HMN
import train

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('--dev_batch_size', default=1, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=3000, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--hidden-size', default=128, type=int)
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--learning-rate', default=0.0015, type=float)
    parser.add_argument('--print-freq', default=1500, type=int)
    parser.add_argument('--test-freq', default=1, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--output', default='./output')

    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument('-snapshot', type=str, default=None,
                        help='filename of model snapshot [default: None]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('-embed_num', type=int, default=100000, help='the num of vocabulary size')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

    parser.add_argument('--train-data-path', type=str, default=None, help='the train data directory')
    parser.add_argument('--test-data-path', type=str, default=None, help='the test data directory')

    parser.add_argument('--nln', action='store_true', help='Use NLN module or not. And train them together.')
    parser.add_argument('--num_workers',  default=400, help='The number of workers of dataloader')
    parser.add_argument('--max_len',  default=10, help='The max len of the text.')
    parser.add_argument('--id', required=True, help='The id for each training.')
    parser.add_argument('--ckpt', type=str, default='/content/drive/MyDrive/Data/HMN/NLN_20_128', help='The path to save the model.')
    parser.add_argument('--valid_fre', help='Validation frequency.',default=1)
    parser.add_argument('--p_num', type=int, default=8, help='The max number of parent label.')
    parser.add_argument('--c_num', type=int, default=12, help='The max number of children label.')

    parser.add_argument('--load_gen', type=str, default=None, help='For separate training, load general model.')
    parser.add_argument('--sep_nln', action = 'store_true', help='Train the two modules separately. This param should appear with --nln.')
    parser.add_argument('--sep_lr', default= 0.001, type=float, help = 'The learning rate for separated training.')
    parser.add_argument('--encoder', type=str, default='gru', help='The choice of encoder, choose from lstm, gru and rnn')

    args = parser.parse_args()

    time_str = datetime.datetime.now().isoformat()
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.output, ''.join([time_str, '.txt'])))


    print("\nLoading data...")
    law_path = "./data/law_dict.pkl"
    word_path = "./data/word_dict_10w.pkl"
    parent_path = "./data/parent_dict.pkl"
    train_data_path = args.train_data_path
    dev_data_path = args.test_data_path

    train_iter, dev_iter, word_num, law_num, parent_num = make_data(train_data_path, dev_data_path,
                                                          law_path, parent_path, word_path, args.batch_size, args.dev_batch_size, args.num_workers, args.max_len, args = args)
    args.model_name = 'HMN'
    args.save_dir = "accu_snapshot"
    args.embed_num = word_num
    args.class_num = law_num
    args.parent_num = parent_num
    args.law_num = law_num
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, args.model_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


    model = HMN(args)
    if args.load_gen is not None and args.nln == True:
        param_list = ['decription.embed.weight', 'decription.lstmLabel.weight_ih_l0',
                      'decription.lstmLabel.weight_hh_l0', 'decription.lstmLabel.bias_ih_l0',
                      'decription.lstmLabel.bias_hh_l0', 'decription.lstmLabel.weight_ih_l0_reverse',
                      'decription.lstmLabel.weight_hh_l0_reverse', 'decription.lstmLabel.bias_ih_l0_reverse',
                      'decription.lstmLabel.bias_hh_l0_reverse', 'decription.label_dynamic_gru.gru.weight_ih_l0',
                      'decription.label_dynamic_gru.gru.weight_hh_l0', 'decription.label_dynamic_gru.gru.bias_ih_l0',
                      'decription.label_dynamic_gru.gru.bias_hh_l0',
                      'decription.label_dynamic_gru.gru.weight_ih_l0_reverse',
                      'decription.label_dynamic_gru.gru.weight_hh_l0_reverse',
                      'decription.label_dynamic_gru.gru.bias_ih_l0_reverse',
                      'decription.label_dynamic_gru.gru.bias_hh_l0_reverse',
                      'RSANModel_Sub.fact_dynamic_lstm.gru.weight_ih_l0',
                      'RSANModel_Sub.fact_dynamic_lstm.gru.weight_hh_l0',
                      'RSANModel_Sub.fact_dynamic_lstm.gru.bias_ih_l0',
                      'RSANModel_Sub.fact_dynamic_lstm.gru.bias_hh_l0',
                      'RSANModel_Sub.fact_dynamic_lstm.gru.weight_ih_l0_reverse',
                      'RSANModel_Sub.fact_dynamic_lstm.gru.weight_hh_l0_reverse',
                      'RSANModel_Sub.fact_dynamic_lstm.gru.bias_ih_l0_reverse',
                      'RSANModel_Sub.fact_dynamic_lstm.gru.bias_hh_l0_reverse', 'RSANModel_Sub.embed.weight',
                      'RSANModel_Sub.lstmInput.weight_ih_l0', 'RSANModel_Sub.lstmInput.weight_hh_l0',
                      'RSANModel_Sub.lstmInput.bias_ih_l0', 'RSANModel_Sub.lstmInput.bias_hh_l0',
                      'RSANModel_Sub.lstmInput.weight_ih_l0_reverse', 'RSANModel_Sub.lstmInput.weight_hh_l0_reverse',
                      'RSANModel_Sub.lstmInput.bias_ih_l0_reverse', 'RSANModel_Sub.lstmInput.bias_hh_l0_reverse',
                      'RSANModel.norm.a_2', 'RSANModel.norm.b_2', 'RSANModel.final_fc.fc.0.weight',
                      'RSANModel.final_fc.fc.0.bias', 'RSANModel.final_fc.fc.1.weight', 'RSANModel.final_fc.fc.1.bias',
                      'RSANModel.final_fc.fc.1.running_mean', 'RSANModel.final_fc.fc.1.running_var',
                      'RSANModel.final_fc.fc.1.num_batches_tracked', 'RSANModel.final_fc.fc.3.weight',
                      'RSANModel.final_fc.fc.3.bias', 'coatt1.norm.a_2', 'coatt1.norm.b_2',
                      'coatt1.final_fc.fc.0.weight', 'coatt1.final_fc.fc.0.bias', 'coatt1.final_fc.fc.1.weight',
                      'coatt1.final_fc.fc.1.bias', 'coatt1.final_fc.fc.1.running_mean',
                      'coatt1.final_fc.fc.1.running_var', 'coatt1.final_fc.fc.1.num_batches_tracked',
                      'coatt1.final_fc.fc.3.weight', 'coatt1.final_fc.fc.3.bias', 'coatt2.norm.a_2', 'coatt2.norm.b_2',
                      'coatt2.final_fc.fc.0.weight', 'coatt2.final_fc.fc.0.bias', 'coatt2.final_fc.fc.1.weight',
                      'coatt2.final_fc.fc.1.bias', 'coatt2.final_fc.fc.1.running_mean',
                      'coatt2.final_fc.fc.1.running_var', 'coatt2.final_fc.fc.1.num_batches_tracked',
                      'coatt2.final_fc.fc.3.weight', 'coatt2.final_fc.fc.3.bias', 'coatt3.norm.a_2', 'coatt3.norm.b_2',
                      'coatt3.final_fc.fc.0.weight', 'coatt3.final_fc.fc.0.bias', 'coatt3.final_fc.fc.1.weight',
                      'coatt3.final_fc.fc.1.bias', 'coatt3.final_fc.fc.1.running_mean',
                      'coatt3.final_fc.fc.1.running_var', 'coatt3.final_fc.fc.1.num_batches_tracked',
                      'coatt3.final_fc.fc.3.weight', 'coatt3.final_fc.fc.3.bias', 'coatt4.norm.a_2', 'coatt4.norm.b_2',
                      'coatt4.final_fc.fc.0.weight', 'coatt4.final_fc.fc.0.bias', 'coatt4.final_fc.fc.1.weight',
                      'coatt4.final_fc.fc.1.bias', 'coatt4.final_fc.fc.1.running_mean',
                      'coatt4.final_fc.fc.1.running_var', 'coatt4.final_fc.fc.1.num_batches_tracked',
                      'coatt4.final_fc.fc.3.weight', 'coatt4.final_fc.fc.3.bias', 'coatt5.norm.a_2', 'coatt5.norm.b_2',
                      'coatt5.final_fc.fc.0.weight', 'coatt5.final_fc.fc.0.bias', 'coatt5.final_fc.fc.1.weight',
                      'coatt5.final_fc.fc.1.bias', 'coatt5.final_fc.fc.1.running_mean',
                      'coatt5.final_fc.fc.1.running_var', 'coatt5.final_fc.fc.1.num_batches_tracked',
                      'coatt5.final_fc.fc.3.weight', 'coatt5.final_fc.fc.3.bias', 'coatt6.norm.a_2', 'coatt6.norm.b_2',
                      'coatt6.final_fc.fc.0.weight', 'coatt6.final_fc.fc.0.bias', 'coatt6.final_fc.fc.1.weight',
                      'coatt6.final_fc.fc.1.bias', 'coatt6.final_fc.fc.1.running_mean',
                      'coatt6.final_fc.fc.1.running_var', 'coatt6.final_fc.fc.1.num_batches_tracked',
                      'coatt6.final_fc.fc.3.weight', 'coatt6.final_fc.fc.3.bias', 'coatt7.norm.a_2', 'coatt7.norm.b_2',
                      'coatt7.final_fc.fc.0.weight', 'coatt7.final_fc.fc.0.bias', 'coatt7.final_fc.fc.1.weight',
                      'coatt7.final_fc.fc.1.bias', 'coatt7.final_fc.fc.1.running_mean',
                      'coatt7.final_fc.fc.1.running_var', 'coatt7.final_fc.fc.1.num_batches_tracked',
                      'coatt7.final_fc.fc.3.weight', 'coatt7.final_fc.fc.3.bias', 'coatt8.norm.a_2', 'coatt8.norm.b_2',
                      'coatt8.final_fc.fc.0.weight', 'coatt8.final_fc.fc.0.bias', 'coatt8.final_fc.fc.1.weight',
                      'coatt8.final_fc.fc.1.bias', 'coatt8.final_fc.fc.1.running_mean',
                      'coatt8.final_fc.fc.1.running_var', 'coatt8.final_fc.fc.1.num_batches_tracked',
                      'coatt8.final_fc.fc.3.weight', 'coatt8.final_fc.fc.3.bias', 'final_fc.fc.0.weight',
                      'final_fc.fc.0.bias', 'final_fc.fc.1.weight', 'final_fc.fc.1.bias', 'final_fc.fc.1.running_mean',
                      'final_fc.fc.1.running_var', 'final_fc.fc.1.num_batches_tracked', 'final_fc.fc.3.weight',
                      'final_fc.fc.3.bias', 'fc1.fc.0.weight', 'fc1.fc.0.bias', 'fc1.fc.1.weight', 'fc1.fc.1.bias',
                      'fc1.fc.1.running_mean', 'fc1.fc.1.running_var', 'fc1.fc.1.num_batches_tracked',
                      'fc1.fc.3.weight', 'fc1.fc.3.bias', 'fc2.fc.0.weight', 'fc2.fc.0.bias', 'fc2.fc.1.weight',
                      'fc2.fc.1.bias', 'fc2.fc.1.running_mean', 'fc2.fc.1.running_var', 'fc2.fc.1.num_batches_tracked',
                      'fc2.fc.3.weight', 'fc2.fc.3.bias', 'fc3.fc.0.weight', 'fc3.fc.0.bias', 'fc3.fc.1.weight',
                      'fc3.fc.1.bias', 'fc3.fc.1.running_mean', 'fc3.fc.1.running_var', 'fc3.fc.1.num_batches_tracked',
                      'fc3.fc.3.weight', 'fc3.fc.3.bias', 'fc4.fc.0.weight', 'fc4.fc.0.bias', 'fc4.fc.1.weight',
                      'fc4.fc.1.bias', 'fc4.fc.1.running_mean', 'fc4.fc.1.running_var', 'fc4.fc.1.num_batches_tracked',
                      'fc4.fc.3.weight', 'fc4.fc.3.bias', 'fc5.fc.0.weight', 'fc5.fc.0.bias', 'fc5.fc.1.weight',
                      'fc5.fc.1.bias', 'fc5.fc.1.running_mean', 'fc5.fc.1.running_var', 'fc5.fc.1.num_batches_tracked',
                      'fc5.fc.3.weight', 'fc5.fc.3.bias', 'fc6.fc.0.weight', 'fc6.fc.0.bias', 'fc6.fc.1.weight',
                      'fc6.fc.1.bias', 'fc6.fc.1.running_mean', 'fc6.fc.1.running_var', 'fc6.fc.1.num_batches_tracked',
                      'fc6.fc.3.weight', 'fc6.fc.3.bias', 'fc7.fc.0.weight', 'fc7.fc.0.bias', 'fc7.fc.1.weight',
                      'fc7.fc.1.bias', 'fc7.fc.1.running_mean', 'fc7.fc.1.running_var', 'fc7.fc.1.num_batches_tracked',
                      'fc7.fc.3.weight', 'fc7.fc.3.bias', 'fc8.fc.0.weight', 'fc8.fc.0.bias', 'fc8.fc.1.weight',
                      'fc8.fc.1.bias', 'fc8.fc.1.running_mean', 'fc8.fc.1.running_var', 'fc8.fc.1.num_batches_tracked',
                      'fc8.fc.3.weight', 'fc8.fc.3.bias']
        print("\nLoading model from {}...".format(args.load_gen))
        gen = torch.load(args.load_gen)
        for i in param_list:
            model.load_state_dict({i:gen[i]}, strict=False)


    if args.snapshot is not None:
        print("\nLoading model from {}...".format(args.snapshot))
        model.load_state_dict(torch.load(args.snapshot))
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    if args.cuda:
        print("cuda")
        #torch.cuda.set_device(args.gpu)
        model = model.cuda()

    print('training start!')
    train.train(train_iter, dev_iter, model, args)
    print('training finished!')


if __name__ == '__main__':
    main()
