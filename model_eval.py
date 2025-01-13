import argparse
import os
import time

import torch

from beer import BeerData,Beer_correlated,BeerAnnotation
from hotel import HotelData,HotelAnnotation
from embedding import get_compressed_embeddings,get_pretrained_glove_embedding
from torch.utils.data import DataLoader

from model import CooperativeGame
from train_util import train_cooperativegame
from validate_util import validate_share,validate_onehead, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from tensorboardX import SummaryWriter


def parse():
    #： nonorm, dis_lr=1, data=beer, save=0
    parser = argparse.ArgumentParser(
        description="classwise rationalization for beer review")
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')
    parser.add_argument('--correlated',
                        type=int,
                        default=0,
                        help='Max sequence length [default: 256]')

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/beer',
                        help='Path of the dataset')
    parser.add_argument('--data_type',
                        type=str,
                        default='beer',
                        help='0:beer,1:hotel')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/beer/annotations.json',
                        help='Path to the annotation')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 256]')


    # model parameters
    parser.add_argument('--dis_lr',
                        type=int,
                        default=1,
                        help='number generator')
    parser.add_argument('--average_test',
                        type=int,
                        default=1,
                        help='0: No, 1: Yes')
    parser.add_argument('--num_gen',
                        type=int,
                        default=2,
                        help='player1,player2')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='share encoder')
    parser.add_argument('--save',
                        type=int,
                        default=1,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 200]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')
    parser.add_argument('--cooperative_game',
                        type=int,
                        default=1,
                        help='Whether to play Cooperative Game, 0: No, 1: Yes [default: 1]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='Ture',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=400,
                        help='Number of training epoch')
    parser.add_argument('--lr_lambda',
                        type=float,
                        default=1,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=11.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=12.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.2,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument(
        '--writer',
        type=str,
        default='./tensorboard',
        help='Regularizer to control highlight percentage [default: .2]')

    # visual parameters
    parser.add_argument(
        '--visual_interval',
        type=int,
        default=50,
        help='How frequent to generate a sample of rationale [default: 50]')
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################
# torch.manual_seed(args.seed)

#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed)


######################
# load embedding
######################
pretrained_embedding, word2idx = get_pretrained_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################
if args.data_type=='beer':       #beer


    annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
elif args.data_type == 'hotel':       #hotel
    args.data_dir='./data/hotel'
    args.annotation_path='./data/hotel/annotations'


    annotation_data = HotelAnnotation(args.annotation_path, args.aspect, word2idx)

# shuffle and batch the dataset


annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
writer=SummaryWriter(args.writer)
model=CooperativeGame(args)
model.to(device)

######################
# Training
######################
# Set up parameters for CooperativeGame
# generator_lr = 1e-3
# predictor_lr = generator_lr / 10
# para = [
#     {'params': player1.parameters(), 'lr': generator_lr},
#     {'params': player2.parameters(), 'lr': generator_lr},
#     {'params': model.predictor.parameters(), 'lr': predictor_lr}
# ]

# para = []
# if args.share == 0:
#     print('share=0')
#     generator_lr = 1e-3
#     predictor_lr = generator_lr / 10
#     para.append({'params': player1.parameters(), 'lr': generator_lr})
#     if args.dis_lr == 1:
#         multi_lr = args.lr_lambda
#         para.append({'params': player2.parameters(), 'lr': generator_lr + generator_lr * multi_lr})
#     else:
#         para.append({'params': player2.parameters(), 'lr': generator_lr})
#     para.append({'params': model.cls_fc.parameters(), 'lr': predictor_lr})
#     para.append({'params': model.cls.parameters(), 'lr': predictor_lr})
#     g_fc_para=filter(lambda p: id(p) not in g_para, model.parameters())
# optimizer = torch.optim.Adam(para)
# optimizer = torch.optim.Adam(model.parameters())
######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
grad=[]
grad_loss=[]
TP = 0
TN = 0
FN = 0
FP = 0
model=torch.load('./trained_model/beer/aspect0_dis1.pkl').eval()
with torch.no_grad():
    print("Annotation")
    if args.average_test==1:
        annotation_results = validate_share(model, annotation_loader, device)
    elif args.average_test==0:
        print('one_head_test')
        annotation_results = validate_onehead(model, annotation_loader, device)
    print(
        "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
        % (100 * annotation_results[0], 100 * annotation_results[1],
           100 * annotation_results[2], 100 * annotation_results[3]))



print(best_all)
print(acc_best_dev)
print(best_dev_epoch)
print(f1_best_dev)
