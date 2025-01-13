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
                        default=1,
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
                        default=0,
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
    if args.correlated==0:
        print('decorrelated')
        train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

        dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
    else:
        print('correlated')
        train_data = Beer_correlated(args.data_dir, args.aspect, 'train', word2idx, balance=True)

        dev_data = Beer_correlated(args.data_dir, args.aspect, 'dev', word2idx,balance=True)

    annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
elif args.data_type == 'hotel':       #hotel
    args.data_dir='./data/hotel'
    args.annotation_path='./data/hotel/annotations'
    train_data = HotelData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

    dev_data = HotelData(args.data_dir, args.aspect, 'dev', word2idx)

    annotation_data = HotelAnnotation(args.annotation_path, args.aspect, word2idx)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

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
para = []
if args.share == 0:
    print('share=0')
    para.append({'params': model.gen_list[0].parameters(), 'lr': args.lr})
    para.append({'params': model.gen_list[1].parameters(), 'lr': args.lr})
    para.append({'params': model.cls_fc.parameters(), 'lr': args.lr / 2})
    para.append({'params': model.cls.parameters(), 'lr': args.lr / 2})
else:
    print('share=1')
    para.append({'params': model.gen_list[0].parameters(), 'lr': args.lr})
    para.append({'params': model.cls_fc.parameters(), 'lr': args.lr / 2})
    para.append({'params': model.cls.parameters(), 'lr': args.lr / 2})




optimizer = torch.optim.Adam(para)




# optimizer = torch.optim.Adam(model.parameters())

######################
# Training
######################
strat_time = time.time()
best_all = 0.0
f1_best_dev = [0.0]
best_dev_epoch = [0]
acc_best_dev = [0.0]
grad = []
grad_loss = []


for epoch in range(args.epochs):
    start = time.time()
    model.train()


    precision, recall, f1_score, accuracy = train_cooperativegame(model, optimizer, train_loader, device, args,
                                                            (writer, epoch), grad, grad_loss)

    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    print(
        f"training epoch:{epoch} recall:{recall:.4f} precision:{precision:.4f} f1-score:{f1_score:.4f} accuracy:{accuracy:.4f}")

    writer.add_scalar('train_acc', accuracy, epoch)
    writer.add_scalar('time', time.time() - strat_time, epoch)


    model.eval()
    print("Validate")
    with torch.no_grad():

        TP = 0
        TN = 0
        FN = 0
        FP = 0


        for batch, (inputs, masks, labels) in enumerate(dev_loader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)


            rationales_list, logits_list = model(inputs, masks)


            for idx in range(len(rationales_list)):
                logits = logits_list[idx]
                logits = torch.softmax(logits, dim=-1)
                _, pred = torch.max(logits, axis=-1)


                TP += ((pred == 1) & (labels == 1)).cpu().sum().item()
                TN += ((pred == 0) & (labels == 0)).cpu().sum().item()
                FN += ((pred == 0) & (labels == 1)).cpu().sum().item()
                FP += ((pred == 1) & (labels == 0)).cpu().sum().item()


        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        print(
            f"dev epoch:{epoch} recall:{recall:.4f} precision:{precision:.4f} f1-score:{f1_score:.4f} accuracy:{accuracy:.4f}")

        writer.add_scalar('dev_acc', accuracy, epoch)
        print("Validate Sentence")
        validate_dev_sentence(model, dev_loader, device, (writer, epoch))

        print("Annotation")
        if args.average_test == 1:
            annotation_results = validate_share(model, annotation_loader, device)
        elif args.average_test == 0:
            print('one_head_test')
            annotation_results = validate_onehead(model, annotation_loader, device)


        if annotation_results:
            print(
                f"The annotation performance: sparsity: {100 * annotation_results[0]:.4f}, precision: {100 * annotation_results[1]:.4f}, "
                f"recall: {100 * annotation_results[2]:.4f}, f1: {100 * annotation_results[3]:.4f}"
            )


            writer.add_scalar('f1', 100 * annotation_results[3], epoch)
            writer.add_scalar('sparsity', 100 * annotation_results[0], epoch)
            writer.add_scalar('p', 100 * annotation_results[1], epoch)
            writer.add_scalar('r', 100 * annotation_results[2], epoch)

            print("Annotation Sentence")
            validate_annotation_sentence(model, annotation_loader, device)

            print("Rationale")
            validate_rationales(model, annotation_loader, device, (writer, epoch))


            if accuracy > acc_best_dev[-1]:
                acc_best_dev[-1] = accuracy
                best_dev_epoch[0] = epoch
                f1_best_dev[-1] = annotation_results[3]


            if annotation_results[3] > best_all:
                best_all = annotation_results[3]


print(f"best_all={best_all:.4f}")
print(f"acc_best_dev={acc_best_dev}")
print(f"best_dev_epoch={best_dev_epoch}")
print(f"f1_best_dev={f1_best_dev}")


if args.save == 1:
    if args.data_type == 'beer':
        torch.save(model, f'./trained_model/beer/aspect{args.aspect}_dis{args.dis_lr}.pkl')
        print('save the model')
    elif args.data_type == 'hotel':
        torch.save(model, f'./trained_model/hotel/aspect{args.aspect}_dis{args.dis_lr}.pkl')
        print('save the model')
else:
    print('not save')