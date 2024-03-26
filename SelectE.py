from builddata_softplus import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from common import *
from buildtrain import *
from model import *
from datetime import datetime

# 超参数设置
parser = ArgumentParser("SelectE", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--data_path", default="./data/", help="Data sources.")
parser.add_argument("--run_folder", default="./", help="Data sources.")
parser.add_argument("--data_name", default="FB15k-237", help="Name of the dataset.")
parser.add_argument("--embedding_dim", default=300, type=int, help="Entity/Relation dimension")
parser.add_argument("--min_lr", default=5e-5, type=float, help='L2 regularization')
parser.add_argument("--batch_size", default=1000, type=int, help='Batch Size')
parser.add_argument("--log_epoch", default=2, type=int, help='how many batches to wait before logging training status')
parser.add_argument("--neg_ratio", default=1.0, help="Number of negative triples generated by positive (default: 1.0)")
parser.add_argument("--input_drop", default=0.4, type=float, help="Dropout on input layer and FC layers in SelectE building blocks")
parser.add_argument("--hidden_drop", default=0.3, type=float, help="Dropout on input layer and FC layers in SelectE building blocks")
parser.add_argument("--feature_map_drop", default=0.3, type=float, help="Dropout on input layer and FC layers in SelectE building blocks")
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--opt", default="Adam", type=str)
parser.add_argument("--learning_rate", default=0.003, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=5e-8, type=float)
parser.add_argument("--factor", default=0.8, type=float)
parser.add_argument("--verbose", default=1, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--max_mrr", default=0.35, type=float)
parser.add_argument("--epoch", default=1000, type=int)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--save_name", default='wn18rr.pt')
parser.add_argument('--output_channel', dest="output_channel", default=20, type=int, help='Number of output channel')
parser.add_argument('--k_w', dest="k_w", default=10, type=int, help='Width of the reshaped matrix')
parser.add_argument('--k_h', dest="k_h", default=20, type=int, help='Height of the reshaped matrix')
parser.add_argument('--filter1_size', type=int, nargs=2, default=[1, 5], help='first kernel size of conv layer')
parser.add_argument('--filter2_size', type=int, nargs=2, default=[3, 3], help='second kernel size of conv layer')
parser.add_argument('--filter3_size', type=int, nargs=2, default=[1, 9], help='third kernel size of conv layer')
# Logging parameters
parser.add_argument("--name", default='runlog_' + str(datetime.now().strftime("%Y-%m-%d_%H-%M")), help='Name of the experiment')
parser.add_argument('--logdir', dest="log_dir", default='./log/', help='Log directory')
parser.add_argument('--config', dest="config_dir", default='./config/', help='Config directory')
parser.add_argument('--seed', type=int, dest="seed", default='2022', help='random seed')
args = parser.parse_args()
setup_seed(args.seed)
cuda_num = int(args.device[-1])
torch.cuda.set_device(cuda_num)

# 数据集预处理和读取
config_dir = args.config_dir
log_dir = args.log_dir
exp_name = args.name
logger = get_logger(exp_name, log_dir, config_dir, args.epoch)
logger.info(vars(args))

pprint(vars(args))
train, valid, test, words_indexes, indexes_words, headTailSelector, entity2id, id2entity, relation2id, id2relation = build_data(path=args.data_path,
                                                                                                                                name=args.data_name)
train_doubles, valid_doubles, test_doubles = get_doubles(train, valid, test, words_indexes)  # 翻转三元组，关系重新定义
x_valid = np.array(valid_doubles).astype(np.int32)
x_test = np.array(test_doubles).astype(np.int32)
rel_set = get_rel_set(train, valid, test)  # 去重的关系
vocab_size = max(rel_set) + len(words_indexes) + 1  # 计算用于模型训练的词汇表大小 1表示保留一个位置用于未知单词。这个计算结果将用于构建神经网络模型中的嵌入层（embedding layer），以便将输入的单词和关系转换为向量表示。
target_dict = get_target_dict(train_doubles, x_valid, x_test)  # {(头实体，关系):(尾实体)}

# 创建模型并初始化
model = SelectE(logger, vocab_size, embedding_dim=args.embedding_dim, input_drop=args.input_drop, hidden_drop=args.hidden_drop, feature_map_drop = args.feature_map_drop,
            k_w = args.k_w, k_h = args.k_h, output_channel= args.output_channel, filter1_size = args.filter1_size, filter2_size = args.filter2_size, filter3_size = args.filter3_size)
device = args.device
model.init()

# 优化器设置
if args.opt == 'Adam':
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.opt == 'SGD':
    opt = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=args.factor, verbose=args.verbose, min_lr=args.min_lr,
                                                       patience=args.patience)

# 开始训练模型
num_batches_per_epoch = len(train_doubles) // args.batch_size + 1
model = train_epoch(train_doubles, num_batches_per_epoch, args.batch_size, model, opt, scheduler, x_valid, target_dict, args.log_epoch,
                    args.device, max_mrr=args.max_mrr, epoch=args.epoch, x_test=x_test, logger=logger)

# 模型性能评估
model.eval()
torch.save(model, f'{args.save_name}')
model = torch.load(args.save_name)
final_result = evaluate(model, x_test, args.batch_size, target_dict)
mr_final, mrr_final, hit1_final,hit3_final, hit10_final = final_result['mr'], final_result['mrr'], final_result['hits1'],final_result['hits3'], final_result['hits10']
logger.info('[Final result]: MR: {}, MRR: {}, Hits@1: {}, Hits@3: {},Hits@10: {}'.format(mr_final, mrr_final, hit1_final, hit3_final,hit10_final))