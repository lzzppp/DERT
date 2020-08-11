
import argparse
from train import train
from evaluate import evaluator, t2vec
import os
import gc
gc.collect()
## toy_data
## python t2vec.py -data="toy_data" -hidden_size=64 -embedding_size=64 -save_freq=100 -vocab_size=43 -epochs 20
## python t2vec.py -data="toy_data" -hidden_size=64 -embedding_size=64 -save_freq=100 -vocab_size=43 -epochs 20 -criterion_name="KLDIV" -knearestvocabs="toy_data/toy-vocab-dist.h5"

## server
## cell100
## python t2vec.py -data portocell100_data -vocab_size 18866
## python t2vec.py -data portocell100_data -vocab_size 18866 -criterion_name "KLDIV" -knearestvocabs "preprocessing/porto-vocab-dist-cell100.h5"
## cell50
## python t2vec.py -data portocell50_data -vocab_size 35335 -criterion_name "KLDIV" -knearestvocabs "preprocessing/porto-vocab-dist-cell50.h5"
## cell25
## python t2vec.py -data portocell25_data -vocab_size 60004 -criterion_name "KLDIV" -knearestvocabs "preprocessing/porto-vocab-dist-cell25.h5"

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data", default="data",
    help="Path to training and validating data")

parser.add_argument("-checkpoint", default="checkpoint.pt",
    help="The saved checkpoint")

parser.add_argument("-pretrained_embedding", default=None,
    help="Path to the pretrained word (cell) embedding")

parser.add_argument("-num_layers", type=int, default=3,
    help="Number of layers in the RNN cell")

parser.add_argument("-bidirectional", type=bool, default=True,
    help="True if use bidirectional rnn in encoder")

parser.add_argument("-hidden_size", type=int, default=256,
    help="The hidden state size in the RNN cell")

parser.add_argument("-embedding_size", type=int, default=256,
    help="The word (cell) embedding size")

parser.add_argument("-dropout", type=float, default=0.1,
    help="The dropout probability")

parser.add_argument("-max_grad_norm", type=float, default=5.0,
    help="The maximum gradient norm")

parser.add_argument("-learning_rate", type=float, default=0.001)

parser.add_argument("-batch", type=int, default=128,
    help="The batch size")

parser.add_argument("-generator_batch", type=int, default=128,
    help="""The maximum number of words to generate each time.
    The higher value, the more memory requires.""")

parser.add_argument("-t2vec_batch", type=int, default=1, 
    help="""The maximum number of trajs we encode each time in t2vec""")

parser.add_argument("-start_iteration", type=int, default=0)

parser.add_argument("-epochs", type=int, default=15,
    help="The number of training epochs")

parser.add_argument("-print_freq", type=int, default=20,
    help="Print frequency")

parser.add_argument("-save_freq", type=int, default=1000,
    help="Save frequency")

parser.add_argument("-cuda", type=bool, default=True,
    help="True if we use GPU to train the model")

parser.add_argument("-criterion_name", default="NLL",
    help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")

parser.add_argument("-knearestvocabs", default=None,
    help="""The file of k nearest cells and distances used in KLDIVLoss,
    produced by preprocessing, necessary if KLDIVLoss is used""")

parser.add_argument("-dist_decay_speed", type=float, default=0.8,
    help="""How fast the distance decays in dist2weight, a small value will
    give high weights for cells far away""")

parser.add_argument("-max_num_line", type=int, default=20000000)

parser.add_argument("-max_length", default=200,
    help="The maximum length of the target sequence")

parser.add_argument("-mode", type=int, default=0,
    help="Running mode (0: train, 1:evaluate, 2:t2vec)")

parser.add_argument("-vocab_size", type=int, default=0,
    help="Vocabulary Size")

parser.add_argument("-bucketsize", default=[(30, 30), (50, 50)],
    help="Bucket size for training")

parser.add_argument("-cityname", default="", type=str, help="Dataset City Name")

parser.add_argument("-grid_size", default=100.0, type=float, help="Grid Size(meters)")

parser.add_argument("-encode_data", default="train", type=str, help="Encode train.t or test.t")

parser.add_argument("-gpu_id", default="1", type=str, help="GPU id for this program")
args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
## __main__
#args.bucketsize = [(30, 30), (30, 60), (60, 60), (60, 90), (90, 90), (90, 120), (120, 120), (120, 150), (150, 150), (150, 180), (180, 180), (180, 210), (210, 210), (210, 240), (240, 240), (240, 270), (270, 270), (270, 300), (300, 300), (300, 330), (330, 330), (330, 360), (360, 360), (360, 390), (390, 390), (390, 420), (420, 420), (420, 450), (450, 450), (450, 480), (480, 480), (480, 510), (510, 510), (510, 540), (540, 540), (540, 570), (570, 570), (570, 600), (600, 600), (600, 630), (630, 630), (630, 660), (660, 660), (660, 690), (690, 690), (690, 720), (720, 720), (720, 750), (750, 750), (750, 780), (780, 780), (780, 810), (810, 810), (810, 840), (840, 840), (840, 870), (870, 870), (870, 900), (900, 900), (900, 930), (930, 930), (930, 960), (960, 960), (960, 990), (990, 990), (990, 1020), (1020, 1020), (1020, 1050), (1050, 1050), (1050, 1080), (1080, 1080), (1080, 1110), (1110, 1110), (1110, 1140), (1140, 1140), (1140, 1170), (1170, 1170), (1170, 1200), (1200, 1200), (1200, 1230), (1230, 1230), (1230, 1260), (1260, 1260), (1260, 1290), (1290, 1290), (1290, 1320), (1320, 1320), (1320, 1350), (1350, 1350), (1350, 1380), (1380, 1380), (1380, 1410), (1410, 1410), (1410, 1440), (1440, 1440)]
args.bucketsize = [(20, 30),(30, 30),(30,50),(50, 50),(50, 70),(70,70),(70,100),(100,100),(100,120),(120,120),(120,140),(140,140),
                   (140,160),(160,160),(160,180),(180,180),(180,200),(200,200),(200,220),(220,220),(220,240),(240,240),(240,260),
                   (260,260),(260,280),(280,280),(280,300),(300,300)]
#                    (500, 500),(500, 650),(650, 650)]
# args.bucketsize = [(20,30),(30,30),(30,50),(50,50),(50,70),(70,70),(70,100),(100,100),(100, 130),(130, 130),(160, 160),(200, 200),(200, 300),(300, 300),(300, 400),(400, 400),(400, 500),
#                    (500, 500),(500, 600),(600, 600),(600, 700),(700, 700)]
#args.bucketsize = [(10, 10), (20, 20), (20, 30)]
#args.vocab_size = 43

if args.mode == 1:
    evaluator(args)
elif args.mode == 2:
    t2vec(args)
else:
    train(args)
