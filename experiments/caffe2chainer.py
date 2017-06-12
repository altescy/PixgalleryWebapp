import argparse
import pickle
from chainer.links.caffe import CaffeFunction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Convert from caffe model to chainer model')
    parser.add_argument('-m', '--model', type=str, required=True,
            help='path to caffe model file')
    parser.add_argument('-o', '--out', type=str, default=None,
            help='path to output file')
    args = parser.parse_args()

    if not args.out:
        outfile = (args.model).rsplit('.', 1)[0] + '.chainermodel.pkl'
    else:
        outfile = args.model
    vgg = CaffeFunction(args.model)
    pickle.dump(vgg, open(outfile, 'wb'))

