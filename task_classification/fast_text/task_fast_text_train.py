# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:zhubinnju@outlook.com
@license: Apache Licence
@software:PyCharm
@file: train_fast_text.py
@time: 2022/7/12 16:30
"""
import argparse
import fasttext



def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

if __name__ == '__main__':
    parse=argparse.ArgumentParser(description="fast text train ")
    parse.add_argument("--train_data",type=str,help="train data path")
    parse.add_argument("--dev_data",type=str,help="test data path")
    parse.add_argument("--lr",type=float,default=0.3,help="learning rate")
    parse.add_argument("--epochs",type=int,default=20,help="training epochs")
    parse.add_argument("--dim",type=int,default=200,help="word embedding dim")
    args=parse.parse_args()

    
    classifier = fasttext.train_supervised(args.train_data, label='__label__',
                                           wordNgrams=2, epoch=args.epochs, lr=args.lr,
                                           dim=args.dim)
    print(classifier.labels)
    test_result = classifier.test(args.dev_data)
    classifier.save_model("model.bin")
    print_results(*test_result)
