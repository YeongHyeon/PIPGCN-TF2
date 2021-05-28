import argparse, time, os, operator

import numpy as np
import tensorflow as tf
import source.connector as con
import source.tf_process as tfp
import source.datamanager as dman

def main():

    dataset = dman.DataSet(dir=FLAGS.dir)

    for idx, phase in enumerate(['training', 'test', 'validation']):
        for idx_r in range(3):
            checker = np.asarray([0, 0])
            print("Round ", idx_r)
            while(True):
                minibatch, terminate = dataset.next_batch(batch_size=FLAGS.batch, ttv=idx)
                if(len(minibatch.keys()) == 0): break
                checker = checker + np.sum(minibatch['label_1hot'], axis=0)
                if(terminate): break
            print("* %s" %(phase))
            print("Negative : %7d" %(checker[0]))
            print("Positive : %7d" %(checker[1]))
            print("Total    : %7d" %(np.sum(checker)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nn', type=int, default=0, help='')
    parser.add_argument('--dir', type=str, default='dataset', help='')
    parser.add_argument('--batch', type=int, default=32, help='')

    FLAGS, unparsed = parser.parse_known_args()

    main()
