from __future__ import absolute_import
from __future__ import print_function
from pathlib import Path

import os
import sys
import model
import numpy as np
import pickle
import logging

import toolkits
import utils as ut

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--resume', default='model/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='speaker_embedding/data/audio', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()

base_path = '/kaggle/working'


def extract_embeddings(mode='train', gpu='0', data_path=None):
    toolkits.initialize_GPU(gpu)
    net_args = {
    'resume':'weights.h5',
    'batch_size':16,
    'data_path':data_path,
    'net':'resnet34s',
    'ghost_cluster':2,
    'vlad_cluster':8,
    'bottleneck_dim':512,
    'aggregation_mode':'gvlad',
    'loss':'softmax',
    'test_type':'normal'
    }

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', net_args=net_args)

    if net_args.get('resume'):
        weight_path = os.path.join(base_path, net_args.get('resume'))
        if os.path.isfile(weight_path):
            print('loading graph')
            network_eval.load_weights(weight_path, by_name=True)
        else:
            return 'Issue with loading graph'
    else:
        return 'Pre-trained graph is required'

    if mode == 'train':
        audio_files = [filename for filename in Path(input_path).rglob('*.wav')]
        total_files = len(audio_files) * 10
        working_file = 0
        emb_store = {}
        for audio in audio_files:
            print(f'processing {os.path.basename(os.path.dirname(audio))} ')
            specs = ut.load_data_aug(audio, win_length=params['win_length'], sr=params['sampling_rate'],
                                 hop_length=params['hop_length'], n_fft=params['nfft'],
                                 spec_len=params['spec_len'], mode='eval')
            count_file = 0
            for sample in specs:
                print(f'Augmentation count is {count_file}')
                print(f'Processing file {working_file} of {total_files}')
                sample_spec = np.expand_dims(np.expand_dims(sample, 0), -1)
                class_label = os.path.basename(os.path.dirname(audio))
                v = network_eval.predict(sample_spec)

                old_data = []
                if class_label in emb_store.keys():
                    pre_data = emb_store.get(class_label)
                    pre_data.append(v[0])
                    old_data = pre_data
                else:
                    old_data.append(v[0])
                emb_store[class_label] = old_data

                count_file += 1
                working_file += 1
                logging.info(f'For {audio} label stored is {class_label}')

        with open('training_features_augmented.pickle', 'wb') as handle:
            pickle.dump(emb_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        specs = ut.load_data(data_path, win_length=params['win_length'], sr=params['sampling_rate'],
                                 hop_length=params['hop_length'], n_fft=params['nfft'],
                                 spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
        vector_embedding = network_eval.predict(specs)[0]
        return vector_embedding


if __name__ == "__main__":
    extract_embeddings()

"""
 py embedding_extraction.py --gpu 0 --net resnet34s --ghost_cluster 2 --vlad_cluster 8 --loss softmax --resume ../model/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5
"""

