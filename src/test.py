from pydub import AudioSegment
from annoy import AnnoyIndex
from sklearn import preprocessing


import os
import numpy as np
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('../tool')

import model
import toolkits
import utils as ut
import embedding_extraction

filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ann_based_speaker.log')

base_path = os.path.dirname(os.path.realpath(__file__))
WEIGHT_PATH = os.path.join(base_path, 'speaker_embedding/model/resnet34_vlad8_ghost2_bdim512_deploy/weights.h5')

le = preprocessing.LabelEncoder()
le.classes_ = np.load('labels.npy')


ann = AnnoyIndex(512)
ann.load('speaker_emb.bin')

network_params = {'gpu': '1', 'resume': WEIGHT_PATH, 'batch_size': 16,
                  'net':'resnet34s', 'ghost_cluster': 2, 'vlad_cluster': 8,
                  'bottleneck_dim': 512, 'loss': 'softmax', 'aggregation_mode': 'gvlad'
                  }


def convert_to_wav(filepath):
    (path, file_extension) = os.path.splitext(filepath)
    file_extension = file_extension.replace('.', '')
    track = AudioSegment.from_file(filepath, file_extension)
    if track.duration_seconds > 0.16:
        track = track[0:30000]
    track.export('temp.wav', format='wav')


def predict(audio_file):
    if not audio_file.endswith('wav'):
        print('Fixing file format')
        convert_to_wav(audio_file)
        audio_file = 'temp.wav'

    embd = embedding_extraction.extract_embeddings(audio_file, mode='test')
    label = ann.get_nns_by_vector(embd, 3, search_k=1000)
    print('Top prediction')
    print(le.inverse_transform(label))


if __name__ == '__main__':
    input_file = '/home/robot/Downloads/y2mate.com - Donald Trump Really Wants You to Take This Drug_ecJXUX1XmfY_144p.mp4'
    predict(input_file)
