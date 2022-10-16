import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])

import numpy as np
import os
from scipy.io import loadmat

max_frame = 130

def gendata(data_path, out_path, roinum, part,isplit):
    inputfilename = data_path + 'ADNI_TS_' + roinum + 'ROI__' + part + '_5split_' + str(isplit) + '.mat'
    matdata = loadmat(inputfilename)

    sample_label = matdata['label'].transpose().tolist()[0]
    sample_name = []
    for i, s in enumerate(tqdm(range(len(sample_label)))):
        sample_name.append(part + '_TS_' + roinum + str(i))

    with open('{}/{}_5split_{}_{}_label.pkl'.format(out_path, roinum, part,isplit), 'wb') as f:
        pickle.dump((sample_name,  list(sample_label)), f)

    fp = np.zeros((len(sample_label), 1, max_frame, int(roinum),6), dtype=np.float32)

    for i, s in enumerate(tqdm(range(len(sample_label)))):
        data = matdata['ROISignals'][i][0]

        data = np.stack([data,
                         matdata['ROISignals_5bands'][0][0][i][0],
                         matdata['ROISignals_5bands'][1][0][i][0],
                         matdata['ROISignals_5bands'][2][0][i][0],
                         matdata['ROISignals_5bands'][3][0][i][0],
                         matdata['ROISignals_5bands'][4][0][i][0]],axis=2)
        fp[i, 0, 0:data.shape[0], :,:] = data
        # nframe = data.shape[1]
        # nrep = int(np.floor(max_frame / nframe))
        # for irep in range(nrep):
        #     fp[i, :, (0 + irep * nframe):((irep + 1) * nframe), :] = data
        # fp[i, :, (0 + nrep * nframe):max_frame, :] = data[:,0:(max_frame-nrep*nframe),:]

    #fp = pre_normalization(fp)
    np.save('{}/{}_5split_{}_{}_data.npy'.format(out_path, roinum, part,isplit), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADHD Data Converter.')
    parser.add_argument('--data_path', default='/data/datasets/ADNI/')
    parser.add_argument('--out_folder', default='/data/datasets/ADNI/Processed/')

    Roinums = ['90', '42']
    part = ['train', 'test']
    arg = parser.parse_args()

    for roinum in Roinums:
        for p in part:
            out_path = os.path.join(arg.out_folder, roinum)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for isplit in range(1,11):
                print(roinum, p,isplit)

                gendata(
                    arg.data_path,
                    out_path,
                    roinum=roinum,
                    part=p, isplit=isplit)
