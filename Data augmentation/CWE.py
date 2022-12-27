\
import argparse
import pandas as pd
import numpy as np
import torch
import os
from transformers import MarianMTModel, MarianTokenizer
import nlpaug.augmenter.word as naw
import sys
sys.path.append('.')


def main(args):

    print(" Start CWE on samples")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    topics = ["CT20-AR-01" ,"CT20-AR-02", "CT20-AR-05", "CT20-AR-08", "CT20-AR-10",
    "CT20-AR-12", "CT20-AR-14", "CT20-AR-19", "CT20-AR-23", "CT20-AR-27", "CT20-AR-30",
    "Covid-19", "CT21-AR-01", "CT21-AR-02"]

    smpl_set = ["CT20-AR-01_smpl_200" ,"CT20-AR-02_smpl_200", "CT20-AR-05_smpl_200", "CT20-AR-08_smpl_200", "CT20-AR-10_smpl_200",
        "CT20-AR-12_smpl_200", "CT20-AR-14_smpl_200", "CT20-AR-19_smpl_200", "CT20-AR-23_smpl_200", "CT20-AR-27_smpl_200", "CT20-AR-30_smpl_200",
        "Covid-19_smpl_200", "CT21-AR-01_smpl_200", "CT21-AR-02_smpl_200"]

    suffix = '.tsv'
    counter = 0
    separators = ['[مستخدم]', 'رابط]', '[رقم]']

    for i in range(0,14):
        counter = counter +1
        print("Itr. no.:", counter)
        print("topic name:", topics[i])

        samples_path = os.path.join(args.samples_data_path,smpl_set[i]+suffix)
        samples_data = pd.read_csv(samples_path, delimiter = "\t" ,encoding='utf-8')
        input_list = samples_data['tweet_text'].astype('string').tolist()
        if counter ==1:
            aug = naw.ContextualWordEmbsAug(model_path='aubmindlab/bert-base-arabertv02', aug_p=0.3)

        augmented_text = []
        for sentence in input_list :
            res = sentence
            for separator in separators:
                res = res.replace(separator,f'-{separator}-')
            splited_sentence = res.split('-')
            for index , item in enumerate(splited_sentence):
                splited_sentence[index] = item.strip()
            splited_sentence = list(filter(None,splited_sentence))	
            for index , item in enumerate(splited_sentence):
                if item not in separators:
                    splited_sentence[index] = aug.augment(item)
            splited_sentence = ''.join(''.join(l) for l in splited_sentence)
            augmented_text.append(splited_sentence)
            
        samples_data["WE_text"] = augmented_text

        output_suffix= '_WE.tsv'
        output_path = os.path.join(args.output_dir, smpl_set[i] + output_suffix)
        samples_data.to_csv(output_path, sep='\t', index=False, header = True, encoding='utf-8')

    print(' CWE done successfully.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')

    parser.add_argument('--samples_data_path', default="./samples_200", help='.path of samples')
    parser.add_argument('--output_dir', default="./WE_200", help='CWE samples output directory')

    parser.add_argument('--device', default='cpu', help='Device')
	
    args = parser.parse_args()
    main(args)