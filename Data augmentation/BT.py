
import argparse
import pandas as pd
import numpy as np
import torch
import os

from transformers import MarianMTModel, MarianTokenizer
import sys
sys.path.append('.')


def BT(input_list, tokenizer, model):
    bt_list = []

    translated = model.generate(**tokenizer(input_list, return_tensors="pt", padding=True))

    for t in translated:
        text = tokenizer.decode(t, skip_special_tokens=True) 
        bt_list.append(text)

    return bt_list

def main(args):

    print(" Start Back translation on samples")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    topics = ["CT20-AR-01" ,"CT20-AR-02", "CT20-AR-05", "CT20-AR-08", "CT20-AR-10",
    "CT20-AR-12", "CT20-AR-14", "CT20-AR-19", "CT20-AR-23", "CT20-AR-27", "CT20-AR-30",
    "Covid-19", "CT21-AR-01", "CT21-AR-02"]

    smpl_set = ["CT20-AR-01_smpl_200" ,"CT20-AR-02_smpl_200", "CT20-AR-05_smpl_200", "CT20-AR-08_smpl_200", "CT20-AR-10_smpl_200",
        "CT20-AR-12_smpl_200", "CT20-AR-14_smpl_200", "CT20-AR-19_smpl_200", "CT20-AR-23_smpl_200", "CT20-AR-27_smpl_200", "CT20-AR-30_smpl_200",
        "Covid-19_smpl_200", "CT21-AR-01_smpl_200", "CT21-AR-02_smpl_200"]

    suffix = '.tsv'
    counter = 0
    for i in range(0,14):
        counter = counter +1
        print("Itr. no.:", counter)
        print("topic name:", topics[i])
        samples_path = os.path.join(args.samples_data_path,smpl_set[i]+suffix)
        samples_data = pd.read_csv(samples_path, delimiter = "\t" ,encoding='utf-8')
        input_list = samples_data['tweet_text'].astype('string').tolist()

        if counter ==1:
            ar_en_model_name = "Helsinki-NLP/opus-mt-tc-big-ar-en"
            global en_tokenizer 
            en_tokenizer = MarianTokenizer.from_pretrained(ar_en_model_name)
            global en_model 
            en_model = MarianMTModel.from_pretrained(ar_en_model_name)

            en_ar_model_name = "Helsinki-NLP/opus-mt-tc-big-en-ar"
            global ar_tokenizer 
            ar_tokenizer = MarianTokenizer.from_pretrained(en_ar_model_name)
            global ar_model
            ar_model = MarianMTModel.from_pretrained(en_ar_model_name)

        en_list = BT(input_list, en_tokenizer, en_model)

        ar_list = BT(en_list, ar_tokenizer, ar_model )

        samples_data["en"] = en_list
        samples_data["ar"] = ar_list

        output_suffix= '_BT.tsv'
        output_path = os.path.join(args.output_dir, smpl_set[i] + output_suffix)
        samples_data.to_csv(output_path, sep='\t', index=False, header = True, encoding='utf-8')

    print(' Back translation done successfully.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')
    # parser.add_argument('--samples_data_path', default="/data/scratch/acw564/CW_Data/samples_200/", help='Data path of sample files')
    # parser.add_argument('--output_dir', default="/data/scratch/acw564/CW_Data/BT_200/", help='BT samples output directory')
    
    parser.add_argument('--samples_data_path', default="./samples_200", help='path of samples')
    parser.add_argument('--output_dir', default="/content/drive/MyDrive/Colab Notebooks/BT", help='BT samples output directory')
    parser.add_argument('--device', default='cpu', help='Device')
	
    args = parser.parse_args()
    main(args)