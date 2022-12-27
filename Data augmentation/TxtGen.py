\
import argparse
import pandas as pd
import numpy as np
import torch
import os
import textwrap
from arabert.preprocess import ArabertPreprocessor
from transformers import pipeline, GPT2TokenizerFast
from arabert.aragpt2.grover.modeling_gpt2 import GPT2LMHeadModel
import sys
sys.path.append('.')

def main(args):

    print(" Start generate text from samples")
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
        samples_data = samples_data.head(5)
        if counter ==1:
            model_name = "aubmindlab/aragpt2-medium"
            arabert_processor = ArabertPreprocessor(model_name=model_name)
            aragpt2_pipeline = pipeline("text-generation",model=model_name,device=device)
        text_prep_list = []
        for i in range(len(samples_data)):
            text_prep = arabert_processor.preprocess(samples_data['tweet_text'][i])
            text_prep_list.append(text_prep)
        generate_text_list = []
        for i in range(len(samples_data)):
            gen_text = aragpt2_pipeline(text_prep_list[i],
                        pad_token_id=0, 
                        num_beams=5,
                        max_length=200,
                        top_p=0.75,
                        repetition_penalty = 3.0,
                        no_repeat_ngram_size = 3)[0]['generated_text']
            generate_text_list.append(gen_text)
        samples_data["Gen_text"] = generate_text_list
        output_suffix= '_Gen.tsv'
        output_path = os.path.join(args.output_dir, smpl_set[i] + output_suffix)
        samples_data.to_csv(output_path, sep='\t', index=False, header = True, encoding='utf-8')

    print(' text generated done successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')
    parser.add_argument('--samples_data_path', default="./samples_200", help='path of samples')
    parser.add_argument('--output_dir', default="./Gen_200", help='text generated samples output directory')
    parser.add_argument('--device', default='cpu', help='Device')
	
    args = parser.parse_args()
    main(args)