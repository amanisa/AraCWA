
import argparse
import os
import pandas as pd
import json
import torch.nn.functional as F
import collections
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
#% matplotlib inline
import seaborn as sns
import logging
import sys
sys.path.append('.')
from utils import _compute_average_precision, _compute_reciprocal_rank, _compute_precisions
from utils import print_thresholded_metric, print_single_metric
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score, recall_score


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 30]

def train_val(device, model, train_dataloader ,validation_dataloader, num_train_epochs):
	optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )
	epochs =  num_train_epochs
	total_steps = len(train_dataloader) * epochs
	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = 0, 
												num_training_steps = total_steps)
	seed_val = 42
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	training_stats = []
	total_t0 = time.time()
	for epoch_i in range(0, epochs):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')
		t0 = time.time()
		total_train_loss = 0
		model.train()
		for step, batch in enumerate(train_dataloader):
			if step % 40 == 0 and not step == 0:
				elapsed = format_time(time.time() - t0)
				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
			model.zero_grad()        
			result = model(b_input_ids, 
						token_type_ids=None, 
						attention_mask=b_input_mask, 
						labels=b_labels,
						return_dict=True)
			loss = result.loss
			logits = result.logits
			total_train_loss += loss.item()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()
		avg_train_loss = total_train_loss / len(train_dataloader)            
		training_time = format_time(time.time() - t0)
		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epcoh took: {:}".format(training_time))
		print("")
		print("Running Validation...")
		t0 = time.time()
		model.eval()
		total_eval_accuracy = 0
		total_eval_loss = 0
		nb_eval_steps = 0
		for batch in validation_dataloader:
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
			with torch.no_grad():        
				result = model(b_input_ids, 
							token_type_ids=None, 
							attention_mask=b_input_mask,
							labels=b_labels,
							return_dict=True)
			loss = result.loss
			logits = result.logits
			total_eval_loss += loss.item()
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()
			total_eval_accuracy += flat_accuracy(logits, label_ids)
		avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
		print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
		avg_val_loss = total_eval_loss / len(validation_dataloader)
		validation_time = format_time(time.time() - t0)
		print("  Validation Loss: {0:.2f}".format(avg_val_loss))
		print("  Validation took: {:}".format(validation_time))
		training_stats.append(
			{
				'epoch': epoch_i + 1,
				'Training Loss': avg_train_loss,
				'Valid. Loss': avg_val_loss,
				'Valid. Accur.': avg_val_accuracy,
				'Training Time': training_time,
				'Validation Time': validation_time
			}
		)
	print("")
	print("Training complete!")
	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
	learning_curve(training_stats)



def learning_curve(training_stats):
	pd.set_option('precision', 2)
	df_stats = pd.DataFrame(data=training_stats)
	df_stats = df_stats.set_index('epoch')
	Var1 = range(1, 26)
	sns.set(style='darkgrid')
	sns.set(font_scale=1.5)
	plt.rcParams["figure.figsize"] = (12,6)
	plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
	plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
	plt.title("Training & Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.xticks(Var1)
	plt.show()

def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def split_data(train_df):
	print("length of training set before splitting: ",len(train_df.index))
	train_data = train_df.sample(frac=0.9, random_state=42) 
	val_data = train_df.drop(train_data.index)
	print("length of training set after splitting: ",len(train_data.index))
	print("length of validation set after splitting: ",len(val_data.index))
	return train_data, val_data

def tokenize(tokenizer, df):
	sentences = df.tweet_text.values
	labels = df.claim_worthiness.values
	input_ids = []
	attention_masks = []
	for sent in sentences:
		encoded_dict = tokenizer.encode_plus(
							sent,                      
							add_special_tokens = True, 
							max_length = 64,           
							pad_to_max_length = True,
							return_attention_mask = True,   
							return_tensors = 'pt',     
					)
		input_ids.append(encoded_dict['input_ids'])
		attention_masks.append(encoded_dict['attention_mask'])
	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)
	dataset = TensorDataset(input_ids, attention_masks, labels)
	return dataset


def dataset_loader(train_dataset, val_dataset, test_dataset, train_batch_size, test_batch_size):
	train_dataloader = DataLoader(
				train_dataset,  
				sampler = RandomSampler(train_dataset), 
				batch_size = train_batch_size 
			)
	validation_dataloader = DataLoader(
				val_dataset, 
				sampler = SequentialSampler(val_dataset), 
				batch_size = train_batch_size 
			)
	test_dataloader = DataLoader(
				test_dataset, 
				sampler = SequentialSampler(test_dataset), 
				batch_size = test_batch_size 
			)
	return train_dataloader ,validation_dataloader, test_dataloader


def Extract_pred(softmax_probs):
    return [item[1] for item in softmax_probs]

def predict(device, model, test_dataloader):
	model.eval()
	predictions = []
	true_labels = []
	softmax_probs = []
	max_probs = []
	test_ids = []
	count = 0
	pred_list = []
	for batch in test_dataloader:
		batch = tuple(t.to(device) for t in batch)
		c_input_ids = batch[0].to(device)
		c_input_mask = batch[1].to(device)
		with torch.no_grad():
			outputs = model(c_input_ids, 
						token_type_ids=None, 
						attention_mask=c_input_mask,
						return_dict=True)
		logits = outputs[0]
		logits = F.softmax(logits, dim=0)
		probs = logits.detach().cpu().tolist()
		pred_labels = torch.argmax(logits, dim=1)
		pred_labels = pred_labels.detach().cpu().numpy()
		for prob in probs:
			softmax_probs.append(prob)
		for pred in pred_labels:
			predictions.append(pred)
	co_score_list = Extract_pred(softmax_probs)
	print('Done prediction')
	return co_score_list, predictions

def save_pred(df_test ,co_score_list,  co_score_path, predictions, pred_path, true_labels, counter):
	score_df = pd.DataFrame()
	score_df["scores"] = co_score_list
	output = df_test[["topic_id", "tweet_id"]]
	output["run_id"] = counter
	output["score"] = score_df
	output["score"] = output["score"].round(4)
	output = output[['topic_id','tweet_id','score', 'run_id']]
	output.head()
	output.to_csv(co_score_path, sep='\t', index=False, header = False, encoding='utf-8')
	pred_df = pd.DataFrame()
	pred_df["predictions"] = predictions
	output2 = output
	output2["predictions"] = pred_df
	output2["true_labels"] = true_labels
	output2 = output2[['run_id','topic_id','tweet_id', 'true_labels', 'predictions', 'score' ]]
	output2.to_csv(pred_path, sep='\t', index=False, header = True, encoding='utf-8')


def _read_gold_and_pred(gold_fpath, pred_fpath):

    logging.info("Reading gold predictions from file {}".format(gold_fpath))
    gold_labels = {}
    with open(gold_fpath, encoding='utf-8') as gold_f:
        for line_res in gold_f:
            (topic_id, tweet_id, tweet_text, claim, check_worthiness) = line_res.strip().split('\t')  
            if topic_id == 'topic_id':
                continue
            label = check_worthiness
            gold_labels[int(tweet_id)] = int(label)
    logging.info('Reading predicted ranking order from file {}'.format(pred_fpath))
    line_score = []
    with open(pred_fpath) as pred_f:
        for line in pred_f:
            topic_id, tweet_id, score , run_id= line.split('\t')
            tweet_id = int(tweet_id.strip())
            score = float(score.strip())
            if tweet_id not in gold_labels:
                logging.error('No such tweet_id: {} in gold file!'.format(tweet_id))
                quit()
            line_score.append((tweet_id, score))
    if len(set(gold_labels).difference([tup[0] for tup in line_score])) != 0:
        logging.error('The predictions do not match the lines from the gold file - missing or extra line_no')
        raise ValueError('The predictions do not match the lines from the gold file - missing or extra line_no')
    return gold_labels, line_score


def evaluate_pred(gold_fpath, pred_fpath, thresholds=None):
    gold_labels, line_score = _read_gold_and_pred(gold_fpath, pred_fpath)
    ranked_lines = [t[0] for t in sorted(line_score, key=lambda x: x[1], reverse=True)]
    if thresholds is None or len(thresholds) == 0:
        thresholds = MAIN_THRESHOLDS + [len(ranked_lines)]
    precisions = _compute_precisions(gold_labels, ranked_lines, len(ranked_lines))
    avg_precision = _compute_average_precision(gold_labels, ranked_lines)
    reciprocal_rank = _compute_reciprocal_rank(gold_labels, ranked_lines)
    num_relevant = len({k for k, v in gold_labels.items() if v == 1})
    return thresholds, precisions, avg_precision, reciprocal_rank, num_relevant	

def show_confusion_matrix(confusion_matrix):
	hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
	hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
	hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
	plt.ylabel('True labels')
	plt.xlabel('Predicted labels');

def get_score(test_path, co_score_path, true_labels , predictions, counter, topic, result_path):

	logging.info(f"Started evaluating results for topic-{topic} ...")
	precision = precision_score(true_labels, predictions).round(2)
	recall = recall_score(true_labels, predictions).round(2)
	f1 = f1_score(true_labels, predictions).round(2)
	print('Precision:', precision)
	print('Recall:', recall)
	print('f1:', f1)
	thresholds, precisions, avg_precision, reciprocal_rank, num_relevant = evaluate_pred(test_path, co_score_path)
	overall_precisions = [0.0] * len(MAIN_THRESHOLDS)
	threshold_precisions = [precisions[th - 1] for th in MAIN_THRESHOLDS]
	r_precision = precisions[num_relevant - 1]
	for idx in range(0, len(MAIN_THRESHOLDS)):
		overall_precisions[idx] = threshold_precisions[idx]
	mean_r_precision = r_precision
	mean_avg_precision = avg_precision
	mean_reciprocal_rank = reciprocal_rank
	filename = os.path.basename(co_score_path)
	logging.info('{:=^120}'.format(' RESULTS for {} '.format(filename)))
	print_single_metric('AVERAGE PRECISION:', avg_precision)
	print_single_metric('RECIPROCAL RANK:', reciprocal_rank)
	print_single_metric('R-PRECISION (R={}):'.format(num_relevant), r_precision)
	print_thresholded_metric('PRECISION@N:', MAIN_THRESHOLDS, threshold_precisions)
	if counter == 1:
		global results 
		results = collections.defaultdict(list)
	results['Topic ID'].append(topic)
	results['Precision'].append(precision)
	results['Recall'].append(recall)
	results['F1'].append(f1)
	results['MAP'].append(round(avg_precision,4))
	results['RR'].append(round(reciprocal_rank,4))
	results['(R=)'].append(num_relevant)
	results['R-PRECISION'].append(round(r_precision,4))
	results['P@1'].append(round(threshold_precisions[0],4))
	results['P@3'].append(round(threshold_precisions[1],4))
	results['P@5'].append(round(threshold_precisions[2],4))
	results['P@10'].append(round(threshold_precisions[3],4))
	results['P@20'].append(round(threshold_precisions[4],4))
	results['P@30'].append(round(threshold_precisions[5],4))
	if counter == 14:
		results_df = pd.DataFrame(results)
		tsv_file = "AraCW.tsv"
		excel_file = "AraCW.xlsx"
		results_df.to_csv(os.path.join(result_path + tsv_file), sep="\t", index=False)
		datatoexcel = pd.ExcelWriter(os.path.join(result_path + excel_file))
		results_df.to_excel(datatoexcel)	
		datatoexcel.save()
		print(' Results are exported to excel and tsv files')
		



def save_model(model, tokenizer, output_dir, topic):
	model_suffix = '_model'
	output_dir = os.path.join(output_dir, topic+model_suffix)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	print("Saving model to %s" % output_dir)
	model_to_save = model.module if hasattr(model, 'module') else model  
	model_to_save.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)




def main(args):

	print(" Start AraCW Experiment")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	topics = ["CT20-AR-01" ,"CT20-AR-02", "CT20-AR-05", "CT20-AR-08", "CT20-AR-10",
    "CT20-AR-12", "CT20-AR-14", "CT20-AR-19", "CT20-AR-23", "CT20-AR-27", "CT20-AR-30",
    "Covid-19", "CT21-AR-01", "CT21-AR-02"]

	training_sets = ["X_CT20-AR-01_part2", "X_CT20-AR-02_part2" ,"X_CT20-AR-05_part2", "X_CT20-AR-08_part2", "X_CT20-AR-10_part2", 
	"X_CT20-AR-12_part2", "X_CT20-AR-14_part2", "X_CT20-AR-19_part2", "X_CT20-AR-23_part2", "X_CT20-AR-27_part2", "X_CT20-AR-30_part2",
	"X_Covid-19", "X_CT21-AR-01_part2", "X_CT21-AR-02_part2"]

	test_sets = ["CT20-AR-01_gold_200" ,"CT20-AR-02_gold_200", "CT20-AR-05_gold_200", "CT20-AR-08_gold_200", "CT20-AR-10_gold_200",
     "CT20-AR-12_gold_200", "CT20-AR-14_gold_200", "CT20-AR-19_gold_200", "CT20-AR-23_gold_200", "CT20-AR-27_gold_200", "CT20-AR-30_gold_200",
     "Covid-19_gold_200", "CT21-AR-01_gold_200", "CT21-AR-02_gold_200"]

	suffix = '.tsv'
	counter = 0
	for i in range(0,14):
		counter = counter +1
		print ("Itr. no.:", counter)
		print ("topic name:", topics[i])
		train_path = os.path.join(args.train_data_path,training_sets[i]+suffix)
		test_path = os.path.join(args.test_data_path,test_sets[i]+suffix)
		co_score_suffix= '_scores_AraCW.tsv'
		co_score_path = os.path.join(args.pred_data_path, topics[i] + co_score_suffix)
		pred_suffix= '_pred_AraCW.tsv'
		pred_path = os.path.join(args.pred_data_path, topics[i] + pred_suffix)
		train_data = pd.read_csv(train_path, delimiter = "\t" ,encoding='utf-8')
		test_data = pd.read_csv(test_path, delimiter = "\t" ,encoding='utf-8')
		train_data, val_data = split_data(train_data)
		print('Loading BERT tokenizer...')
		tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02', do_lower_case=True)
		train_dataset = tokenize (tokenizer, train_data)
		val_dataset = tokenize (tokenizer, val_data)
		test_dataset = tokenize (tokenizer, test_data)
		train_dataloader ,validation_dataloader , test_dataloader= dataset_loader(train_dataset, val_dataset, test_dataset ,train_batch_size=args.train_batch_size, test_batch_size= args.test_batch_size)

		model = BertForSequenceClassification.from_pretrained(
			"aubmindlab/bert-base-arabertv02",
			num_labels = 2,    
			output_attentions = False, 
			output_hidden_states = False, 
		)
		model.cuda()
		train_val(device, model, train_dataloader ,validation_dataloader, num_train_epochs=args.num_train_epochs)
		co_score_list, predictions =predict(device, model, test_dataloader)
		true_labels = test_data.claim_worthiness.values
		save_pred(test_data, co_score_list, co_score_path, predictions, pred_path, true_labels, counter= counter)
		get_score(test_path, co_score_path, true_labels , predictions, counter, topic=topics[i], result_path= args.result_dir)
		save_model(model, tokenizer, output_dir=args.output_dir, topic=topics[i])
		print(' The experiment done successfully.')



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')
	parser.add_argument('--train_data_path', default="./train/", help='path of training set')
	parser.add_argument('--test_data_path', default="./test/", help='path of test set')
	parser.add_argument('--pred_data_path', default="./Pred/", help='path for prediction')
	parser.add_argument('--output_dir', default="./Saved_models/", help='Model output directory')
	parser.add_argument('--result_dir', default="./results/", help='Results output directory')
	parser.add_argument('--model_name', default='aubmindlab/bert-base-arabertv02', help='Model name')
	parser.add_argument('--model_type', default='aubmindlab/bert-base-arabertv02', help='Model type')
	parser.add_argument('--max_length', default=64, type=int, help='Max length of text')
	parser.add_argument('--num_train_epochs', default=25, type=int, help='Num of epoch during training')
	parser.add_argument('--train_batch_size', default=16, type=int, help='Batch size')
	parser.add_argument('--test_batch_size', default=32, type=int, help='Batch size')
	parser.add_argument('--device', default='cpu', help='Device')
	
	args = parser.parse_args()

	main(args)


