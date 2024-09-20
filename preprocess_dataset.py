import pandas as pd
import pickle
import argparse
import numpy as np
import random

from transformers import AutoTokenizer

import numpy as np
import os

# Credits https://github.com/varsha33/LCL_loss
np.random.seed(0)
random.seed(0)



def preprocess_data(sent_emb_model, dataset, tokenizer_type):
	os.makedirs("preprocessed_data", exist_ok=True)

	if "ihc" in dataset:
		class2int = {'not_hate':0 ,'implicit_hate': 1}
		data_home = f"clustered_dataset/{sent_emb_model}/{dataset}/"
		data_dict = {}

		for datatype in ["train","valid","test"]:

			datafile = data_home + datatype + ".tsv"
			data = pd.read_csv(datafile, sep='\t') 
			# data.iteritems = data.items

			label1, label2, post = [],[],[]

			for i, one_class in enumerate(data["class"]):
				label1.append(class2int[one_class])
			
			# label을 class가 아닌 cluster가 되게 한다.
			cluster_value = []
			post_value = []
			for (columnName, columnData) in data.iteritems():
				if columnName == 'cluster':
					cluster_value = columnData.values
				elif columnName == 'post':
					post_value = columnData.values
			for one_cluster, one_post in zip(cluster_value, post_value):
				label2.append(one_cluster)
				post.append(one_post)


			if datatype == "train":
				augmented_post = []
				for i,one_class in enumerate(data["cluster"]):
					augmented_post.append(data["centroid_sample"][i])

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented =tokenizer.batch_encode_plus(augmented_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
				combined_prompt = [list(i) for i in zip(post,augmented_post)]
				combined_label = [list(i) for i in zip(label1,label1)]
				combined_cluster_label = [list(i) for i in zip(label2,label2)]

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["cluster_label"] = combined_cluster_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data
				

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post = tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label1
				processed_data["cluster_label"] = label2
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		
		with open(f"./preprocessed_data/preprocessed_{sent_emb_model}_{dataset}.pkl", 'wb') as f:
			pickle.dump(data_dict, f)
		print(f'The tokenized data is saved at ./preprocessed_data/preprocessed_{sent_emb_model}_{dataset}.pkl')




	elif "sbic" in dataset:
		class2int = {'not_offensive':0 ,'offensive': 1}
		data_dict = {}
		data_home = f"clustered_dataset/{sent_emb_model}/{dataset}/"

		for datatype in ["train","valid","test"]:
			datafile = data_home + datatype + ".csv"
			data = pd.read_csv(datafile, sep=',')
			label1,label2,post = [],[],[]

			for i,one_class in enumerate(data["offensiveLABEL"]):
				label1.append(class2int[one_class])

			cluster_value = []
			post_value = []

			for (columnName, columnData) in data.iteritems():
				if columnName == 'cluster':
					cluster_value = columnData.values
				elif columnName == 'post':
					post_value = columnData.values


			for one_cluster, one_post in zip(cluster_value, post_value):
				label2.append(one_cluster)
				post.append(one_post)

			if datatype == "train":
				augmented_post = []
				for i, one_class in enumerate(data["cluster"]):
					augmented_post.append(data["centroid_sample"][i])

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented =tokenizer.batch_encode_plus(augmented_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
				combined_prompt = [list(i) for i in zip(post,augmented_post)]
				combined_label = [list(i) for i in zip(label1,label1)]
				combined_cluster_label = [list(i) for i in zip(label2,label2)]

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["cluster_label"] = combined_cluster_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label1
				processed_data["cluster_label"] = label2
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		with open(f"./preprocessed_data/preprocessed_{sent_emb_model}_{dataset}.pkl", 'wb') as f:
			pickle.dump(data_dict, f)
		print(f'The tokenized data is saved at ./preprocessed_data/preprocessed_{sent_emb_model}_{dataset}.pkl')




	elif "dynahate" in dataset:
		class2int = {'nothate':0 ,'hate': 1}
		data_dict = {}
		data_home = f"clustered_dataset/{sent_emb_model}/{dataset}/"

		for datatype in ["train","valid","test"]:
			datafile = data_home + datatype + ".csv"
			data = pd.read_csv(datafile, sep=',') 
			data.iteritems = data.items		# AttributeError: 'DataFrame' object has no attribute 'iteritems'

			label1,label2,post = [],[],[]

			for i,one_class in enumerate(data["label"]):
				label1.append(class2int[one_class])

			cluster_value = []
			post_value = []

			for (columnName, columnData) in data.iteritems():
				if columnName == 'cluster':
					cluster_value = columnData.values
				elif columnName == 'text':
					post_value = columnData.values

			for one_cluster, one_post in zip(cluster_value, post_value):
				label2.append(one_cluster)
				post.append(one_post)

			if datatype == "train":
				augmented_post = []
				for i, one_class in enumerate(data["cluster"]):
					augmented_post.append(data["centroid_sample"][i])

				## in the case of DynaHate using 'aug_sent'
				# for i, one_aug_sent in enumerate(data["aug_sent1_of_text"]):
				# 	augmented_post.append(one_aug_sent)

				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids
				tokenized_post_augmented =tokenizer.batch_encode_plus(augmented_post).input_ids

				tokenized_combined_prompt = [list(i) for i in zip(tokenized_post,tokenized_post_augmented)]
				combined_prompt = [list(i) for i in zip(post,augmented_post)]
				combined_label = [list(i) for i in zip(label1,label1)]
				combined_cluster_label = [list(i) for i in zip(label2,label2)]

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_combined_prompt
				processed_data["label"] = combined_label
				processed_data["cluster_label"] = combined_cluster_label
				processed_data["post"] = combined_prompt

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

			else:
				print("Tokenizing data")
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
				tokenized_post =tokenizer.batch_encode_plus(post).input_ids

				processed_data = {}
				processed_data["tokenized_post"] = tokenized_post
				processed_data["label"] = label1
				processed_data["cluster_label"] = label2
				processed_data["post"] = post

				processed_data = pd.DataFrame.from_dict(processed_data)
				data_dict[datatype] = processed_data

		with open(f"./preprocessed_data/preprocessed_{sent_emb_model}_{dataset}.pkl", 'wb') as f:
			pickle.dump(data_dict, f)
		print(f'The tokenized data is saved at ./preprocessed_data/preprocessed_{sent_emb_model}_{dataset}.pkl')


	else:
		raise NotImplementedError
	

	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Enter tokenizer type')

	parser.add_argument('-m', default="simcse", type=str, help='Enter sentence embedding model')
	parser.add_argument('-d', default="ihc_pure_c10",type=str, help='Enter dataset')
	parser.add_argument('-t', default="bert-base-uncased",type=str, help='Enter tokenizer type')
	args = parser.parse_args()

	preprocess_data(args.m, args.d, args.t)