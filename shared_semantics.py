import pandas as pd
import os
import numpy as np
import random
import argparse
from angle_emb import AnglE
from simcse import SimCSE
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances


np.random.seed(0)
random.seed(0)



# given a dataset, compute clustering and select the closest data from each cluster
def cluster_n_select(dataset, sent_emb_model, input_col, is_not_hate):
    ## clustering
    ### encode models using the sentence embedding model
    embeddings = sent_emb_model.encode(dataset[input_col].to_list())

    # # if CUDA out of memory while using AnglE model, use this instead.
    # dummy = dataset.post.to_list()
    # block_size = 500
    # dummies = [dummy[i:i+block_size] for i in range(0,len(dummy), block_size)]
    # embeddingss = []
    # for dummy in dummies:
    #     embeddings = sent_emb_model.encode(dummy, to_numpy=True)
    #     embeddingss.append(embeddings)
    # embeddings = np.concatenate(tuple(embeddingss))

    all_data = [i for i in range(embeddings.shape[0])]

    
    ### kmeans clustering
    kmeans = KMeans(n_clusters=args.cluster_num, random_state=0, n_init="auto").fit(embeddings)
    m_clusters = kmeans.labels_.tolist()
    centers = np.array(kmeans.cluster_centers_)
    
    ## select the closest data from the cluster
    closest_data = []
    for i in range(args.cluster_num):
        center_vec = centers[i]
        center_vec = center_vec.reshape(1, -1) 
        
        data_idx_within_i_cluster = [idx for idx, clu_num in enumerate(m_clusters) if clu_num == i]

        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , centers.shape[1]))
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row
        
        closest, _ = pairwise_distances_argmin_min(center_vec, one_cluster_tf_matrix)
        closest_idx_in_one_cluster_tf_matrix = closest[0]
        closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
        data_id = all_data[closest_data_row_num]

        closest_data.append(data_id)
    assert len(closest_data) == args.cluster_num
    
    centroid_sample = [dataset[input_col][closest_data[i]] for i in m_clusters]

    ## distinguish the non-hate label
    if is_not_hate:
        m_clusters = [clu_num + 1000 for clu_num in m_clusters]
    
    dataset['cluster'] = m_clusters
    dataset['centroid_sample'] = centroid_sample
    
    return dataset

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_num', default=10,type=int, help='Enter the number of cluster')
    parser.add_argument('--load_dataset', default="ihc_pure",type=str, help='Enter the path of the dataset')
    parser.add_argument('--load_sent_emb_model', default="simcse",type=str, help='Enter the path/type of the sentence embedding model')
    args = parser.parse_args()

    # load raw dataset
    if args.load_dataset == "ihc_pure":
        train_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'train.tsv'), delimiter='\t', header=0)
        valid_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'valid.tsv'), delimiter='\t', header=0)
        test_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'test.tsv'), delimiter='\t', header=0)
        input_col = 'post'
        class_col = 'class'
        hate_class = "implicit_hate"
        not_hate_class = "not_hate"
    elif args.load_dataset == "sbic":
        train_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'train.csv'), delimiter=',', header=0)
        valid_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'dev.csv'), delimiter=',', header=0)
        test_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'test.csv'), delimiter=',', header=0)
        input_col = 'post'
        class_col = 'offensiveLABEL'
        hate_class = "offensive"
        not_hate_class = "not_offensive"
    elif args.load_dataset == "dynahate":
        train_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'train.csv'), delimiter=',', header=0)
        valid_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'dev.csv'), delimiter=',', header=0)
        test_dataset = pd.read_csv(os.path.join('raw_dataset', args.load_dataset, 'test.csv'), delimiter=',', header=0)
        input_col = 'text'
        class_col = 'label'
        hate_class = "hate"
        not_hate_class = "nothate"
    else:
        raise NotImplementedError
    
    
    # load the sentence embedding model
    print(f'LOAD_SENT_EMB_MODEL: {args.load_sent_emb_model}', flush=True)
    if args.load_sent_emb_model == "simcse":
        model = SimCSE("./sup-simcse-roberta-large")
    elif args.load_sent_emb_model == "angle":
        model = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', pooling_strategy='cls_avg').cuda()
    elif args.load_sent_emb_model == "sbert":
        model = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        raise NotImplementedError
    
    
    # processing each classes
    print("processing each classes...")

    ## 1) hate class
    mask_implicit_hate1 = train_dataset[class_col] == hate_class
    implicit_hate1 = train_dataset.loc[mask_implicit_hate1,:]
    implicit_hate1 = implicit_hate1.reset_index(drop=True)

    mask_implicit_hate2 = valid_dataset[class_col] == hate_class
    implicit_hate2 = valid_dataset.loc[mask_implicit_hate2,:]
    implicit_hate2 = implicit_hate2.reset_index(drop=True) 

    mask_implicit_hate3 = test_dataset[class_col] == hate_class
    implicit_hate3 = test_dataset.loc[mask_implicit_hate3,:]
    implicit_hate3 = implicit_hate3.reset_index(drop=True)

    ### cluster samples and and select the closest sample to the centroid per cluster
    implicit_hate1 = cluster_n_select(implicit_hate1, model, input_col, is_not_hate=False)
    implicit_hate2 = cluster_n_select(implicit_hate2, model, input_col, is_not_hate=False)
    implicit_hate3 = cluster_n_select(implicit_hate3, model, input_col, is_not_hate=False)
    print(f"class: implicit_hate DONE")
    
    ## 2) not_hate
    mask_not_hate1 = train_dataset[class_col] == not_hate_class
    not_hate1 = train_dataset.loc[mask_not_hate1,:]
    not_hate1 = not_hate1.reset_index(drop=True)

    mask_not_hate2 = valid_dataset[class_col] == not_hate_class
    not_hate2 = valid_dataset.loc[mask_not_hate2,:]
    not_hate2 = not_hate2.reset_index(drop=True)

    mask_not_hate3 = test_dataset[class_col] == not_hate_class
    not_hate3 = test_dataset.loc[mask_not_hate3,:]
    not_hate3 = not_hate3.reset_index(drop=True)

    ### cluster samples and and select the closest sample to the centroid per cluster
    not_hate1 = cluster_n_select(not_hate1, model, input_col, is_not_hate=True)
    not_hate2 = cluster_n_select(not_hate2, model, input_col, is_not_hate=True)
    not_hate3 = cluster_n_select(not_hate3, model, input_col, is_not_hate=True)
    print(f"class: not_hate DONE")
    
    
    # concat the samples of each class
    total_train_dataset = pd.concat([implicit_hate1, not_hate1])    
    total_train_dataset = total_train_dataset.sample(frac=1).reset_index(drop=True)

    total_valid_dataset = pd.concat([implicit_hate2, not_hate2])    
    total_valid_dataset = total_valid_dataset.sample(frac=1).reset_index(drop=True)

    total_test_dataset = pd.concat([implicit_hate3, not_hate3])    
    total_test_dataset = total_test_dataset.sample(frac=1).reset_index(drop=True)
    
    
    # save the dataset
    os.makedirs(f"clustered_dataset/{args.load_sent_emb_model}/{args.load_dataset}_c{args.cluster_num}", exist_ok=True)
    if args.load_dataset == "ihc_pure":
        total_train_dataset.to_csv(os.path.join(f"clustered_dataset/{args.load_sent_emb_model}/{args.load_dataset}_c{args.cluster_num}", "train.tsv"), sep="\t", index=False)
        total_valid_dataset.to_csv(os.path.join(f"clustered_dataset/{args.load_sent_emb_model}/{args.load_dataset}_c{args.cluster_num}", "valid.tsv"), sep="\t", index=False)
        total_test_dataset.to_csv(os.path.join(f"clustered_dataset/{args.load_sent_emb_model}/{args.load_dataset}_c{args.cluster_num}", "test.tsv"), sep="\t", index=False)
    else:
        total_train_dataset.to_csv(os.path.join(f"clustered_dataset/{args.load_sent_emb_model}/{args.load_dataset}_c{args.cluster_num}", "train.csv"), sep=",", index=False)
        total_valid_dataset.to_csv(os.path.join(f"clustered_dataset/{args.load_sent_emb_model}/{args.load_dataset}_c{args.cluster_num}", "valid.csv"), sep=",", index=False)
        total_test_dataset.to_csv(os.path.join(f"clustered_dataset/{args.load_sent_emb_model}/{args.load_dataset}_c{args.cluster_num}", "test.csv"), sep=",", index=False)