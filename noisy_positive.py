import pandas as pd
import os
import numpy as np
import random
import argparse
from angle_emb import AnglE
from simcse import SimCSE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances


np.random.seed(0)
random.seed(0)



# given a dataset, compute clustering and select the closest data from each cluster
def cluster_n_select(dataset, sent_emb_model):
    ## clustering
    ### encode models using the sentence embedding model
    embeddings = sent_emb_model.encode(dataset.post.to_list())
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
        
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]

        one_cluster_tf_matrix = np.zeros( (  len(data_idx_within_i_cluster) , centers.shape[1] ) )
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row
        
        # Choose a random center among the top-3 closests from centroid
        distances = pairwise_distances(center_vec, one_cluster_tf_matrix)
        sorted_indices = np.argsort(distances)
        num = random.choice([0,1,2])
        second_closest_idx_in_one_cluster_tf_matrix = sorted_indices[0][num]
        closest_data_row_num = data_idx_within_i_cluster[second_closest_idx_in_one_cluster_tf_matrix]
        data_id = all_data[closest_data_row_num]

        closest_data.append(data_id)

    assert len(closest_data) == args.cluster_num
    
    centroid_sample = [dataset['post'][closest_data[i]] for i in m_clusters]
    
    dataset['cluster'] = m_clusters
    dataset['centroid_sample'] = centroid_sample
    
    return dataset
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_num', default=10,type=int, help='Enter the number of cluster')
    parser.add_argument('--load_dataset', default="dataset/ihc_pure",type=str, help='Enter the path of the dataset')
    parser.add_argument('--load_sent_emb_model', default="./sup-simcse-roberta-large",type=str, help='Enter the path/type of the sentence embedding model')
    args = parser.parse_args()
    print(args)

    # load implicit hate corpus dataset
    ## NOTE that we are only processing train dataset!
    train_dataset = pd.read_csv(os.path.join(args.load_dataset, 'train.tsv'), delimiter='\t', header=0)
    valid_dataset = pd.read_csv(os.path.join(args.load_dataset, 'valid.tsv'), delimiter='\t', header=0)
    test_dataset = pd.read_csv(os.path.join(args.load_dataset, 'test.tsv'), delimiter='\t', header=0)
    
    
    # load the sentence embedding model
    # print(f'LOAD_SENT_EMB_MODEL: {args.load_sent_emb_model}', flush=True)
    # AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.load_sent_emb_model)
    model = SimCSE(args.load_sent_emb_model)
    
    
    # processing each classes
    print("processing each classes...")
    ## 1) implicit_hate
    mask_implicit_hate1 = train_dataset['class'] == 'implicit_hate'
    implicit_hate1 = train_dataset.loc[mask_implicit_hate1,:]
    implicit_hate1 = implicit_hate1.reset_index(drop=True)

    mask_implicit_hate2 = valid_dataset['class'] == 'implicit_hate'
    implicit_hate2 = valid_dataset.loc[mask_implicit_hate2,:]
    implicit_hate2 = implicit_hate2.reset_index(drop=True)

    mask_implicit_hate3 = test_dataset['class'] == 'implicit_hate'
    implicit_hate3 = test_dataset.loc[mask_implicit_hate3,:]
    implicit_hate3 = implicit_hate3.reset_index(drop=True)

    ### cluster samples and and select the closest sample to the centroid per cluster
    implicit_hate1 = cluster_n_select(implicit_hate1, model)
    implicit_hate2 = cluster_n_select(implicit_hate2, model)
    implicit_hate3 = cluster_n_select(implicit_hate3, model)
    print(f"class: implicit_hate DONE")
    
    ## 2) not_hate
    mask_not_hate1 = train_dataset['class'] == 'not_hate'
    not_hate1 = train_dataset.loc[mask_not_hate1,:]
    not_hate1 = not_hate1.reset_index(drop=True)

    mask_not_hate2 = valid_dataset['class'] == 'not_hate'
    not_hate2 = valid_dataset.loc[mask_not_hate2,:]
    not_hate2 = not_hate2.reset_index(drop=True)

    mask_not_hate3 = test_dataset['class'] == 'not_hate'
    not_hate3 = test_dataset.loc[mask_not_hate3,:]
    not_hate3 = not_hate3.reset_index(drop=True)

    ### cluster samples and and select the closest sample to the centroid per cluster
    not_hate1 = cluster_n_select(not_hate1, model)
    not_hate2 = cluster_n_select(not_hate2, model)
    not_hate3 = cluster_n_select(not_hate3, model)
    print(f"class: not_hate DONE")
    
    
    # concat the samples of each class
    total_train_dataset = pd.concat([implicit_hate1, not_hate1])    
    total_train_dataset = total_train_dataset.sample(frac=1).reset_index(drop=True)

    total_valid_dataset = pd.concat([implicit_hate2, not_hate2])    
    total_valid_dataset = total_valid_dataset.sample(frac=1).reset_index(drop=True)

    total_test_dataset = pd.concat([implicit_hate3, not_hate3])    
    total_test_dataset = total_test_dataset.sample(frac=1).reset_index(drop=True)
    
    
    # save the dataset
    os.makedirs(f"dataset/forfig_ihc_c{args.cluster_num}", exist_ok=True)
    total_train_dataset.to_csv(os.path.join(f"dataset/forfig_ihc_c{args.cluster_num}", "train.tsv"), sep="\t", index=False)
    total_valid_dataset.to_csv(os.path.join(f"dataset/forfig_ihc_c{args.cluster_num}", "valid.tsv"), sep="\t", index=False)
    total_test_dataset.to_csv(os.path.join(f"dataset/forfig_ihc_c{args.cluster_num}", "test.tsv"), sep="\t", index=False)
    print(f"The processed dataset is saved at {os.path.join(f'dataset/ihc_c{args.cluster_num}', 'train.tsv')}")
    print(f"The processed dataset is saved at {os.path.join(f'dataset/ihc_c{args.cluster_num}', 'valid.tsv')}")
    print(f"The processed dataset is saved at {os.path.join(f'dataset/ihc_c{args.cluster_num}', 'test.tsv')}")