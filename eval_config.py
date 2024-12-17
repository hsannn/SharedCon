tuning_param  = ["dataset", "load_dir"]
dataset = ["ihc_pure","sbic","dynahate"] # dataset for evaluation

# saved model location
load_dir = ['./model_location1', "./model_location2"]


train_batch_size = 8
eval_batch_size = 8
hidden_size = 768
model_type = "bert-base-uncased"
SEED = 0

param = {"dataset":dataset,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"dataset":dataset,"SEED":SEED,"model_type":model_type, "load_dir":load_dir}

