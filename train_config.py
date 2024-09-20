
dataset = ["simcse_sbic_c25"]


tuning_param  = ["lambda_loss", "main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","SEED","dataset", "decay"] ## list of possible paramters to be tuned
train_batch_size = [8]
eval_batch_size = [8]
decay = [0.0]               # default value of AdamW
hidden_size = 768
nepoch = [6]    
lambda_loss = [0.75]        # scaling factor (CE vs. SCL)
temperature = [0.5]
main_learning_rate = [1e-5]

run_name = "simcse"
loss_type = ""              # only for saving file name
model_type = "bert-base-uncased"
SEED = [0]
w_aug = True                # using shared semantics as positive pairs
w_double = False
w_separate = False
w_sup = False
save = True                 # saving model parameters

param = {"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate}