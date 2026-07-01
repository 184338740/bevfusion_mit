import pickle

path = "/mnt/bevfusion_mit_xmy/tools/tyjt_data_infos_v04/tyjt_dbinfos_train.pkl"

with open(path, 'rb') as f:
    data = pickle.load(f)
print(data.keys())