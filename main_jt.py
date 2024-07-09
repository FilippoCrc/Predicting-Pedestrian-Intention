from jaad_data import JAAD
import torch

# Load the JAAD dataset
jaad_dt = JAAD(data_path='../JAAD')
#jaad.generate_database()
#jaad_dt.get_data_stats()

data_opts = {
    'sample_type': 'beh'
}

seq_train = jaad_dt.generate_data_trajectory_sequence('train', **data_opts)  
seq_test = jaad_dt.generate_data_trajectory_sequence('test', **data_opts)  
print(type(seq_train))
print(seq_train.keys())
#print(seq_train['image'])
#print(seq_train['pid'][0][0][0])
#print(seq_train['pid'][1][0])
print(seq_train['bbox'][0][1])
