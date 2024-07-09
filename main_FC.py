from jaad_data import JAAD

# Load the JAAD dataset
jaad = JAAD(data_path='../JAAD')
#jaad.generate_database()
#jaad.get_data_stats()


data_opts = {
    'sample_type': 'beh'
}
seq_train = jaad.generate_data_trajectory_sequence('train', **data_opts)  
seq_test = jaad.generate_data_trajectory_sequence('test', **data_opts)
# print (type(seq_train))
# print (seq_train.keys())
# print(seq_train['bbox'])

#id del video è solo il nuimero 

# # get sequences
#         beh_seq_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
#         beh_seq_val = None
#         # Uncomment the line below to use validation set
#         # beh_seq_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])
#         beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts']) ## load_dataset

#         # get the model
#         method_class = action_prediction(configs['model_opts']['model'])(**configs['net_opts'])

#         # train and save the model
#         saved_files_path = method_class.train(beh_seq_train, beh_seq_val, **configs['train_opts'],
#                                               model_opts=configs['model_opts'])
#         # test and evaluate the model
#         acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)

#         # save the results
#         data = {}
#         data['results'] = {}
#         data['results']['acc'] = float(acc)
#         data['results']['auc'] = float(auc)
#         data['results']['f1'] = float(f1)
#         data['results']['precision'] = float(precision)
#         data['results']['recall'] = float(recall)
#         write_to_yaml(yaml_path=os.path.join(saved_files_path, 'results.yaml'), data=data)

#         data = configs
#         write_to_yaml(yaml_path=os.path.join(saved_files_path, 'configs.yaml'), data=data)

#         print('Model saved to {}'.format(saved_files_path))