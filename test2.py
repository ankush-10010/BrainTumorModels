import pickle
with open(r"C:\Users\ankus\Downloads\cifar-10-batches-py\data_batch_1","rb") as file:
    dict=pickle.load(file,encoding='bytes')
print(dict.attribute)