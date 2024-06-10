import h5py
import numpy as np

def merge_hdf5_files(file1, file2, output_file):
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2, h5py.File(output_file, 'w') as f_out:
        # Merge datasets
        for name in f1:
            data1 = f1[name][:]
            if name in f2:
                data2 = f2[name][:]
                merged_data = np.concatenate((data1, data2), axis=0)
            else:
                merged_data = data1

            f_out.create_dataset(name, data=merged_data)

        for name in f2:
            if name not in f1:
                data2 = f2[name][:]
                f_out.create_dataset(name, data=data2)

        # Copy attributes if needed
        for attr in f1.attrs:
            f_out.attrs[attr] = f1.attrs[attr]
        for attr in f2.attrs:
            if attr not in f1.attrs:
                f_out.attrs[attr] = f2.attrs[attr]

# Usage
datasets = ["halfcheetah", "hopper", "walker"]
ratios = [0.05, 0.1]
for dataset in datasets:
    for keep_ratio in ratios:
        if dataset == "walker":
            dataset0 = "walker2d"
        else:
            dataset0 = dataset
        file1 = f'./datasets/{dataset}_mixed.hdf5'
        file2 = f'./generated_datasets/{dataset0}-medium-replay-v0_reconstructed_{keep_ratio}.hdf5'
        output_file = f'./generated_datasets/{dataset0}-medium-replay-v0_Reconstructed_{keep_ratio}.hdf5'
        merge_hdf5_files(file1, file2, output_file)