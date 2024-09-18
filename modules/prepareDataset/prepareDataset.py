from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict

def load_custom_dataset(custom_data_dir):
    print(f"Loading dataset from: {custom_data_dir}")

    ds = load_dataset("minecraft", data_dir=custom_data_dir)     
    # Split the main dataset into training and testing sets
    ds['train'] = ds['train'].train_test_split(test_size=0.3, stratify_by_column="label")
    ds_test = ds['train'].train_test_split(test_size=0.5, stratify_by_column="label")  # 30% test --> 15% valid, 15% test
    
    ds = DatasetDict({
        'train': ds['train'],
        'test': ds_test['test'],
        'valid': ds_test['train']
    })
    
    return ds