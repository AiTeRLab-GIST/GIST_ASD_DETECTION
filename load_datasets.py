import conf
from datasets import load_dataset
def load_datasets():
    data_files = {
        "train": f"{conf.save_path}/{conf.df_names[0]}",
        "valid": f"{conf.save_path}/{conf.df_names[1]}",
        "test": f"{conf.save_path}/{conf.df_names[2]}"}

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    valid_dataset = dataset["valid"]
    test_dataset = dataset["test"]

    print(train_dataset)
    print(valid_dataset)
    print(test_dataset)

    label_list = train_dataset.unique('asd')
    label_list.sort()
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    return train_dataset, valid_dataset, test_dataset, label_list
