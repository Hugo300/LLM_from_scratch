import torch
from torch.utils.data import Dataset, DataLoader

class LLMDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    

def create_dataloader(
        text, batch_size=4, max_length=256, 
        stride=128, shuffle=True, 
        drop_last=True, num_workers=0
    ):

    # get tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # create the dataset
    dataset = LLMDataset(text, tokenizer, max_length, stride)

    # create the data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader, tokenizer.n_vocab