from torch.utils.data import DataLoader

from classes.dataset import LLMDataset    


def create_dataloader(
        text, tokenizer, batch_size=4, max_length=256, 
        stride=128, shuffle=True, 
        drop_last=True, num_workers=0
    ):

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

    return dataloader


