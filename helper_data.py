import torchvision
from torch.utils.data import DataLoader
def get_dataloaders_celeba(batch_size, num_workers, 
                           train_transforms = None,
                           test_transforms = None,
                           download = True):
    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()

    if test_transforms is None:    
        test_transforms = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CelebA(root='./data',
                                                split = 'train',
                                                transform = train_transforms,
                                                download = download
                                                )
    
    valid_dataset = torchvision.datasets.CelebA(root = './data',
                                               split = 'valid',
                                               transform = test_transforms,
                                               download = download
                                               )

    test_dataset = torchvision.datasets.CelebA(root = './data',
                                               split = 'test',
                                               transform = test_transforms,
                                               download = download
                                               ) 
    
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = batch_size,
                              num_workers = num_workers,
                              shuffle = True)
    
    valid_loader = DataLoader(dataset = valid_dataset,
                              batch_size = batch_size,
                              num_workers = num_workers,
                              shuffle = False)
    
    test_loader = DataLoader(dataset = test_dataset,
                              batch_size = batch_size,
                              num_workers = num_workers,
                              shuffle = False)
    
    return train_loader, valid_loader, test_loader