# utils.py:
import torch
from torch.utils.data import DataLoader
import torchvision
from customDataset import CableRobotDataset

def save_checkpoint(state, filename='model.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    torch.load_state_dict(checkpoint['state_dict'])
    torch.load_state_dict(checkpoint['optimizer'])

    return checkpoint.get('epoch', 0)

def get_loaders(train_image_dir, train_mask_dir, train_transform, val_image_dir, val_mask_dir, val_transform, batch_size, pin_memory=True, num_workers=4):
    
    train_data = CableRobotDataset(image_dir=train_image_dir, mask_dir= train_mask_dir, transform = train_transform)
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

    val_data = CableRobotDataset(image_dir=val_image_dir, mask_dir=val_mask_dir, transform=val_transform)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    return train_loader, val_loader

def check_accuracy(model, loader, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in (loader):
        
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct = (preds == y).sum()
            num_pixels = torch.numel(preds)
            dice_score = (2*(preds*y).sum())/((preds+y).sum() + 1e-8)

    print(f"Accuracy = {(num_correct/num_pixels)*100:.2f}")
    print(f"Dice score = {(dice_score)/len(loader)}")
    model.train()

def save_preds_as_images(loader, model, folder='saved_images/', device='cuda'):

    for idx, (x,y) in enumerate(loader):

        x = x.to(device)
        model.eval()
        with torch.no_grad():
        
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            torchvision.utils.save_image(tensor = preds, fp = f"{folder}/pred_{idx}.png")

        torchvision.utils.save_image(tensor=y.unsqueeze(1), fp = f"{folder}{idx}.png")

    model.train()
    