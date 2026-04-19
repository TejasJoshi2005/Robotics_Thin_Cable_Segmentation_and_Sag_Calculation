import torch
import os
import torch.nn as nn
import albumentations as A
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNET 
from utils import save_checkpoint, load_checkpoint, get_loaders, save_preds_as_images, check_accuracy

# hyperparamters: 
learning_rate = 5e-5
num_epochs = 100
batch_size = 2
num_workers = 2
pin_memory = True
image_height = 512
image_width = 512
load_model = False
train_img_dir = "Split_Dataset/train/images/"
train_mask_dir = "Split_Dataset/train/masks/"
val_img_dir = "Split_Dataset/val/images"
val_mask_dir = "Split_Dataset/val/masks"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):

        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        with torch.autocast(device_type='cuda'):
            scores = model(data)
            loss = loss_fn(scores, targets) + dice_loss(scores, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred*target).sum()
    return 1 - ((2*intersection + smooth)/ (pred.sum() + target.sum() + smooth))


        
def main():
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], is_check_shapes=False
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], is_check_shapes=False
    )



    # model = UNET(in_channels=3, out_channels=1).to(device)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    train_loader, val_loader = get_loaders(train_img_dir, train_mask_dir, train_transform, val_img_dir, val_mask_dir, val_transform, batch_size, pin_memory, num_workers)

    if load_model:
        load_checkpoint('model.pth.tar')

        
    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        print(f"Processing epoch {epoch+1}")
        train_loop(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {"state_dict" : model.state_dict(),
                      "optimizer" : optimizer.state_dict(), 
                      "epoch" : epoch}
        
        save_checkpoint(state=checkpoint)

        check_accuracy(loader=val_loader, model=model, device=device)

        save_folder = "saved_images"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        save_preds_as_images(loader=val_loader, model=model, folder=save_folder, device=device)

        # import matplotlib.pyplot as plt

        # data, target = next(iter(val_loader))
        # model.eval()

        # with torch.no_grad():
        #     preds = torch.sigmoid(model(data.to(device)))
        #     preds = (preds > 0.5).float()

        # idx = 0
        # plt.figure(figsize=(12,4))
        # plt.subplot(1,3,1); plt.imshow(data[idx].cpu().permute(1,2,0)); plt.title("Input")
        # plt.subplot(1,3,2); plt.imshow(target[idx].cpu(), cmap='gray'); plt.title("Ground Truth")
        # plt.subplot(1,3,3); plt.imshow(preds[idx].cpu().squeeze(), cmap="gray"); plt.title("Prediction")
        # plt.show()

if __name__ == "__main__":
    main()
