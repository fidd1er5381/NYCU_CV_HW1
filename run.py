import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def conv3x3(in_planes, out_planes, stride=1):
    """Create a 3x3 convolution layer."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """Create a 1x1 convolution layer."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class SEModule(nn.Module):
    """Squeeze-and-Excitation Module."""

    def __init__(self, channels, reduction=16):
        """
        Initialize Squeeze-and-Excitation Module.

        Args:
            channels (int): Number of input channels
            reduction (int, optional): Reduction ratio. Defaults to 16.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        Forward pass of the Squeeze-and-Excitation Module.

        Args:
            input_tensor (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Scaled input tensor
        """
        x = self.avg_pool(input_tensor)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input_tensor * x


class Bottleneck(nn.Module):
    """ResNet Bottleneck Block."""

    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None,
        use_se=False, se_reduction=32
    ):
        """
        Initialize Bottleneck Block.

        Args:
            inplanes (int): Number of input channels
            planes (int): Number of output channels
            stride (int, optional): Convolution stride. Defaults to 1.
            downsample (nn.Module, optional): Downsample layer. Defaults to None.
            use_se (bool, optional): Use Squeeze-and-Excitation. Defaults to False.
            se_reduction (int, optional): SE reduction ratio. Defaults to 32.
        """
        super().__init__()
        width = int(planes)

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se

        if use_se:
            self.se = SEModule(
                planes * self.expansion, reduction=se_reduction
            )

    def forward(self, x):
        """
        Forward pass of the Bottleneck Block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetRS(nn.Module):
    """ResNet-RS Model."""

    def __init__(
        self, block, layers, num_classes=1000,
        use_se=True, se_reduction=32
    ):
        """
        Initialize ResNetRS model.

        Args:
            block (nn.Module): Block type (e.g., Bottleneck)
            layers (list): Number of blocks in each layer
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            use_se (bool, optional): Use Squeeze-and-Excitation. Defaults to True.
            se_reduction (int, optional): SE reduction ratio. Defaults to 32.
        """
        super().__init__()
        self.inplanes = 64
        self.use_se = use_se
        self.se_reduction = se_reduction

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=9, stride=2, padding=4, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a layer with multiple blocks.

        Args:
            block (nn.Module): Block type
            planes (int): Number of output channels
            blocks (int): Number of blocks in the layer
            stride (int, optional): Convolution stride. Defaults to 1.

        Returns:
            nn.Sequential: Layer of blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample,
                use_se=self.use_se, se_reduction=self.se_reduction
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes,
                    use_se=self.use_se, se_reduction=self.se_reduction
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNetRS model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnetrs50(num_classes=1000, **kwargs):
    """
    Create a ResNetRS50 model.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 1000.

    Returns:
        ResNetRS: ResNetRS50 model
    """
    model = ResNetRS(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
        use_se=False, se_reduction=32, **kwargs
    )
    return model


class PredictionDataset(torch.utils.data.Dataset):
    """Dataset for prediction."""

    def __init__(self, img_dir, transform=None):
        """
        Initialize prediction dataset.

        Args:
            img_dir (str): Directory containing images
            transform (callable, optional): Image transformations. Defaults to None.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        """
        Get dataset length.

        Returns:
            int: Number of images
        """
        return len(self.img_names)

    def __getitem__(self, idx):
        """
        Get image and filename.

        Args:
            idx (int): Index of the image

        Returns:
            tuple: (transformed image, image filename)
        """
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


def get_transform(is_train=True):
    """
    Get image transformations.

    Args:
        is_train (bool, optional): Whether transformations are for training. Defaults to True.

    Returns:
        transforms.Compose: Image transformations
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.25)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(293),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        ])


def train_model(
    model, train_loader, val_loader, criterion, optimizer,
    scheduler, num_epochs, device
):
    """
    Train the model.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        scheduler (lr_scheduler): Learning rate scheduler
        num_epochs (int): Number of training epochs
        device (torch.device): Device to train on

    Returns:
        tuple: Lists of train/val losses and accuracies
    """
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in tqdm.tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(epoch_acc.item())
        val_accs.append(val_acc.item())
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_resnetrs.pth')

    print(f'Best val Acc: {best_acc:.4f}')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    return train_losses, val_losses, train_accs, val_accs


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training and validation curves.

    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_accs (list): Training accuracies
        val_accs (list): Validation accuracies
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.close()


def plot_confusion_matrix(
    model, loader, classes, normalize=False,
    cmap='Blues', tick_every=10, device=None
):
    """
    Plot confusion matrix.

    Args:
        model (nn.Module): Trained model
        loader (DataLoader): Data loader
        classes (list): List of class names
        normalize (bool, optional): Normalize confusion matrix. Defaults to False.
        cmap (str, optional): Colormap. Defaults to 'Blues'.
        tick_every (int, optional): Tick frequency. Defaults to 10.
        device (torch.device, optional): Device to run on. Defaults to None.

    Returns:
        numpy.ndarray: Confusion matrix
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm, annot=False, cmap=cmap,
        xticklabels=False, yticklabels=False,
        square=True, cbar=True, linewidths=0
    )

    tick_positions = list(range(0, len(classes), tick_every))
    tick_labels = [str(classes[i]) for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels, rotation=0)

    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return cm


def evaluate(model, loader, criterion, device):
    """
    Evaluate model performance.

    Args:
        model (nn.Module): Model to evaluate
        loader (DataLoader): Data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to evaluate on

    Returns:
        tuple: Loss and accuracy
    """
    model.eval()
    running_loss, running_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / len(loader.dataset)
    acc = running_corrects.double() / len(loader.dataset)
    return loss, acc


def predict(model, test_loader, train_dataset, device):
    """
    Generate predictions for test dataset.

    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        train_dataset (datasets.ImageFolder): Training dataset
        device (torch.device): Device to predict on
    """
    model.eval()
    predictions, img_names = [], []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for filename, pred in zip(filenames, preds.cpu().numpy()):
                pred_label = train_dataset.classes[pred]
                predictions.append(pred_label)
                img_names.append(filename)

    df = pd.DataFrame({
        'image_name': img_names,
        'pred_label': predictions,
    })
    df.to_excel('output_resnetrs.xlsx', index=False)
    print("Predict Done!")


def main():
    """Main function to run the training pipeline."""
    # Configuration
    data_dir = './data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    batch_size = 128
    num_epochs = 100
    learning_rate = 0.0001
    num_classes = len(os.listdir(train_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data preparation
    train_transform = get_transform(is_train=True)
    val_test_transform = get_transform(is_train=False)

    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    val_dataset = datasets.ImageFolder(val_dir, val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model preparation
    model = resnetrs50(num_classes=num_classes)

    try:
        state_dict = torch.load('./best_model_resnetrs.pth')
        model_dict = model.state_dict()
        filtered_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print("Successfully loaded partial pre-trained weights")
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}")

    model = model.to(device)

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.05
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Test dataset preparation
    test_dataset = PredictionDataset(test_dir, transform=val_test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Training
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, num_epochs, device
    )

    # Load best model
    model.load_state_dict(torch.load('best_model_resnetrs.pth'))

    # Confusion matrix
    classes = train_dataset.classes
    plot_confusion_matrix(model, val_loader, classes, device=device)

    # Prediction
    predict(model, test_loader, train_dataset, device)


if __name__ == '__main__':
    main()
