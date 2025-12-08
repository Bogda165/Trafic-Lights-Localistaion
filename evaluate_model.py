import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import os
import multiprocessing

# Set multiprocessing start method for macOS
try:
    multiprocessing.freeze_support()
except Exception:
    pass

# Always try to set the start method to 'spawn' for macOS compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError as e:
        print(f"Failed to set multiprocessing start method: {e}")
else:
    print("Multiprocessing start method is already set to 'spawn'")

# Setup device (M1 GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class (copied from train_model.ipynb)
class SingleTrafficLightDataset(Dataset):
    def __init__(self, csv_file, target_size=(128, 128), is_train=True, cache_size=100):
        self.annotations = pd.read_csv(csv_file)
        self.target_size = target_size
        self.cache_size = cache_size
        self.cache = {}  # Simple LRU cache for images

        # Filter to only images that exist
        self.annotations = self.annotations[self.annotations['file_path'].apply(os.path.exists)]

        # Take first traffic light per image (simplified for study)
        self.annotations = self.annotations.groupby('file_path').first().reset_index()

        self.class_to_idx = {'go': 0, 'stop': 1, 'warning': 2, 'stopLeft': 3, 
                            'goForward': 4, 'goLeft': 5, 'warningLeft': 6}

        # Simple transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = row['file_path']

        # Check if image is in cache
        if img_path in self.cache:
            image = self.cache[img_path]
        else:
            # Load and compress to small size (128x128 for speed)
            image = Image.open(img_path).convert('RGB').resize(self.target_size, Image.BILINEAR)

            # Add to cache if not full
            if len(self.cache) < self.cache_size:
                self.cache[img_path] = image
            elif self.cache_size > 0:
                # Simple LRU: remove a random item (first one in dict)
                self.cache.pop(next(iter(self.cache)))
                self.cache[img_path] = image

        # Get single traffic light annotation
        target = torch.tensor([
            row['norm_center_x'],
            row['norm_center_y'],
            row['norm_width'],
            row['norm_height'],
            self.class_to_idx.get(row['Annotation tag'], 0)
        ], dtype=torch.float32)

        return self.transform(image), target

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

# Model definition (copied from train_model.ipynb)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 128x128x3 -> 64x64x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64x64x32 -> 32x32x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 16x16x128 -> 8x8x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # x, y, w, h
            nn.Sigmoid()  # Normalize to [0, 1]
        )

        # Classification head
        self.class_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)  # 7 classes
        )

    def forward(self, x):
        # Extract features
        features = self.features(x)

        # Predict bbox and class separately
        bbox = self.bbox_head(features)
        class_scores = self.class_head(features)

        return bbox, class_scores

# Loss function (copied from train_model.ipynb)
class DetectionLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_class=1.0):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_class = lambda_class
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_bbox, pred_class, targets):
        # Bounding box loss (MSE for coordinates)
        bbox_loss = self.mse(pred_bbox, targets[:, :4])

        # Classification loss (CrossEntropy)
        class_loss = self.ce(pred_class, targets[:, 4].long())

        # Combined loss
        total_loss = self.lambda_coord * bbox_loss + self.lambda_class * class_loss

        return total_loss, bbox_loss, class_loss

# Validation function (copied from train_model.ipynb)
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_bbox_loss = 0
    total_class_loss = 0
    
    # For metrics calculation
    all_pred_classes = []
    all_true_classes = []
    all_pred_bboxes = []
    all_true_bboxes = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device)

            pred_bbox, pred_class = model(images)
            loss, bbox_loss, class_loss = criterion(pred_bbox, pred_class, targets)

            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            total_class_loss += class_loss.item()
            
            # Store predictions and ground truth for metrics calculation
            pred_classes = torch.argmax(pred_class, dim=1).cpu().numpy()
            true_classes = targets[:, 4].long().cpu().numpy()
            
            all_pred_classes.extend(pred_classes)
            all_true_classes.extend(true_classes)
            all_pred_bboxes.append(pred_bbox.cpu().numpy())
            all_true_bboxes.append(targets[:, :4].cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(loader)
    avg_bbox_loss = total_bbox_loss / len(loader)
    avg_class_loss = total_class_loss / len(loader)
    
    # Convert lists to numpy arrays
    all_pred_classes = np.array(all_pred_classes)
    all_true_classes = np.array(all_true_classes)
    all_pred_bboxes = np.concatenate(all_pred_bboxes)
    all_true_bboxes = np.concatenate(all_true_bboxes)
    
    # Calculate classification accuracy
    accuracy = np.mean(all_pred_classes == all_true_classes)
    
    # Calculate IoU (Intersection over Union) for bounding boxes
    def calculate_iou(box1, box2):
        # Convert from center format to corner format
        box1_x1 = box1[0] - box1[2]/2
        box1_y1 = box1[1] - box1[3]/2
        box1_x2 = box1[0] + box1[2]/2
        box1_y2 = box1[1] + box1[3]/2
        
        box2_x1 = box2[0] - box2[2]/2
        box2_y1 = box2[1] - box2[3]/2
        box2_x2 = box2[0] + box2[2]/2
        box2_y2 = box2[1] + box2[3]/2
        
        # Calculate intersection area
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        return iou
    
    # Calculate IoU for each prediction
    ious = []
    for i in range(len(all_pred_bboxes)):
        iou = calculate_iou(all_pred_bboxes[i], all_true_bboxes[i])
        ious.append(iou)
    
    avg_iou = np.mean(ious)
    
    return {
        'loss': avg_loss,
        'bbox_loss': avg_bbox_loss,
        'class_loss': avg_class_loss,
        'accuracy': accuracy,
        'iou': avg_iou
    }

# Visualize predictions function (copied from train_model.ipynb)
def visualize_predictions(model, dataset, num_samples=5):
    model.eval()
    class_names = ['go', 'stop', 'warning', 'stopLeft', 'goForward', 'goLeft', 'warningLeft']
    colors = ['green', 'red', 'yellow', 'orange', 'cyan', 'blue', 'magenta']

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, target = dataset[idx]

            # Load original image (128x128)
            row = dataset.annotations.iloc[idx]
            img_path = row['file_path']
            orig_img = Image.open(img_path).convert('RGB').resize((128, 128))

            # Predict
            pred_bbox, pred_class = model(image.unsqueeze(0).to(device))
            pred_bbox = pred_bbox[0].cpu().numpy()
            pred_class = torch.softmax(pred_class[0], dim=-1).cpu().numpy()

            # Ground truth
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')

            cx, cy, w, h, cls = target
            x1, y1 = (cx - w/2) * 128, (cy - h/2) * 128
            rect = patches.Rectangle((x1, y1), w*128, h*128, linewidth=2, 
                                    edgecolor=colors[int(cls)], facecolor='none')
            axes[i, 0].add_patch(rect)
            axes[i, 0].text(x1, y1-3, class_names[int(cls)], 
                          color=colors[int(cls)], fontsize=10, weight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            # Prediction
            axes[i, 1].imshow(orig_img)
            axes[i, 1].set_title('Prediction', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')

            pred_cls = np.argmax(pred_class)
            confidence = np.max(pred_class)

            cx, cy, w, h = pred_bbox
            x1, y1 = (cx - w/2) * 128, (cy - h/2) * 128
            rect = patches.Rectangle((x1, y1), w*128, h*128, linewidth=2,
                                    edgecolor=colors[pred_cls], facecolor='none', linestyle='--')
            axes[i, 1].add_patch(rect)
            axes[i, 1].text(x1, y1-3, f'{class_names[pred_cls]} ({confidence:.2f})',
                          color=colors[pred_cls], fontsize=10, weight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig('validation_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("🔍 EVALUATING BEST MODEL ON VALIDATION DATASET")
    print("="*60)
    
    # Load validation dataset
    val_dataset = SingleTrafficLightDataset('val_annotations.csv', target_size=(128, 128), is_train=False)
    
    # Configure DataLoader
    BATCH_SIZE = 64
    NUM_WORKERS = 0 if torch.backends.mps.is_available() else min(6, os.cpu_count() - 1)
    use_pin_memory = False
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=use_pin_memory, 
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    print(f"✅ Validation dataset loaded: {len(val_dataset)} images")
    
    # Create model
    model = SimpleCNN(num_classes=7).to(device)
    
    # Load best model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model from epoch {checkpoint['epoch']+1}")
    
    # Create loss function
    criterion = DetectionLoss()
    
    # Evaluate model
    print("\n📊 Evaluating model on validation dataset...")
    metrics = validate(model, val_loader, criterion)
    
    # Print metrics
    print("\n" + "="*60)
    print("📊 VALIDATION METRICS")
    print("="*60)
    print(f"Total Loss:      {metrics['loss']:.4f}")
    print(f"Bounding Box Loss: {metrics['bbox_loss']:.4f}")
    print(f"Classification Loss: {metrics['class_loss']:.4f}")
    print(f"Classification Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average IoU:     {metrics['iou']:.4f}")
    print("="*60)
    
    # Visualize predictions
    print("\n🖼️ Visualizing predictions...")
    visualize_predictions(model, val_dataset, num_samples=5)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()