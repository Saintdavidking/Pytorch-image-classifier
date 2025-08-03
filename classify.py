import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime
from PIL import Image
import glob

# Configuration
CONFIG = {
    'batch_size': 64,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'data_dir': './data',
    'model_save_path': 'simple_cnn_cifar10.pth',
    'num_workers': 2,  # For faster data loading
    'print_freq': 100,  # Print progress every N batches
}

def setup_device():
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        device = torch.device('cpu')
        print('Using CPU (consider using GPU for faster training)')
    return device

def get_data_loaders():
    """Create and return train and test data loaders"""
    print("Setting up data loaders...")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    try:
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=CONFIG['data_dir'], train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=CONFIG['data_dir'], train=False, download=True, transform=transform_test
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True,
            num_workers=CONFIG['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"Dataset loaded: {len(train_dataset)} training, {len(test_dataset)} test images")
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your internet connection and try again.")
        return None, None

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv + BatchNorm + ReLU + Pool layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Train the model and return training history"""
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (batch_idx + 1) % CONFIG['print_freq'] == 0:
                current_acc = 100 * correct / total
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Running Acc: {current_acc:.2f}%')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - epoch_start_time
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'  Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.1f}s')
        print('-' * 60)
    
    return train_losses, train_accuracies

def evaluate_model(model, test_loader, device, classes):
    """Evaluate the model on test data"""
    print("\nEvaluating model on test data...")
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print(f'\nTest Results:')
    print(f'Overall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})')
    
    # Per-class accuracy
    print('\nPer-class Accuracy:')
    for i in range(len(classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'  {classes[i]:>8}: {accuracy:>6.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
    
    return overall_accuracy

def plot_training_history(train_losses, train_accuracies):
    """Plot training loss and accuracy"""
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, 'g-', linewidth=2)
    plt.title('Training Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot both together
    plt.subplot(1, 3, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(train_losses, 'b-', linewidth=2, label='Loss')
    line2 = ax2.plot(train_accuracies, 'g-', linewidth=2, label='Accuracy')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy (%)', color='g')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('Training Progress', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, test_loader, device, classes, num_images=8):
    """Visualize model predictions"""
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Denormalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(num_images):
        # Denormalize image
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get prediction confidence
        confidence = probabilities[i][predicted[i]].item() * 100
        
        # Plot image
        axes[i].imshow(img)
        
        # Color code: green for correct, red for incorrect
        color = 'green' if labels[i] == predicted[i] else 'red'
        
        # Title with true label, prediction, and confidence
        title = f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]} ({confidence:.1f}%)'
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions (Green=Correct, Red=Incorrect)', fontsize=16)
    plt.tight_layout()
    plt.show()

def save_model(model, filepath):
    """Save the trained model"""
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': 'SimpleCNN',
            'num_classes': 10,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, filepath)
        print(f"\nModel saved successfully: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_trained_model(filepath, device):
    """Load a trained model from file"""
    try:
        model = SimpleCNN(num_classes=10)
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from: {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_uploaded_image(image_path):
    """Preprocess uploaded image for classification"""
    try:
        # Open and convert image
        image = Image.open(image_path)
        
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 32x32 (CIFAR-10 size)
        image = image.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Apply the same normalization as training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Transform and add batch dimension
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def classify_uploaded_images(model, device, classes, image_folder="./test_images"):
    """Classify all images in the specified folder"""
    print(f"\n{'='*60}")
    print("CLASSIFYING UPLOADED IMAGES")
    print(f"{'='*60}")
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []
    
    # Find all image files
    if os.path.exists(image_folder):
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
            image_paths.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    
    if not image_paths:
        print(f"No images found in '{image_folder}' folder.")
        print("Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP")
        print(f"\nTo test your own images:")
        print(f"1. Create a folder: mkdir {image_folder}")
        print(f"2. Copy your images to: {image_folder}/")
        print(f"3. Run the classifier again")
        return
    
    print(f"Found {len(image_paths)} images to classify")
    print(f"Image folder: {os.path.abspath(image_folder)}")
    
    # Process images
    results = []
    successful_classifications = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Preprocess image
        image_tensor, original_image = preprocess_uploaded_image(image_path)
        
        if image_tensor is not None:
            # Classify
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = classes[predicted.item()]
                confidence_score = confidence.item() * 100
                
                # Get top 3 predictions
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                top3_predictions = [(classes[idx.item()], prob.item() * 100) 
                                  for idx, prob in zip(top3_idx[0], top3_prob[0])]
                
                results.append({
                    'filename': os.path.basename(image_path),
                    'predicted_class': predicted_class,
                    'confidence': confidence_score,
                    'top3': top3_predictions,
                    'original_image': original_image
                })
                
                print(f"  Prediction: {predicted_class} ({confidence_score:.1f}% confidence)")
                print(f"  Top 3: {', '.join([f'{cls}({conf:.1f}%)' for cls, conf in top3_predictions])}")
                successful_classifications += 1
        else:
            print(f"  Failed to process image")
    
    # Display results
    if successful_classifications > 0:
        display_classification_results(results)
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully classified: {successful_classifications}/{len(image_paths)} images")
        
        # Group by predicted class
        class_counts = {}
        for result in results:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("\nPredicted classes distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} image(s)")

def display_classification_results(results):
    """Display classification results in a grid"""
    if not results:
        return
    
    # Calculate grid size
    n_images = len(results)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i < len(axes):
            # Display image
            axes[i].imshow(result['original_image'])
            
            # Create title with prediction and confidence
            title = f"{result['filename']}\n"
            title += f"Predicted: {result['predicted_class']}\n"
            title += f"Confidence: {result['confidence']:.1f}%"
            
            # Color based on confidence
            if result['confidence'] > 70:
                color = 'green'
            elif result['confidence'] > 50:
                color = 'orange'
            else:
                color = 'red'
            
            axes[i].set_title(title, color=color, fontsize=10)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Classification Results for Uploaded Images', fontsize=16)
    plt.tight_layout()
    plt.show()

def interactive_classify():
    """Interactive mode for classifying images"""
    print(f"\n{'='*60}")
    print("INTERACTIVE IMAGE CLASSIFICATION")
    print(f"{'='*60}")
    
    # Check if model exists
    model_path = CONFIG['model_save_path']
    if not os.path.exists(model_path):
        print(f"No trained model found at: {model_path}")
        print("Please train the model first by running the full training pipeline.")
        return
    
    # Setup device and load model
    device = setup_device()
    model = load_trained_model(model_path, device)
    if model is None:
        return
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    while True:
        print(f"\n{'-'*50}")
        print("OPTIONS:")
        print("1. Classify images in folder (./test_images)")
        print("2. Classify a specific image")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            classify_uploaded_images(model, device, classes)
        
        elif choice == '2':
            image_path = input("Enter the path to your image: ").strip()
            if os.path.exists(image_path):
                # Process single image
                image_tensor, original_image = preprocess_uploaded_image(image_path)
                if image_tensor is not None:
                    with torch.no_grad():
                        image_tensor = image_tensor.to(device)
                        outputs = model(image_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        predicted_class = classes[predicted.item()]
                        confidence_score = confidence.item() * 100
                        
                        # Get top 3 predictions
                        top3_prob, top3_idx = torch.topk(probabilities, 3)
                        
                        print(f"\nClassification Results:")
                        print(f"Image: {os.path.basename(image_path)}")
                        print(f"Predicted: {predicted_class} ({confidence_score:.1f}% confidence)")
                        print(f"\nTop 3 predictions:")
                        for i, (idx, prob) in enumerate(zip(top3_idx[0], top3_prob[0])):
                            print(f"  {i+1}. {classes[idx.item()]}: {prob.item()*100:.1f}%")
                        
                        # Display image
                        plt.figure(figsize=(6, 6))
                        plt.imshow(original_image)
                        plt.title(f"Predicted: {predicted_class} ({confidence_score:.1f}%)", 
                                color='green' if confidence_score > 70 else 'orange' if confidence_score > 50 else 'red')
                        plt.axis('off')
                        plt.show()
                else:
                    print("Failed to process the image.")
            else:
                print("Image file not found.")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("PYTORCH IMAGE CLASSIFIER - CIFAR-10 DATASET")
    print("=" * 70)
    print(f"Configuration: {CONFIG}")
    print("=" * 70)
    
    # Check if user wants to skip training and just classify
    if os.path.exists(CONFIG['model_save_path']):
        print(f"\nFound existing trained model: {CONFIG['model_save_path']}")
        choice = input("Do you want to:\n1. Train a new model\n2. Use existing model to classify images\n3. Both (train then classify)\nEnter choice (1-3): ").strip()
        
        if choice == '2':
            interactive_classify()
            return
        elif choice == '3':
            train_new = True
            classify_after = True
        else:
            train_new = True
            classify_after = False
    else:
        train_new = True
        classify_after = input("\nAfter training, do you want to classify your own images? (y/n): ").lower().startswith('y')
    
    if train_new:
        # Setup device
        device = setup_device()
        
        # Load data
        train_loader, test_loader = get_data_loaders()
        if train_loader is None:
            print("Failed to load data. Exiting...")
            return
        
        # CIFAR-10 class names
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        # Initialize model, loss, and optimizer
        print(f"\nInitializing model...")
        model = SimpleCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Train the model
        start_time = time.time()
        train_losses, train_accuracies = train_model(
            model, train_loader, criterion, optimizer, device, CONFIG['num_epochs']
        )
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
        
        # Evaluate the model
        test_accuracy = evaluate_model(model, test_loader, device, classes)
        
        # Plot training history
        print("\nGenerating training plots...")
        plot_training_history(train_losses, train_accuracies)
        
        # Visualize predictions
        print("Generating prediction visualizations...")
        visualize_predictions(model, test_loader, device, classes)
        
        # Save the model
        if save_model(model, CONFIG['model_save_path']):
            model_size = os.path.getsize(CONFIG['model_save_path']) / (1024**2)  # MB
            print(f"Model file size: {model_size:.1f} MB")
        
        # Final summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Dataset: CIFAR-10 (50,000 train, 10,000 test)")
        print(f"Architecture: Simple CNN with Batch Normalization")
        print(f"Training time: {training_time:.1f} seconds")
        print(f"Epochs: {CONFIG['num_epochs']}")
        print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")
        print(f"Model saved: {CONFIG['model_save_path']}")
        print("=" * 70)
    
    # Classify user images if requested
    if classify_after:
        interactive_classify()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your setup and try again.")
