"""
Weight Stacking Ensemble Trainer

This script trains a neural network meta-learner that learns optimal weights
for combining predictions from the 4 base models (ELA, Pixel, Frequency, Xception).

The meta-learner takes 4 inputs (0-100% AI confidence from each model) and 
outputs a single probability, effectively learning the best way to combine the models.
"""

import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


class StackingMetaLearner(nn.Module):
    """
    Neural network meta-learner for weight stacking ensemble.
    
    Takes 4 model predictions (ELA, Pixel, Frequency, Xception) and learns
    the optimal way to combine them into a single final prediction.
    """
    def __init__(self, input_dim=4, hidden_dims=[16, 8]):
        super().__init__()
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer (sigmoid for binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 4) with model predictions [ELA, Pixel, Freq, Xcep]
        Returns:
            Tensor of shape (batch_size, 1) with combined prediction (0-1 probability)
        """
        return self.network(x)


def generate_synthetic_training_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic training data for stacking meta-learner.
    
    Creates realistic prediction patterns:
    - AI-generated images: models generally agree (high agreement)
    - Real images: models generally agree (low agreement)
    - Some conflicting predictions (diversity)
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    X_list = []
    y_list = []
    
    # Half AI-generated, half real
    for i in range(n_samples):
        is_ai = i < n_samples // 2
        
        if is_ai:
            # AI images: models predict high confidence for AI
            # Add some realistic variation
            base_confidence = np.random.uniform(70, 99)
            noise = np.random.normal(0, 8, 4)
            predictions = np.clip(base_confidence + noise, 0, 100)
            label = 1.0
        else:
            # Real images: models predict low confidence for AI
            base_confidence = np.random.uniform(1, 30)
            noise = np.random.normal(0, 8, 4)
            predictions = np.clip(base_confidence + noise, 0, 100)
            label = 0.0
        
        # Normalize to [0, 1]
        predictions_normalized = predictions / 100.0
        
        X_list.append(predictions_normalized)
        y_list.append([label])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    logger.info(f"Generated {n_samples} synthetic training samples")
    logger.info(f"  AI samples: {np.sum(y):.0f}")
    logger.info(f"  Real samples: {n_samples - np.sum(y):.0f}")
    logger.info(f"  X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


def train_stacking_model(X_train=None, y_train=None, epochs=100, batch_size=32, 
                         learning_rate=0.001, validation_split=0.2):
    """
    Train the stacking meta-learner on provided data (or synthetic data if none provided).
    
    Args:
        X_train: Training input data (n_samples, 4) with model predictions
        y_train: Training labels (n_samples, 1) with ground truth (0=real, 1=AI)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation
    
    Returns:
        Trained model, training history, and normalized input stats
    """
    # Generate synthetic data if not provided
    if X_train is None or y_train is None:
        logger.info("No training data provided - generating synthetic data...")
        X_train, y_train = generate_synthetic_training_data(n_samples=1000)
    else:
        logger.info(f"Using provided training data: X shape {X_train.shape}, y shape {y_train.shape}")
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).float()
    
    # Split train/validation
    n_samples = len(X_tensor)
    n_train = int(n_samples * (1 - validation_split))
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train_split = X_tensor[train_indices]
    y_train_split = y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]
    
    logger.info(f"Train split: {len(X_train_split)} samples")
    logger.info(f"Validation split: {len(X_val)} samples")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_split, y_train_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = StackingMetaLearner(input_dim=4, hidden_dims=[16, 8])
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    logger.info("\nStarting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            # Forward
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * X_batch.size(0)
            predictions = (outputs >= 0.5).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_predictions = (val_outputs >= 0.5).float()
            val_acc = (val_predictions == y_val).sum().item() / len(y_val)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    logger.info(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    
    # Calculate normalization stats (for input preprocessing)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0  # Avoid division by zero
    
    normalization_stats = {
        'mean': X_mean.tolist(),
        'std': X_std.tolist()
    }
    
    return model, history, normalization_stats


def save_stacking_model(model, history, normalization_stats, output_path=None):
    """
    Save trained stacking meta-learner to pickle file.
    """
    if output_path is None:
        output_path = BASE_DIR / "model" / "stacking_meta_learner.pkl"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 4,
            'hidden_dims': [16, 8],
            'class': 'StackingMetaLearner'
        },
        'normalization_stats': normalization_stats,
        'training_history': history,
        'model_type': 'NeuralNetworkStackingEnsemble'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"Saved stacking model to {output_path}")
    return output_path


def load_stacking_model(model_path=None):
    """
    Load trained stacking meta-learner from pickle file.
    """
    if model_path is None:
        model_path = BASE_DIR / "model" / "stacking_meta_learner.pkl"
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Stacking model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Reconstruct model
    model = StackingMetaLearner(
        input_dim=checkpoint['model_config']['input_dim'],
        hidden_dims=checkpoint['model_config']['hidden_dims']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded stacking model from {model_path}")
    return model, checkpoint


if __name__ == '__main__':
    # Train stacking model
    logger.info("=" * 60)
    logger.info("Weight Stacking Ensemble - Meta-Learner Training")
    logger.info("=" * 60)
    
    # Generate and train on synthetic data
    model, history, norm_stats = train_stacking_model(
        epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Save model
    model_path = save_stacking_model(model, history, norm_stats)
    
    # Verify loading
    loaded_model, checkpoint = load_stacking_model(model_path)
    logger.info(f"\n✓ Model saved and verified at {model_path}")
    logger.info(f"  Normalization stats: {checkpoint['normalization_stats']}")
    
    # Test inference on sample data
    logger.info("\nTesting inference on sample predictions...")
    test_samples = torch.tensor([
        [0.99, 0.99, 0.99, 0.97],  # All models agree: AI
        [0.47, 0.01, 0.00, 0.53],  # All models agree: REAL (mixed votes)
        [0.70, 0.50, 0.60, 0.65],  # Mixed predictions
    ]).float()
    
    with torch.no_grad():
        test_outputs = loaded_model(test_samples)
    
    logger.info(f"Sample test outputs (0-1 probability):")
    for i, output in enumerate(test_outputs):
        logger.info(f"  Sample {i+1}: {output.item():.4f} → {'AI' if output.item() >= 0.5 else 'REAL'}")
