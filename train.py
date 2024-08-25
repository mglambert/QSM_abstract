import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from red import UNet3D


# Assuming we have already defined the UNet3D class from the previous example

def train_unet3d(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training completed.")


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_channels = 1
    output_channels = 1
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = UNet3D(input_channels, output_channels)

    # Assume we have defined custom Dataset classes: TrainDataset and ValDataset
    train_dataset = TrainDataset(...)
    val_dataset = ValDataset(...)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Train the model
    train_unet3d(model, train_loader, val_loader, num_epochs, learning_rate, device)

    # Save the trained model
    torch.save(model.state_dict(), "unet3d_model.pth")
