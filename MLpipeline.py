import torch
import MlTools as mlt

model = mlt.EyeMLP()
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

X_train, X_test, y_train, y_test = mlt.load_data("face_data1761048351.csv")
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# Define loss function and optimizer and MAE loss
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)

# loss du predictor constant = 0.5/0.5 est de 0.0833 pour la MSE et de 0.25 pour la MAE
# Training loop
num_epochs = 12500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")


# Save the trained model
torch.save(model.state_dict(), "eye_mlp_model.pth")
