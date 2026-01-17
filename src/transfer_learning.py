import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

DATA_DIR = "data/scia_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001

# We use MTCNN to detect and align faces before feeding them to the network
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
)

# Standard transformation + FaceNet normalization
dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Load pre-trained model on VGGFace2
# classify=True adds a final Linear layer we can train
# num_classes must match your number of identities
resnet = InceptionResnetV1(
    pretrained="vggface2", classify=True, num_classes=len(dataset.classes)
).to(device)

# Freeze all layers first
for param in resnet.parameters():
    param.requires_grad = False

# Unfreeze the last fully connected layer (logits) so we can train it
for param in resnet.logits.parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss()

print("Starting training...")
for epoch in range(EPOCHS):
    resnet.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnet(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")

print("Training Finished.")

# Save model
os.makedirs("./checkpoints", exist_ok=True)
torch.save(resnet.state_dict(), "./checkpoints/facenet_transfer_learning.pth")
print("Model saved to ./checkpoints/facenet_transfer_learning.pth")
