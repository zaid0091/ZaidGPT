import torch
import torch.nn as nn


# ==========================================
# 1. Training data
# ==========================================

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
    [10.0]
])


# ==========================================
# 2. Model
# ==========================================

model = nn.Linear(1, 1)


# ==========================================
# 3. Loss function
# ==========================================

loss_function = nn.MSELoss()


# ==========================================
# 4. Optimizer
# ==========================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# ==========================================
# 5. Training loop
# ==========================================

for epoch in range(1000):

    # Forward pass
    prediction = model(x)

    # Calculate loss
    loss = loss_function(prediction, y)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    # Display progress
    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss.item():.6f}"
        )


# ==========================================
# 6. Test the trained model
# ==========================================

test = torch.tensor([[10.0]])

prediction = model(test)

print("\nPrediction for 10:")
print(prediction.item())


# ==========================================
# 7. Show learned parameters
# ==========================================

print("\nLearned weight:")
print(model.weight.item())

print("\nLearned bias:")
print(model.bias.item())

torch.save(
    model.state_dict(),
    "day1_model.pt"
)

print("\nModel saved to day1_model.pt")