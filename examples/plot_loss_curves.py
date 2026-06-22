import numpy as np
import matplotlib.pyplot as plt
from miniml.linear_model import LinearRegression

def plot_loss_curves(model, X_train, y_train, X_val, y_val, epochs=100):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.fit(X_train, y_train)
        train_loss = model.loss(X_train, y_train)
        val_loss = model.loss(X_val, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')


plot_loss_curves(model=LinearRegression(),
                    X_train=np.random.rand(100, 1),
                    y_train=np.random.rand(100, 1),
                    X_val=np.random.rand(20, 1),
                    y_val=np.random.rand(20, 1),
                    epochs=50)
plt.legend()
plt.title('Loss Curves')
plt.show()

