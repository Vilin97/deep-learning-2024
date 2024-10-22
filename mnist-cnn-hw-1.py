# %%
### Define the CNN model ###
from flax import nnx
import matplotlib.pyplot as plt
import optax


class CNN(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nnx.Conv(1, 6, (5, 5), rngs=rngs)
        self.conv2 = nnx.Conv(6, 16, (5, 5), rngs=rngs)
        # an affine operation: y = Wx + b
        self.fc1 = nnx.Linear(16 * 7 * 7, 120, rngs=rngs)  # 7*7 from image dimension
        self.fc2 = nnx.Linear(120, 84, rngs=rngs)
        self.fc3 = nnx.Linear(84, 10, rngs=rngs)

    def __call__(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels, 5x5 square convolution, it uses RELU activation function, and outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = nnx.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional, this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = nnx.max_pool(c1, (2, 2), strides=(2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels, 5x5 square convolution, it uses RELU activation function, and outputs a (N, 16, 14, 14) Tensor
        c3 = nnx.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional, this layer does not have any parameter, and outputs a (N, 16, 7, 7) Tensor
        s4 = nnx.max_pool(c3, (2, 2), strides=(2, 2))
        # Flatten operation: purely functional, outputs a (N, 16 * 7 * 7) Tensor
        s4 = s4.reshape(s4.shape[0], -1)
        # Fully connected layer F5: (N, 16 * 7 * 7) Tensor input, and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = nnx.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input, and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = nnx.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


# Instantiate the model.
model = CNN(rngs=nnx.Rngs(0))
# Visualize it.
nnx.display(model)

# %%
### Load the MNIST dataset ###
import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

def load_mnist_dataset(batch_size, train_steps):
    tf.random.set_seed(0)  # Set the random seed for reproducibility.

    train_ds: tf.data.Dataset = tfds.load("mnist", split="train")
    test_ds: tf.data.Dataset = tfds.load("mnist", split="test")

    train_ds = train_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )  # normalize train set
    test_ds = test_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )  # Normalize the test set.

    # Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
    train_ds = train_ds.repeat().shuffle(1024)
    # Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
    train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
    # Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    
    return train_ds, test_ds

# %%
### Define the training and evaluation steps ###
def loss_fn(model: CNN, batch):
    logits = model(batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch["label"]).mean()
    return loss, logits


@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.
    optimizer.update(grads)  # In-place updates.


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])  # In-place updates.


# %%
### Train the model ###
def train_and_evaluate(model, optimizer, metrics, train_ds, test_ds, eval_every):
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        # Run the optimization for one step and make a stateful update to the following:
        # - The train state's model parameters
        # - The optimizer state
        # - The training loss and accuracy batch metrics
        train_step(model, optimizer, metrics, batch)

        if step % eval_every == 0:  # One training epoch has passed.
            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f"train_{metric}"].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            for test_batch in test_ds.as_numpy_iterator():
                eval_step(model, metrics, test_batch)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f"test_{metric}"].append(value)
            metrics.reset()  # Reset the metrics for the next training epoch.

            print(f"[train] step: {step}, " f"loss: {metrics_history['train_loss'][-1]:.2f}, " f"accuracy: {metrics_history['train_accuracy'][-1] * 100:.2f}")
            print(f"[test]  step: {step}, " f"loss: {metrics_history['test_loss'][-1]:.2f}, " f"accuracy: {metrics_history['test_accuracy'][-1] * 100:.2f}")

    return metrics_history

#%%
# Plot the data
def plot_loss(ax, metrics_histories, eval_every, hyperparams={}):
    ax.set_title("Cross-entropy loss")

    for label, metrics_history in metrics_histories:
        dataset = "test"
        ax.plot([x * eval_every for x in range(len(metrics_history[f"{dataset}_loss"]))], metrics_history[f"{dataset}_loss"], label=label)

    # Add thin dotted grey lines
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5, label='loss=0')
    ax.set_ylim(-0.1, 3)
    ax.set_xlabel("Training step")
    ax.legend()

    # Add hyperparameters info to the plot
    hyperparams_text = "\n".join([f"{key}: {value}" for key, value in hyperparams.items()])
    ax.text(0.95, 0.95, hyperparams_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

def plot_accuracy(ax, metrics_histories, eval_every):
    ax.set_title("Accuracy, %")

    for label, metrics_history in metrics_histories:
        dataset = "test"
        ax.plot([x * eval_every for x in range(len(metrics_history[f"{dataset}_accuracy"]))], [acc * 100 for acc in metrics_history[f"{dataset}_accuracy"]], label=label)

    # Add thin dotted grey lines
    ax.axhline(y=100, color='grey', linestyle='--', linewidth=0.5, label='accuracy=100%')
    ax.set_ylim(0, 105)
    ax.set_xlabel("Training step")
    ax.legend()

# %%
# compare learning rates
batch_size = 32
train_steps = 401
eval_every = 20

learning_rates = [0.01, 0.001, 0.0001]
# Generate the data
metrics_histories = {}
for lr in learning_rates:
    train_ds, test_ds = load_mnist_dataset(batch_size, train_steps)
    model = CNN(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(lr))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    metrics_history = train_and_evaluate(model, optimizer, metrics, train_ds, test_ds, eval_every)
    metrics_histories[lr] = metrics_history
#%% 
# Plot the data
hyperparams = {'batch_size': batch_size}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plot_loss(ax1, [("lr={}".format(lr), metrics_history) for lr, metrics_history in metrics_histories.items()], eval_every, hyperparams)
plot_accuracy(ax2, [("lr={}".format(lr), metrics_history) for lr, metrics_history in metrics_histories.items()], eval_every)
plt.show()
# %%
# Compare different batch sizes and learning rates
batch_sizes = [32, 256]
learning_rates = [0.01, 0.001, 0.0001]
train_steps = 401
eval_every = 20

# Generate the data
metrics_histories = {}
for batch_size in batch_sizes:
    metrics_histories[batch_size] = {}
    for lr in learning_rates:
        train_ds, test_ds = load_mnist_dataset(batch_size, train_steps)
        model = CNN(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adamw(lr))
        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average("loss"),
        )
        metrics_history = train_and_evaluate(model, optimizer, metrics, train_ds, test_ds, eval_every)
        metrics_histories[batch_size][lr] = metrics_history
# %%
# Plot the data
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, batch_size in enumerate(batch_sizes):
    plot_loss(axes[i, 0], [(f'lr={lr}', metrics_history) for lr, metrics_history in metrics_histories[batch_size].items()], eval_every)
    plot_accuracy(axes[i, 1], [(f'lr={lr}', metrics_history) for lr, metrics_history in metrics_histories[batch_size].items()], eval_every)
    axes[i, 0].set_title(f"Cross-entropy loss (Batch size {batch_size})")
    axes[i, 1].set_title(f"Accuracy, % (Batch size {batch_size})")

plt.tight_layout()
plt.show()

#%% 
# Compare different learning rates and decays using AdaGrad
def learning_rate_fn(lr, eta):
    return lambda step : lr / (1 + (step - 1) * eta)

batch_size = 64
train_steps = 401
learning_rate_decays = [0, 0.1, 0.2]
learning_rates = [0.01, 0.001, 0.0001]
eval_every = 20

# Generate the data
metrics_histories = {}
for decay in learning_rate_decays:
    metrics_histories[decay] = {}
    for lr in learning_rates:
        train_ds, test_ds = load_mnist_dataset(batch_size, train_steps)
        model = CNN(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adagrad(learning_rate_fn(lr, decay)))
        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average("loss"),
        )
        metrics_history = train_and_evaluate(model, optimizer, metrics, train_ds, test_ds, eval_every)
        metrics_histories[decay][lr] = metrics_history

#%%
# Plot the data
hyperparams = {'batch_size': batch_size}
fig, axes = plt.subplots(len(learning_rate_decays), 2, figsize=(15, 5 * len(learning_rate_decays)))
for i, decay in enumerate(learning_rate_decays):
    plot_loss(axes[i, 0], [("lr={}".format(lr), metrics_history) for lr, metrics_history in metrics_histories[decay].items()], eval_every, hyperparams)
    plot_accuracy(axes[i, 1], [("lr={}".format(lr), metrics_history) for lr, metrics_history in metrics_histories[decay].items()], eval_every)
    axes[i, 0].set_title(f"Cross-entropy loss (Decay {decay})")
    axes[i, 1].set_title(f"Accuracy, % (Decay {decay})")

plt.tight_layout()
plt.show()

#%%
# Compare different values of β1 and β2 using Adam
batch_size = 64
train_steps = 401
learning_rate = 0.001
beta1_values = [0.4, 0.9, 0.999]
beta2_values = [0.4, 0.9, 0.999]
eval_every = 20

# Generate the data
metrics_histories = {}
for beta1 in beta1_values:
    metrics_histories[beta1] = {}
    for beta2 in beta2_values:
        train_ds, test_ds = load_mnist_dataset(batch_size, train_steps)
        model = CNN(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate, b1=beta1, b2=beta2))
        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average("loss"),
        )
        metrics_history = train_and_evaluate(model, optimizer, metrics, train_ds, test_ds, eval_every)
        metrics_histories[beta1][beta2] = metrics_history

#%%
# Plot the data
hyperparams = {'batch_size': batch_size, 'learning_rate': learning_rate}
fig, axes = plt.subplots(len(beta1_values), 2, figsize=(15, 5 * len(beta1_values)))
for i, beta1 in enumerate(beta1_values):
    plot_loss(axes[i, 0], [("β2={}".format(beta2), metrics_history) for beta2, metrics_history in metrics_histories[beta1].items()], eval_every, hyperparams)
    plot_accuracy(axes[i, 1], [("β2={}".format(beta2), metrics_history) for beta2, metrics_history in metrics_histories[beta1].items()], eval_every)
    axes[i, 0].set_title(f"Cross-entropy loss (β1={beta1})")
    axes[i, 1].set_title(f"Accuracy, % (β1={beta1})")

plt.tight_layout()
plt.show()
# %%
