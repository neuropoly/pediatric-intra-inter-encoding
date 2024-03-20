import matplotlib.pyplot as plt
import numpy as np

def plot_distribution_idx_per_subject(indices_by_subject, save_path):
    # Count the number of indices per subject
    num_indices_per_subject = [len(indices) for indices in indices_by_subject.values()]

    # Create a histogram plot
    plt.hist(num_indices_per_subject, bins=100, alpha=0.75, edgecolor='k')
    plt.title("Distribution of Number of Indices per Subject")
    plt.xlabel("Number of Indices")
    plt.ylabel("Frequency")

    # Show the plot
    plt.savefig(f'{save_path}/distribution_idx_per_subject.png')

def plot_example_intra_inter_colormap(x_train, y_train, save_path):
    # Generate a colorbar to go from -1 to 1
    cmap = plt.cm.get_cmap('coolwarm')
    norm = plt.Normalize(-1, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(2, 7, figsize=(14, 4))

    for i in range(14):
        row = i // 7
        col = i % 7
        image = x_train[i, :, :, :]
        label = y_train[i]
        axes[row, col].imshow(np.squeeze(image[:, :, 32]), cmap='coolwarm', norm=norm)
        if label == 1:
            axes[row, col].set_title("1: intra")
        elif label == 0:
            axes[row, col].set_title("0: inter")
        axes[row, col].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    plt.savefig(f'{save_path}/example_intra_inter_coolwarm_cmap.png')


def plot_example_intra_inter(x_train, y_train, save_path):
    fig, axes = plt.subplots(2, 7, figsize=(14, 4))

    for i in range(28, 42):
        row = (i-28) // 7
        col = (i-28) % 7
        image = x_train[i, :, :, :]
        label = y_train[i]
        axes[row, col].imshow(np.squeeze(image[:, :, 32]), cmap='gray')
        if label == 1:
            axes[row, col].set_title("1: intra")
        elif label == 0:
            axes[row, col].set_title("0: inter")
        axes[row, col].axis('off')

    plt.savefig(f'{save_path}/example_intra_inter_gray_cmap.png')

def plot_log_jacobian_per_age_interval(df_279, save_path):
    # Filter data based on the 'group' column
    group_inter = df_279[df_279['group'] == 'inter']
    group_intra = df_279[df_279['group'] == 'intra']

    # Scatter plot for 'inter' group in red color and 'intra' group in blue color
    plt.scatter(group_inter['age_interval'], group_inter['avg_abs_log_jacobian'], c='red', label='Inter')
    plt.scatter(group_intra['age_interval'], group_intra['avg_abs_log_jacobian'], c='blue', label='Intra')

    plt.title('Scatter Plot of avg_abs_log_jacobian vs age_interval')
    plt.xlabel('age_interval')
    plt.ylabel('avg_abs_log_jacobian')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/log_jacobian_per_age_interval.png')

def visualize_model_performance(model, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax = ax.ravel()

    for i, metric in enumerate(["acc", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric))
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
        ax[i].legend(["train", "val"])  
    # Save the plot as a PNG file at the specified location
    plt.savefig(f'{save_path}/performance_graph.png')
    # Show the plot in your Python environment
    plt.show()