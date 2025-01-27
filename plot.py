import matplotlib.pyplot as plt

def plot_values(
        epoches_seen, examples_seen, train_values, val_values, fig_name, label='loss'
):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epoches_seen, train_values, label=f'Training {label}')
    ax1.plot(
        epoches_seen, val_values, linestyle='-.',
        label='fValidation {label}'
    )
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f'{fig_name}_{label}-plot.pdf')
    plt.show()