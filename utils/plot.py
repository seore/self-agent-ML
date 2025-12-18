import matplotlib.pyplot as plt
import numpy as np


def plot(scores, out_path, window=100):
    plt.figure(figsize=(10, 5))

    scores_np = np.array(scores)

    # rolling mean
    if len(scores_np) >= window:
        rolling = np.convolve(
            scores_np,
            np.ones(window) / window,
            mode="valid"
        )
        plt.plot(
            range(window - 1, len(scores_np)),
            rolling,
            label=f"Rolling Avg ({window})",
            linewidth=2
        )

    # raw scores (light + transparent)
    plt.plot(
        scores_np,
        alpha=0.25,
        linewidth=0.7,
        label="Level Score"
    )

    plt.xlabel("Level")
    plt.ylabel("Score")
    plt.title("Training Performance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
