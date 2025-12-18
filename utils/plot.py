import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, out_path: str):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.figure()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.legend(["Score", "Mean Score"])
    plt.pause(0.001)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
