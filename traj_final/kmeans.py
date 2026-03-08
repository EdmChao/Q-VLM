import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def main():
    vocab_size = 8192
    ori_traj = np.load('your_path_to_npy')
    L, HORIZON, DIM = ori_traj.shape
    all_traj = ori_traj.reshape(L, -1)
    clustering = KMeans(
        vocab_size,
        verbose=True,
        tol=0.0,
        init="k-means++",
        n_init="auto",
    ).fit(all_traj)
    anchors = clustering.cluster_centers_.reshape(vocab_size, HORIZON, DIM)

    filename = f'./{vocab_size}.npy'
    np.save(filename, anchors)
    print(f'result saved to {filename}')
    vis(anchors)


def vis(data):
    vocab_size = data.shape[0]
    fig, ax = plt.subplots()
    for i in range(vocab_size):
        ax.plot(data[i, :, 0], data[i, :, 1])

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
