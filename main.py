import torch
from preprocess.preprocess import make_joint_dataset
from model.vae import VAE


def find_device(force_cpu):

    if not force_cpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def main(force_cpu=True):
    device = find_device(force_cpu)
    print("Using device", device)

    # Load the data
    train_dataset = make_joint_dataset(device=device)
    # vae = VAE.train_routine(train_dataset=train_dataset, device=device)

    x = train_dataset.untransform_and_unscale()
    print(x.shape)


if __name__ == "__main__":
    main()
