import torch


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


import hashlib

def path_to_alphanumeric(path):
    # Create a new sha256 hash object
    hash_object = hashlib.sha256()

    # Encode the path to bytes and hash it
    hash_object.update(path.encode())

    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()
