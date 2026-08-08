from src.core.updater import PyTorchLocalUpdater

class ByzantineUpdater(PyTorchLocalUpdater):
    """
    Subclass of PyTorchLocalUpdater preserved for backward compatibility.
    All Byzantine attack vectors (label_flip, sign_flip, gradient_ascent, random_noise)
    are natively implemented in PyTorchLocalUpdater.
    """
    pass

