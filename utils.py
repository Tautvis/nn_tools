from typing import Optional, Union, Callable, Iterable, List

def receptive_field(kernel_sizes: Union[int, List[int], np.ndarray],
                    dilations: Union[int, List[int], np.ndarray] = 1,
                    strides: Union[int, List[int], np.ndarray] = 1,
                    n_layers: Optional[int] = None) -> int:
    """
    Compute receptive field.
    :param kernel_sizes: list of kernel sizes or a single integer if they are all the same size.
    :param dilations: dilation for each layer.
    :param strides: stride for each layer.
    :param n_layers: number of layers. Optional.
    :return: Receptive field.
    """
    if n_layers is None:
        if hasattr(kernel_sizes, '__len__'):
            n_layers = len(kernel_sizes)
        if hasattr(dilations, '__len__'):
            n_layers = len(dilations)
        if hasattr(strides, '__len__'):
            n_layers = len(strides)
        if n_layers is None:
            raise ValueError('At least one of the inputs has to be a list or n_layers specified.')
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * n_layers
    if isinstance(dilations, int):
        dilations = [dilations] * n_layers
    if isinstance(strides, int):
        strides = [strides] * n_layers
    assert len(kernel_sizes) == len(dilations) == len(strides)
    rc = 1
    jump_factor = 1
    for k, d, s in zip(kernel_sizes, dilations, strides):
        effective_k = d * (k - 1) + 1
        rc = rc + (effective_k - 1) * jump_factor
        jump_factor = jump_factor * s

    return rc
