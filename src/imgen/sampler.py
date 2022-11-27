import torch

from cmap_utils import CENTER_CMAP_64

def expand(blocks, stride):
    """
    =============================================================================

    Expand blocks of an image based on stride.

    For example, when stride is half of a 2x2-blocked image, it will result in 
    an expanded matrix like this:

                                        -------------------
                        ---------       |  A  | A B |  B  |
                        | A | B |       -------------------
                        ---------  ->   | A C | ... | B D |
                        | C | D |       -------------------
                        ---------       |  C  | C D |  D  |
                                        -------------------

    ============================================================================
    """
    rows = blocks.shape[0]
    cols = blocks.shape[1]
    w = blocks.shape[2]
    h = blocks.shape[3]

    res = torch.zeros((cols * h // stride - 1, rows * w // stride - 1, 4, h, w))
    idx_x = torch.arange(0, res.shape[1], w // stride)
    idx_y = torch.arange(0, res.shape[0], h // stride)

    idx = torch.stack(torch.meshgrid(idx_y, idx_x, indexing='xy')).T.reshape(-1,2)
    idx_y = idx[:,0]
    idx_x = idx[:,1]

    res[idx_y, idx_x] = blocks.clone()

    res[idx_y[:-2] + 1, idx_x[:-2], :, :h // 2, :] = res[idx_y[:-2], idx_x[:-2], :, h//2:h, :] 
    res[idx_y[:-2] + 1, idx_x[:-2], :, h // 2:, :] = res[idx_y[:-2]+2, idx_x[:-2], :, :h//2, :] 

    ## TODO: This is incorrect, but for NxN dim, works. Fix.
    res[idx_x[:-2], idx_y[:-2] + 1, :, :, :w // 2] = res[idx_x[:-2], idx_y[:-2], :, :, w//2:] 
    res[idx_x[:-2], idx_y[:-2] + 1, :, :, w // 2:] = res[idx_x[:-2], idx_y[:-2]+2, :, :, :w//2] 

    res[idx_y[:-2] + 1, idx_y[:-2] + 1, :, :, :w // 2] = res[idx_y[:-2] + 1, idx_y[:-2], :, :, w//2:] 
    res[idx_y[:-2] + 1, idx_y[:-2] + 1, :, :, w // 2:] = res[idx_y[:-2] + 1, idx_y[:-2] + 2, :, :, :w//2] 

    return res


def combine(blocks, stride, blend_fn):
    rows = blocks.shape[0]
    cols = blocks.shape[1]
    w = blocks.shape[2]
    h = blocks.shape[3]

    idx_x = torch.arange(0, blocks.shape[1], w // stride)
    idx_y = torch.arange(0, blocks.shape[0], h // stride)

    idx = torch.stack(torch.meshgrid(idx_y, idx_x, indexing='xy')).T.reshape(-1,2)
    idx_y = idx[:,0]
    idx_x = idx[:,1]

    dst = blocks[idx_y, idx_x].clone()
    src = np.zeros_like(res)
    src[1:-1, :, 4, stride:, :] = blocks[idx_y[:-2] + 1, idx_x[:-2], 4, :, :]

    mask = src.gt(0)

    return blend_fn(src, mask, dst)


def generate(
    sampler,                    # The sampler to draw from
    width=64,                   # The desired width of the output image
    height=width,               # The desired height of the output image; equal to width if not specified
    cmap=CENTER_CMAP_64,        # The guiding composition map, in the form of a matrix with dim (W, H)
    overlap=8,                  # How much to overlap each block
    stride=16,                  # How many pixels to traverse before sampling again
    blend_fn=DEFAULT_BLEND,     # Takes in any blending function in the form of fn(src, mask, dst)
    layers=1,                   # How many convolutional layers to combine during sampling
    layer_mult=[1]              # List of multipliers to weight each channel
    ):
    """Given a sampler, generate a width x height image."""
    # Loop:
        # Expand
        # Sample
        # Combine

    return


class Sampler:
    """A sampler encapsulates a model from which the generator function draws samples."""
    def __init__(self, model, dim):
        self.model = model
        self.dim = dim

    def sample(self):
        return
