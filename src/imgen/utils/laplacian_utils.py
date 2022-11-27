from torchvision.transforms import resize, GaussianBlur

def get_laplacians(im, levels=7):
    gaussians = get_gaussians(im)
    res = []
    for i in range(0, levels):
        upper = gaussians[i]
        lower = resize(gaussians[i+1], (gaussians[i].shape[0], gaussians[i].shape[1]))
        diff = upper-lower
        res.append(diff)
    return res


def get_gaussians(im, levels=7):
    res = []
    res.append(im)
    for i in range(0, levels):
        im = GaussianBlur(im, (5,5), 2)
        im = resize(im, (im.shape[0]//2, im.shape[1]//2))
        res.append(im)
    return res 