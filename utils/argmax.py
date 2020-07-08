import numpy
import kornia
import torch
import torch.nn as nn
# from kornia import spatial_soft_argmax2d


# argmax and soft-argmax operation for heatmap coordinate extracting.

def hm_argmax(input_hm):
    '''
    get x and y coordinate from heatmap
    Input : tensor [H x W]
    Output : [x, y]
    '''
    hm_size_w = input_hm.size()[-1]
    hm_max = torch.argmax(input_hm)
    x = hm_max % hm_size_w
    y = hm_max // hm_size_w
    coords = x, y
    return coords


def soft_argmax(voxels):
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert voxels.dim() == 5
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 100.0
    N, C, H, W, D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N, C, -1)*alpha, dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0, end=H*W*D).unsqueeze(0)
    indices_kernel = indices_kernel.view((H, W, D))
    conv = soft_max*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices % D
    y = (indices/D).floor() % W
    x = (((indices/D).floor())/W).floor() % H
    coords = torch.stack([x, y, z], dim=2)
    return coords

def spatial_soft_argmax2d(input, temperature, normalized_coordinates, eps = 1e-08):
    """ Docs of kornia.spatial_soft_argmax2d:
    
        Function that computes the Spatial Soft-Argmax 2D of a given input heatmap.
        Returns the index of the maximum 2d coordinates of the give map.
        The output order is x-coord and y-coord.

        Arguments:
            temperature (torch.Tensor): factor to apply to input. Default is 1.
            normalized_coordinates (bool): whether to return the 
            coordinates normalized in the range of [-1, 1]. Otherwise,
            it will return the coordinates in the range of the input shape.
            Default is True.
            eps (float): small value to avoid zero division. Default is 1e-8.

        Shape:
            - Input: :math:`(B, N, H, W)`
            - Output: :math:`(B, N, 2)`

        Examples:
            >>> input = torch.tensor([[[
                [0., 0., 0.],
                [0., 10., 0.],
                [0., 0., 0.]]]])
            >>> coords = kornia.spatial_soft_argmax2d(input, 1, False)
            tensor([[[1.0000, 1.0000]]])    
    """
    coors = kornia.spatial_soft_argmax2d(input, temperature, normalized_coordinates, eps)
    # kornia 基于 PyTorch 提供可微分操作,所以要 request_grad = False
    coors = coors.detach()
    return coors

# Attention: temperature is a factor to apply to input. But the default value 1 is not suitable,
# maybe 1000 is better by my experiment, and you should must instantiate it in input.

if __name__ == '__main__':

    from heatmap import ThreeSigmaGaussian

    hm = ThreeSigmaGaussian(100, 100, 22, 59, 100)
    hm = torch.from_numpy(hm)

    print(hm_argmax(hm))

    hm = hm.view(1, 1, 100, 100)
    print(spatial_soft_argmax2d(hm,1000,False))

    hm = hm.view(1,1,100,100,1)
    print(soft_argmax(hm))
    '''
    >>> (tensor(22), tensor(59))
        tensor([[[22.0000, 59.0000]]], dtype=torch.float64)
        tensor([[[59.0000, 22.0000,  0.1757]]], dtype=torch.float64)
    '''