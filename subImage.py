import torch


def get_sub_image(img, patch_list, print_size=False, data_loader_inst=True):
    """ Funct to get the sub image from a dataloader / img by giving a
     patch_size_list (list of 4 values)"""
    if data_loader_inst:
        subImage = img[0][patch_list[0]:patch_list[1],
                          patch_list[2]:patch_list[3], :]
    else:
        subImage = img[patch_list[0]:patch_list[1],
                       patch_list[2]:patch_list[3], :]

    if print_size:
        print("Size : ", subImage.size())
    return subImage


def patch_sizes(image_max_dim, sub_image_size, log=False):
    """ Given an image max dim and sub_image_size,
    returns a dictionary with the patch sizes values in a list
    For eg. {'Patch_1': [0, 64, 0, 64]}"""

    patch_size_list = {}
    n = 0
    for i in range(0, image_max_dim, sub_image_size):
        for j in range(0, image_max_dim, sub_image_size):
            n += 1
            patch_size_list['Patch_' +
                            str(n)] = list((i, i+sub_image_size, j, j+sub_image_size))

    if log:
        print(f"No. of patch_sizes in patch list : {n}")

    return patch_size_list


def append_pos(subTensor, patch_no, img_type=None):
    """Func. to append location/index to an single pixel tensor"""

    patch_no_t = torch.tensor([patch_no])
    tensor_with_point = torch.cat(
        (subTensor.view(1, 3), patch_no_t.view(1, len(patch_no_t))), dim=1)
    return tensor_with_point


def get_tensor(subTensor, patch_no=None, img_type=None):
    tensor_with_point = subTensor.view(1, 3)  # 1,3 is the size of the tensor
    return tensor_with_point


def append_posAndImg(subTensor, patch_no, img_type=None):
    """Func. to append location/index + image type to an single pixel tensor"""

    patch_no_t = torch.tensor([patch_no])
    img_type_t = torch.tensor([img_type])
    tensor_with_point = torch.cat(
        (subTensor.view(1, 3), patch_no_t.view(1, len(patch_no_t)), img_type_t.view(1, len(img_type_t))), dim=1)
    return tensor_with_point
