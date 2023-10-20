import torch
import numpy as np
import matplotlib.pyplot as plt


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


def create_difference_heatmap(image1, image2, title, cmap='hot', alpha=0.85, show=True):
    """
    Create a heat map highlighting pixel differences between two images and display percentage difference.
    """
    # Convert the PyTorch tensors to NumPy arrays
    if type(image1) != np.ndarray:
        image1 = image1.numpy()
    if type(image2) != np.ndarray:
        image2 = image2.numpy()
    # Calculate pixel-wise absolute differences between the two images
    diff_image = np.abs(image1 - image2)
    diff_image_sum = np.sum(diff_image, axis=2)
    # Calculate percentage difference
    percentage_diff = (np.sum(diff_image_sum) /
                       np.sum(np.sum(image1, axis=2))) * 100
    # Create a heat map using the difference values
    if show:
        plt.imshow(diff_image_sum, cmap=cmap, alpha=alpha)
        # add info contains both percentage diff and total diff
        additional_info = f"Percentage Difference: {np.mean(percentage_diff):.4f}%\nTotal Difference: {np.sum(diff_image_sum):.3f}"
        info_x = 0.5  # X-coordinate (normalized)
        info_y = -0.02  # Y-coordinate (normalized)
        plt.figtext(info_x, info_y, additional_info, fontsize=9, ha='center')
        # Add a colorbar for reference
        plt.colorbar()
        plt.title(title, fontsize=11)
        # Display the heat map
        plt.show()
    return np.mean(percentage_diff), np.sum(diff_image_sum)


class neighbourPixels():
    def __init__(self, sub_image_size):
        self.sub_image_size = sub_image_size

    def returnNeigbourPixels(self, i, j):
        return [(i, j), (i+1, j), (i+1, j+1), (i, j+1), (i-1, j+1), (i-1, j), (i-1, j-1), (i, j-1), (i+1, j-1)]

    def checkPixels(self, n_list_values):
        for pixel in n_list_values:
            # print(pixel)
            if -1 in pixel:
                n_list_values[n_list_values.index(pixel)] = (0, 0)
                continue
            if 64 in pixel:
                n_list_values[n_list_values.index(pixel)] = (0, 0)
        return n_list_values

    def neighbour9Pixels(self):
        n_list = {}
        for i in range(self.sub_image_size):
            for j in range(self.sub_image_size):
                n_list[(i, j)] = self.returnNeigbourPixels(i, j)
                n_list[(i, j)] = self.checkPixels(n_list[(i, j)])

        return n_list
