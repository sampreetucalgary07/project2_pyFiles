
import torch
import matplotlib.pyplot as plt
from subImage import get_sub_image, patch_sizes, append_patch_no, get_tensor


def predict_patch_test(l0_sub, r1_sub, model, sub_image_size, applyFunc, data_loader_inst=False, scaled=False):
    sub_patch_list = patch_sizes(sub_image_size, 1)
    pred_img_sub = torch.zeros((sub_image_size, sub_image_size, 3))
    min_value = 1
    max_value = len(list(sub_patch_list.values()))
    for enum, sub_patch in enumerate(list(sub_patch_list.values())):
        p_no = ((enum+1) - min_value) / (max_value - min_value)
        r1_n_sub = get_sub_image(
            r1_sub, sub_patch, data_loader_inst=data_loader_inst)
        l0_n_sub = get_sub_image(
            l0_sub, sub_patch, data_loader_inst=data_loader_inst)

        l0_n_sub = applyFunc(subTensor=l0_n_sub, patch_no=p_no, img_type=0.0)
        r1_n_sub = applyFunc(subTensor=r1_n_sub, patch_no=p_no, img_type=1.0)

        pred_img_sub[sub_patch[0]:sub_patch[1], sub_patch[2]:sub_patch[3], :] = model(l0_n_sub, r1_n_sub)[:, :3]

    min_r0_sub = torch.min(r1_sub)
    max_r0_sub = torch.max(r1_sub)
    min_pred = torch.min(pred_img_sub)
    max_pred = torch.max(pred_img_sub)

    if scaled:
        scaled_pred = (pred_img_sub - min_pred) / (max_pred -
                                                   min_pred) * (max_r0_sub - min_r0_sub) + min_r0_sub
        return scaled_pred

    else:
        return pred_img_sub


def all_model_pred(model_list, patch_list, img_L0_test, img_R1_test, log=True, scaled=False):
    pred_img = torch.zeros((256, 256, 3))
    for i, (model, patch) in enumerate(zip(model_list.values(), patch_list.values())):
        img_L0_test_sub = get_sub_image(
            img_L0_test, patch, print_size=False, data_loader_inst=False)
        img_R1_test_sub = get_sub_image(
            img_R1_test, patch, print_size=False, data_loader_inst=False)
        if log:
            print(f"Model No. {i+1} | Patch No. {i+1} | Patch Size : {patch}")
        pred_img[patch[0]:patch[1], patch[2]:patch[3], :] = predict_patch_test(
            img_L0_test_sub, img_R1_test_sub, model, sub_image_size=64, data_loader_inst=False, scaled=scaled)
    pred_img = pred_img.detach().numpy()

    return pred_img


def compare_plot_all(img_list, title_list, figsize=(15, 15)):
    fig, ax = plt.subplots(1, len(img_list), figsize=figsize)
    for i, (img, title) in enumerate(zip(img_list, title_list)):
        ax[i].imshow(img)
        ax[i].set_title(title)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()


def compare_plot(R0_img, predR0_img):
    """show side by side the above plots"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(R0_img)
    axs[0].set_title('R-view')
    axs[1].imshow(predR0_img)
    axs[1].set_title('predicted R-view')
    plt.show()
