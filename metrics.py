
from predImg import all_model_pred
import tensorflow as tf
from tqdm import tqdm


def metrics(inp_image, out_image, max_val=1.0):
    inp_image = tf.constant(inp_image, dtype=tf.float32)
    out_image = tf.constant(out_image, dtype=tf.float32)
    ssim = tf.image.ssim(inp_image, out_image, max_val=max_val).numpy()
    psnr = tf.image.psnr(inp_image, out_image, max_val=max_val).numpy()
    return ssim, psnr


def test_pred(test_R0, test_L0, test_R1, model_list, patch_list, log=True):
    avg_ssim = 0
    avg_psnr = 0
    for img_R0_test, img_L0_test, img_R1_test in tqdm(zip(test_R0, test_L0, test_R1),
                                                      desc="Testing Samples : ", total=len(test_R0)):
        img_R0_test = img_R0_test[0]
        img_L0_test = img_L0_test[0]
        img_R1_test = img_R1_test[0]
        pred_img = all_model_pred(
            model_list, patch_list, img_L0_test, img_R1_test, log=False)
        s, p = metrics(img_R0_test, pred_img, max_val=1.0)
        avg_ssim += s
        avg_psnr += p
    if log:
        print(f"Average SSIM : {avg_ssim/len(test_R0)}")
        print(f"Average PSNR : {avg_psnr/len(test_R0)}")

    return avg_ssim/len(test_R0), avg_psnr/len(test_R0)
