
from subImage import get_sub_image, patch_sizes, append_patch_no, get_tensor
from tqdm import tqdm


def training_model_pixel(trainR0, trainL0, trainR1, model, patch_list,
                         applyFunc, applyFuncR0,
                         criterion, epochs, opt,
                         ):
    epoch_list = []
    loss_list = []
    sub_image_size = int(patch_list[1] - patch_list[0])
    for epoch in range(epochs):
        print(f"Epoch : {epoch+1}")
        model.train()
        for i, (r0, l0, r1) in tqdm(enumerate(zip(trainR0, trainL0, trainR1)), desc="Training Samples : ", total=len(trainR0)):
            r0_sub = get_sub_image(r0, patch_list)
            l0_sub = get_sub_image(l0, patch_list)
            r1_sub = get_sub_image(r1, patch_list)
            sub_patch_list = patch_sizes(sub_image_size, 1)
            min_value = 1
            max_value = len(list(sub_patch_list.values()))

            for enum, sub_patch_list in enumerate(list(sub_patch_list.values())):
                # print(enum)
                # print(sub_patch_list)
                p_no = ((enum+1) - min_value) / (max_value - min_value)
                opt.zero_grad()
                r0_64_sub = get_sub_image(
                    r0_sub, sub_patch_list, data_loader_inst=False)
                r1_64_sub = get_sub_image(
                    r1_sub, sub_patch_list, data_loader_inst=False)
                l0_64_sub = get_sub_image(
                    l0_sub, sub_patch_list, data_loader_inst=False)

                r0_64_sub = applyFuncR0(subTensor=r0_64_sub)
                l0_64_sub = applyFunc(
                    subTensor=l0_64_sub, patch_no=p_no, img_type=0.0)
                r1_64_sub = applyFunc(
                    subTensor=r1_64_sub, patch_no=p_no, img_type=1.0)

                outputs = model(l0_64_sub, r1_64_sub)
                loss = criterion(outputs, r0_64_sub)
                loss.backward()
                opt.step()
        print(f"loss : {loss}")
        loss_list.append(loss)
        epoch_list.append(epoch)

    return epoch_list, loss_list


def training_model_sq(trainR0, trainL0, trainR1, model, patch_list, criterion, epochs, opt):
    # print(model)
    # print(patch_list)

    epoch_list = []
    loss_list = []
    for epoch in range(epochs):
        model.train()
        for i, (r0, l0, r1) in enumerate(zip(trainR0, trainL0, trainR1)):
            opt.zero_grad()
            r0_sub = get_sub_image(r0, patch_list)
            l0_sub = get_sub_image(l0, patch_list)
            r1_sub = get_sub_image(r1, patch_list)
            outputs = model(l0_sub, r1_sub)
            loss = criterion(outputs, r0_sub.reshape(1, -1))
            loss.backward()
            opt.step()
        print(f"Epoch : {epoch+1} , loss : {loss}")
        loss_list.append(loss)
        epoch_list.append(epoch)

    return epoch_list, loss_list
