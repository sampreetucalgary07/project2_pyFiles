import os
import shutil


def copyFilter_L0(R0Folder_path, L0Folder_path, destFolder_path):
    """
    Given a folder path, this function prints the contents of the folder.
    """
    # first check if the folder exists
    if not os.path.isdir(destFolder_path):
        os.mkdir(destFolder_path)
    contents = os.listdir(R0Folder_path)
    for i in contents:
        # select the files without extension
        image_name = i.split(".")[0]
        # replace the last letter with L
        image_name = image_name[:-1] + "L"
        # add the extension
        image_name = image_name + ".jpg"
        # join the path with the image name
        image_path = os.path.join(L0Folder_path, image_name)
        # check if the file exists
        if os.path.isfile(image_path):
            print(f"{i} and {image_name} exists")
        else:
            print(f"{i} and {image_name} does not exist")
            break
        # copy image to the folder destination

        shutil.copy(image_path, destFolder_path)

    print(" Total number of files in the input folder : ", len(contents))
    print(" Total number of files in the dest folder : ",
          len(os.listdir(destFolder_path)))
