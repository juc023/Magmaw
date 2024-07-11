import random
import torch
import torch.nn.functional as F

def random_crop_and_pad_image_and_labels(image, labels, size):
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size() # [3, 256, 448]

    combined_pad = F.pad(combined, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))

    #print(combined_pad.size()) # [6, 256, 448]

    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :])

def fvc_random_crop_and_pad_image_and_labels(image, labels_1, labels_2, labels_3, size):
    combined = torch.cat([image, labels_1], 0)
    combined2 = torch.cat([labels_2, labels_3], 0)

    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(combined, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    combined_pad2 = F.pad(combined2, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))

    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[2]) - size[1])

    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    combined_crop2 = combined_pad2[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]

    return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :], combined_crop2[:last_image_dim, :, :], combined_crop2[last_image_dim:, :, :])

def random_flip(images, labels):
    
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1


    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])

    return images, labels

def fvc_random_flip(images, labels_1, labels_2, labels_3):
    
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1


    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [1])
        labels_1 = torch.flip(labels_1, [1])
        labels_2 = torch.flip(labels_2, [1])
        labels_3 = torch.flip(labels_3, [1])

    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels_1 = torch.flip(labels_1, [2])
        labels_2 = torch.flip(labels_2, [2])
        labels_3 = torch.flip(labels_3, [2])

    return images, labels_1, labels_2, labels_3