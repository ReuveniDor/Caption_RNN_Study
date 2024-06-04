import math
import os
import time
import multiprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from a5_helper import load_coco_captions, decode_captions, train_captioner
from rnn_lstm_captioning import CaptioningRNN
from torchvision import transforms
from torchvision.utils import make_grid

# This project is based on EECS 498-007/598-005 Assignment 5-1: Image captioning with RNNs and LSTMs
# The project is to implement an image captioning model using RNNs and LSTMs with attention mechanism
# The project is implemented in PyTorch

# Settings for the plots
plt.style.use("seaborn-v0_8")  # Prettier plots
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["font.size"] = 24
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# Choose the device to be used for training
if torch.cuda.is_available():
    print("Good to go!")
    DEVICE = torch.device("cuda")
else:
    print("Please set GPU via Edit -> Notebook Settings.")
    DEVICE = torch.device("cpu")

# Define some common variables for dtypes/devices.
# These can be keyword arguments while defining new tensors.
to_float = {"dtype": torch.float32, "device": DEVICE}
to_double = {"dtype": torch.float64, "device": DEVICE}

# Set a few constants related to data loading.
IMAGE_SHAPE = (112, 112)
NUM_WORKERS = multiprocessing.cpu_count()

# Batch size used for full training runs:
BATCH_SIZE = 256

# Batch size used for overfitting sanity checks:
OVR_BATCH_SIZE = BATCH_SIZE // 8

# Batch size used for visualization:
VIS_BATCH_SIZE = 4

def data_downloading():
    '''
    Use serialized COCO data from coco.pt
    It contains a dictionary of
    "train_images" - resized training images (IMAGE_SHAPE)
    "val_images" - resized validation images (IMAGE_SHAPE)
    "train_captions" - tokenized and numericalized training captions
    "val_captions" - tokenized and numericalized validation captions
    "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"
    '''
    if os.path.isfile("./datasets/coco.pt"):
        print("COCO data exists!")
    else:
        print("downloading COCO dataset")
        !wget http://web.eecs.umich.edu/~justincj/teaching/eecs498/coco.pt -P ./datasets/

    # load COCO data from coco.pt, loaf_COCO is implemented in a5_helper.py
    data_dict = load_coco_captions(path="./datasets/coco.pt")

    num_train = data_dict["train_images"].size(0)
    num_val = data_dict["val_images"].size(0)

    # declare variables for special tokens
    NULL_index = data_dict["vocab"]["token_to_idx"]["<NULL>"]
    START_index = data_dict["vocab"]["token_to_idx"]["<START>"]
    END_index = data_dict["vocab"]["token_to_idx"]["<END>"]
    UNK_index = data_dict["vocab"]["token_to_idx"]["<UNK>"]
    return data_dict, num_train, num_val, NULL_index, START_index, END_index, UNK_index

# Sample a minibatch and show the reshaped 112x112 images and captions
def minibatch_show():
    sample_idx = torch.randint(0, num_train, (VIS_BATCH_SIZE, ))
    sample_images = data_dict["train_images"][sample_idx]
    sample_captions = data_dict["train_captions"][sample_idx]
    for i in range(VIS_BATCH_SIZE):
        plt.imshow(sample_images[i].permute(1, 2, 0))
        plt.axis("off")
        caption_str = decode_captions(
            sample_captions[i], data_dict["vocab"]["idx_to_token"]
        )
        plt.title(caption_str)
        plt.show()

# Overfitting test - shiuld yield a loss of less than 0.5
def overfitting_test():
        
    # data input
    small_num_train = 50
    sample_idx = torch.linspace(0, num_train - 1, steps=small_num_train).long()
    small_image_data = data_dict["train_images"][sample_idx]
    small_caption_data = data_dict["train_captions"][sample_idx]

    # optimization arguments
    num_epochs = 80

    # create the image captioning model
    model = CaptioningRNN(
        cell_type="rnn",
        word_to_idx=data_dict["vocab"]["token_to_idx"],
        input_dim=400,  # hard-coded, do not modify
        hidden_dim=512,
        wordvec_dim=256,
        ignore_index=NULL_index,
    )
    model = model.to(**to_float)

    for learning_rate in [1e-3]:
        print("learning rate is: ", learning_rate)
        rnn_overfit, _ = train_captioner(
            model,
            small_image_data,
            small_caption_data,
            num_epochs=num_epochs,
            batch_size=OVR_BATCH_SIZE,
            learning_rate=learning_rate,
            device=DEVICE,
        )

# Train RNN model. If the model is already trained, load the model
def rnn_model_train():
    if not os.path.exists("rnnDICT.py"):
        # data input
        small_num_train = num_train
        sample_idx = torch.randint(num_train, size=(small_num_train,))
        small_image_data = data_dict["train_images"][sample_idx]
        small_caption_data = data_dict["train_captions"][sample_idx]

        # create the image captioning model
        rnn_model = CaptioningRNN(
            cell_type="rnn",
            word_to_idx=data_dict["vocab"]["token_to_idx"],
            input_dim=400,  # hard-coded, do not modify
            hidden_dim=512,
            wordvec_dim=256,
            ignore_index=NULL_index,
        )

        for learning_rate in [1e-3]:
            print("learning rate is: ", learning_rate)
            rnn_model_submit, rnn_loss_submit = train_captioner(
                rnn_model,
                small_image_data,
                small_caption_data,
                num_epochs=60,
                batch_size=BATCH_SIZE,
                learning_rate=learning_rate,
                device=DEVICE,
            )
    else:
        rnn_model = CaptioningRNN(cell_type="rnn",
            word_to_idx=data_dict["vocab"]["token_to_idx"],
            input_dim=400,  # hard-coded, do not modify
            hidden_dim=512,
            wordvec_dim=256,
            ignore_index=NULL_index,)
        rnn_model.load_state_dict(torch.load('rnnDICT.py', map_location=torch.device('cpu')))
    return rnn_model

# Show the results of the trained RNN model on the training and validation sets
def show_result(rnn_model):
    rnn_model.eval()

    for split in ["train", "val"]:
        sample_idx = torch.randint(
            0, num_train if split == "train" else num_val, (VIS_BATCH_SIZE,)
        )
        sample_images = data_dict[split + "_images"][sample_idx]
        sample_captions = data_dict[split + "_captions"][sample_idx]

        # decode_captions is loaded from a5_helper.py
        gt_captions = decode_captions(sample_captions, data_dict["vocab"]["idx_to_token"])

        generated_captions = rnn_model.sample(sample_images.to(DEVICE))
        generated_captions = decode_captions(
            generated_captions, data_dict["vocab"]["idx_to_token"]
        )

        for i in range(VIS_BATCH_SIZE):
            plt.imshow(sample_images[i].permute(1, 2, 0))
            plt.axis("off")
            plt.title(
                f"[{split}] RNN Generated: {generated_captions[i]}\nGT: {gt_captions[i]}"
            )
            plt.show()

# Attention model training
def attn_model_train():
    small_num_train = num_train
    sample_idx = torch.randint(num_train, size=(small_num_train,))
    small_image_data = data_dict["train_images"][sample_idx]
    small_caption_data = data_dict["train_captions"][sample_idx]

    # create the image captioning model
    attn_model = CaptioningRNN(
        cell_type="attn",
        word_to_idx=data_dict["vocab"]["token_to_idx"],
        input_dim=400,  # hard-coded, do not modify
        hidden_dim=512,
        wordvec_dim=256,
        ignore_index=NULL_index,
    )
    
    if not os.path.exists("attn_modelDICT.py"):
        
        attn_model = attn_model.to(DEVICE)

        for learning_rate in [1e-3]:
            print("learning rate is: ", learning_rate)
            attn_model_submit, attn_loss_submit = train_captioner(
                attn_model,
                small_image_data,
                small_caption_data,
                num_epochs=60,
                batch_size=BATCH_SIZE,
                learning_rate=learning_rate,
                device=DEVICE,
            )
    else:
        attn_model.load_state_dict(torch.load('attn_modelDICT.py'))
    return attn_model

# Show the results of the trained attention model on the training and validation sets
def show_attn_result(attn_model):
    attn_model.eval()
    for split in ["train", "val"]:
        sample_idx = torch.randint(
            0, num_train if split == "train" else num_val, (VIS_BATCH_SIZE,)
        )
        sample_images = data_dict[split + "_images"][sample_idx]
        sample_captions = data_dict[split + "_captions"][sample_idx]

        # decode_captions is loaded from a5_helper.py
        gt_captions = decode_captions(sample_captions, data_dict["vocab"]["idx_to_token"])
        attn_model.eval()
        generated_captions, attn_weights_all = attn_model.sample(sample_images.to(DEVICE))
        generated_captions = decode_captions(
            generated_captions, data_dict["vocab"]["idx_to_token"]
        )

        for i in range(VIS_BATCH_SIZE):
            plt.imshow(sample_images[i].permute(1, 2, 0))
            plt.axis("off")
            plt.title(
                "%s\nAttention LSTM Generated:%s\nGT:%s"
                % (split, generated_captions[i], gt_captions[i])
            )
            plt.show()

            tokens = generated_captions[i].split(" ")

            vis_attn = []
            for j in range(len(tokens)):
                img = sample_images[i]
                attn_weights = attn_weights_all[i][j]
                token = tokens[j]
                img_copy = attention_visualizer(img, attn_weights, token)
                vis_attn.append(transforms.ToTensor()(img_copy))

            plt.rcParams["figure.figsize"] = (20.0, 20.0)
            vis_attn = make_grid(vis_attn, nrow=8)
            plt.imshow(torch.flip(vis_attn, dims=(0,)).permute(1, 2, 0))
            plt.axis("off")
            plt.show()
            plt.rcParams["figure.figsize"] = (10.0, 8.0)
    
def attention_visualizer(img, attn_weights, token):
    """
    Visuailze the attended regions on a single frame from a single query word.
    Inputs:
    - img: Image tensor input, of shape (3, H, W)
    - attn_weights: Attention weight tensor, on the final activation map
    - token: The token string you want to display above the image

    Outputs:
    - img_output: Image tensor output, of shape (3, H+25, W)

    """
    C, H, W = img.shape
    assert C == 3, "We only support image with three color channels!"

    # Reshape attention map
    attn_weights = cv2.resize(
        attn_weights.data.numpy().copy(), (H, W), interpolation=cv2.INTER_NEAREST
    )
    attn_weights = np.repeat(np.expand_dims(attn_weights, axis=2), 3, axis=2)

    # Combine image and attention map
    img_copy = img.float().div(255.0).permute(1, 2, 0).numpy()[:, :, ::-1].copy()
    masked_img = cv2.addWeighted(attn_weights, 0.5, img_copy, 0.5, 0)
    img_copy = np.concatenate((np.zeros((25, W, 3)), masked_img), axis=0)

    # Add text
    cv2.putText(
        img_copy,
        "%s" % (token),
        (10, 15),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (255, 255, 255),
        thickness=1,
    )

    return img_copy

data_dict, num_train, num_val, NULL_index, START_index, END_index, UNK_index = data_downloading()
