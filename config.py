##################################################
# Config
##################################################

# generate testing/valid dataset

model_folder = "./models/"
train_path = "./data/train_images/"  # training dataset path
coco_train_file = "./data/pascal_train.json"
test_path = "./data/test_images/"  # testing images
##################################################
# Training Config
##################################################


# model parameter
model_name = "mask_rcnn_v1"


batch_size = 4  # batch size
accumulation_steps = 4  # Gradient Accumulation
num_workers = 4  # number of Dataloader workers

# mode choose
bifpn = False  # If True, use BiFPN instead of FPN
eval_train = False  # If True, Enable evaluate training


# traning parameter
min_size = 680
max_size = 680
num_classes = 21
epochs = 60

# Scheduler parameter
learning_rate = 5e-3
momentum = 0.9
weight_decay = 5e-4
T_mult = 1
eta_min = 5e-5

##################################################
# Testing Config
# ##################################################
result_pth = "./result/"
train_result = "submission_train.json"
json_name = "0856566.json"  # submit json name
