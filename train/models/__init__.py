import torchvision
from .LTP import M_GCN

model_dict = {'M_GCN': M_GCN}

def get_model(num_classes, args):
    res101 = torchvision.models.resnet101(pretrained=True)
    model = model_dict[args.model_name](res101, num_classes)
    return model