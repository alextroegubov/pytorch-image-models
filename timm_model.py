import torch
import torch.nn as nn
from timm import create_model
from argparse import ArgumentParser

# class TimmNet(ConvNeXt):
#     def __init__(self, model):
#         super(TimmNet, self).__init__()
#         self.model = model
#         self.model.head.add_module("softmax", nn.Softmax(dim=1))
#         self.model.head.softmax.requires_grad_(False)
#         # self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.model(x)
#         # x = self.softmax(x)
#         x = self.model.head.softmax(x)
#         return x


class TimmModel(nn.Module):
    def __init__(self, model):
        super(TimmModel, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        self.softmax.requires_grad_(False)
  
    def forward(self, x):
        x = self.model(x)
        return self.softmax(x)


def save_model_with_softmax(model_name: str, checkpoint: str,
                            num_classes: int, dst_name: str):
    model = create_model(
        model_name=model_name,
        checkpoint_path=checkpoint,
        num_classes=num_classes,
        pretrained=True
    )

    model.load_state_dict(torch.load(checkpoint)['state_dict'])

    timm_model = TimmModel(model)
    timm_model.eval()

    torch.save(timm_model, dst_name)


parser = ArgumentParser()
parser.add_argument('--model_name', type=str,
                    default='convnext_pico.d1_in1k')
parser.add_argument('--checkpoint', type=str,
                    default='/home/user/Documents/repos/pytorch-image-models/output/convnext_pico.d1_in1k-50_100_pretrained_pad_around_3_cls-v7/model_best.pth.tar')
parser.add_argument('--num_classes', type=int,
                    default=3)
parser.add_argument('--dst_name', type=str,
                    default='plate_model_sm.pt')

if __name__ == '__main__':
    args = parser.parse_args()
    save_model_with_softmax(args.model_name, args.checkpoint, args.num_classes,
                            args.dst_name)
