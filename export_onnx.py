import torch
import argparse
from pathlib import Path
from timm_model import TimmModel


def parse_args():
    parser = argparse.ArgumentParser(description='Export dynamic batch classifier to onnx, tensorrt')

    parser.add_argument('--weights', help='weights file', default='../weights/convnext_pico_256.pt')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[50, 100], help='image (h, w)')

    # onnx params
    parser.add_argument('--simplify', action='store_true', help='simplify onnx')
    parser.add_argument('--opset', type=int, default=17, help='onnx opset version')

    return parser.parse_args()


def export_onnx(model: torch.nn.Module, img: torch.Tensor, file: Path, simplify: bool = True, opset: int = 12):
    import onnx
    print('Starting export with onnx')

    out_file = file.with_suffix('.onnx')

    dynamic = {
        'images': {0: 'batch'},
        'output': {0: 'batch'},
    }

    torch.onnx.export(
        model,
        img,
        str(out_file),
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic
    )

    model_onnx = onnx.load(str(out_file))

    if simplify:
        import onnxsim
        model_onnx, check = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, str(out_file))


def main(args):
    model_path = Path(args.weights)
    model = torch.load(model_path)
    model.eval()
    img = torch.zeros([1, 3, args.imgsz[0], args.imgsz[1]])
    export_onnx(model, img, model_path, simplify=args.simplify, opset=args.opset)


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
