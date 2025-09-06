#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import onnxsim

from lib.models.common import Conv, SPP, BottleneckCSP, Focus, Concat, Detect, SharpenConv  # noqa: F401
from torch.nn import Upsample
from lib.utils import check_anchor_order, initialize_weights

YOLOP_CFG = [
    [24, 33, 42],
    [-1, Focus, [3, 32, 3]],
    [-1, Conv, [32, 64, 3, 2]],
    [-1, BottleneckCSP, [64, 64, 1]],
    [-1, Conv, [64, 128, 3, 2]],
    [-1, BottleneckCSP, [128, 128, 3]],
    [-1, Conv, [128, 256, 3, 2]],
    [-1, BottleneckCSP, [256, 256, 3]],
    [-1, Conv, [256, 512, 3, 2]],
    [-1, SPP, [512, 512, [5, 9, 13]]],
    [-1, BottleneckCSP, [512, 512, 1, False]],
    [-1, Conv, [512, 256, 1, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [[-1, 6], Concat, [1]],
    [-1, BottleneckCSP, [512, 256, 1, False]],
    [-1, Conv, [256, 128, 1, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [[-1, 4], Concat, [1]],
    [-1, BottleneckCSP, [256, 128, 1, False]],
    [-1, Conv, [128, 128, 3, 2]],
    [[-1, 14], Concat, [1]],
    [-1, BottleneckCSP, [256, 256, 1, False]],
    [-1, Conv, [256, 256, 3, 2]],
    [[-1, 10], Concat, [1]],
    [-1, BottleneckCSP, [512, 512, 1, False]],
    [[17, 20, 23], Detect,
     [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]],
      [128, 256, 512]]],
    [16, Conv, [256, 128, 3, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, BottleneckCSP, [128, 64, 1, False]],
    [-1, Conv, [64, 32, 3, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, Conv, [32, 16, 3, 1]],
    [-1, BottleneckCSP, [16, 8, 1, False]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, Conv, [8, 2, 3, 1]],
    [16, Conv, [256, 128, 3, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, BottleneckCSP, [128, 64, 1, False]],
    [-1, Conv, [64, 32, 3, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, Conv, [32, 16, 3, 1]],
    [-1, BottleneckCSP, [16, 8, 1, False]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, Conv, [8, 2, 3, 1]]
]

class MCnet(nn.Module):
    def __init__(self, block_cfg):
        super().__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        self.num_anchors = 3
        self.num_outchannel = 5 + self.nc
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block
            if block is Detect:
                self.detector_index = i
            m = block(*args)
            m.index, m.from_ = i, from_
            layers.append(m)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        assert self.detector_index == block_cfg[0][0]
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        det = self.model[self.detector_index]
        if isinstance(det, Detect):
            s = 128
            with torch.no_grad():
                detects, _, _ = self.forward(torch.zeros(1, 3, s, s))
                det.stride = torch.tensor([s / x.shape[-2] for x in detects])
            det.anchors /= det.stride.view(-1, 1, 1)
            check_anchor_order(det)
            self.stride = det.stride
        initialize_weights(self)

    def forward(self, x):
        cache, outs = [], []
        det_out = None
        for i, m in enumerate(self.model):
            if m.from_ != -1:
                x = cache[m.from_] if isinstance(m.from_, int) else [x if j == -1 else cache[j] for j in m.from_]
            x = m(x)
            if i in self.seg_out_idx:
                outs.append(torch.sigmoid(x))
            if i == self.detector_index:
                det_out = x if self.training else x[0]
            cache.append(x if m.index in self.save else None)
        return det_out, outs[0], outs[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='weights/End-to-end.pth')
    ap.add_argument('--height', type=int, default=640)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--opset', type=int, default=12)
    ap.add_argument('--simplify', action='store_true')
    ap.add_argument('--out', type=str, default='weights/yolop-e2e-640.onnx')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MCnet(YOLOP_CFG)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    h, w = args.height, args.width
    dummy = torch.randn(1, 3, h, w, device='cpu')
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[Export] -> {out_path}")
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["images"],
        output_names=["det_out", "drive_area_seg", "lane_line_seg"],
        opset_version=args.opset,
        dynamic_axes={"images": {0: "batch"}},
        do_constant_folding=True
    )

    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)
    print("[ONNX] Check OK")

    if args.simplify:
        print("[ONNX] Simplify ...")
        model_onnx, check = onnxsim.simplify(model_onnx, check_n=3)
        assert check, "Simplify check failed"
        onnx.save(model_onnx, out_path)
        print("[ONNX] Simplified saved")

    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    print("[Runtime] inputs:", [i.name for i in sess.get_inputs()])
    print("[Runtime] outputs:", [o.name for o in sess.get_outputs()])

if __name__ == "__main__":
    main()
