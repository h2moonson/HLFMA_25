# export_onnx_trt.py
import argparse, math, torch, torch.nn as nn
import onnx
try:
    import onnxsim
except Exception:
    onnxsim = None

from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv
from torch.nn import Upsample
from lib.utils import check_anchor_order, initialize_weights

YOLOP = [
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
     [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], [128, 256, 512]]],

    [16, Conv, [256, 128, 3, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, BottleneckCSP, [128, 64, 1, False]],
    [-1, Conv, [64, 32, 3, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, Conv, [32, 16, 3, 1]],
    [-1, BottleneckCSP, [16, 8, 1, False]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, Conv, [8, 2, 3, 1]],  # 33 drive_area_seg

    [16, Conv, [256, 128, 3, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, BottleneckCSP, [128, 64, 1, False]],
    [-1, Conv, [64, 32, 3, 1]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, Conv, [32, 16, 3, 1]],
    [-1, BottleneckCSP, [16, 8, 1, False]],
    [-1, Upsample, [None, 2, 'nearest']],
    [-1, Conv, [8, 2, 3, 1]]   # 42 lane_line_seg
]

class MCnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = cfg[0][0]
        self.seg_out_idx = cfg[0][1:]
        self.num_anchors = 3
        self.num_outchannel = 5 + self.nc
        for i, (from_, block, args) in enumerate(cfg[1:]):
            block = eval(block) if isinstance(block, str) else block
            if block is Detect:
                self.detector_index = i
            m = block(*args)
            m.index, m.from_ = i, from_
            layers.append(m)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        assert self.detector_index == cfg[0][0]
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        Det = self.model[self.detector_index]
        if isinstance(Det, Detect):
            s = 128
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _ = model_out
                Det.stride = torch.tensor([s / x.shape[-2] for x in detects])
            Det.anchors /= Det.stride.view(-1,1,1)
        initialize_weights(self)

    def forward(self, x):
        cache, out = [], []
        det_out = None
        for i, m in enumerate(self.model):
            if m.from_ != -1:
                x = cache[m.from_] if isinstance(m.from_, int) else [x if j==-1 else cache[j] for j in m.from_]
            x = m(x)
            if i in self.seg_out_idx:
                out.append(torch.sigmoid(x))
            if i == self.detector_index:
                det_out = x if self.training else x[0]
            cache.append(x if m.index in self.save else None)
        return det_out, out[0], out[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/End-to-end.pth')
    ap.add_argument('--height', type=int, default=640)
    ap.add_argument('--width',  type=int, default=640)
    ap.add_argument('--opset',  type=int, default=12)
    ap.add_argument('--out',    default=None)
    ap.add_argument('--simplify', action='store_true')
    args = ap.parse_args()

    H, W = args.height, args.width
    out_path = args.out or f'weights/yolop-{H}-{W}.onnx'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MCnet(YOLOP).to('cpu').eval()  # export는 CPU로 충분

    ckpt = torch.load(args.weights, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state, strict=False)

    dummy = torch.randn(1, 3, H, W, device='cpu')

    print(f'Export -> {out_path}')
    torch.onnx.export(
        model, dummy, out_path,
        opset_version=args.opset,
        input_names=["images"],
        # C++/TRT에서 쓰는 정확한 이름으로 고정!
        output_names=["det_out", "drive_area_seg", "lane_line_seg"],
        dynamic_axes={"images": {0: "batch"}},
        do_constant_folding=True
    )

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX check: OK')

    if args.simplify:
        if onnxsim is None:
            raise RuntimeError("onnx-simplifier(onnxsim) 미설치: pip install onnxsim")
        print(f'Simplify with onnx-simplifier {onnxsim.__version__}')
        simp, ok = onnxsim.simplify(onnx_model, check_n=3)
        assert ok, 'onnxsim check failed'
        onnx.save(simp, out_path)
        print('Simplified saved:', out_path)

if __name__ == "__main__":
    main()
