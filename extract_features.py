"""Extract patch-level features for every slide with a chosen backbone.

Writes <feat_dir>/h5_files/<slide>.h5 (features + coords) and
<feat_dir>/pt_files/<slide>.pt (features only, consumed by main.py / eval.py).
"""
import argparse
import os
import sys
import time

import h5py
import openslide
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from utils.file_utils import save_hdf5
from utils.utils import collate_features

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATS = os.path.join(ROOT, 'create_features_scripts')
BACKBONES = ['ctranspath', 'hipt4k', 'imagenet', 'lunit', 'retccl', 'simclr', 'uni']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(name, ckpt_dir):
    """Returns (model, imagenet_normalisation). Heavy imports are done lazily per backbone."""
    sys.path.insert(0, FEATS)
    if name == 'imagenet':
        from models.resnet_custom import resnet50_baseline
        return resnet50_baseline(pretrained=True), True
    if name == 'uni':
        import timm
        from huggingface_hub import hf_hub_download
        local_dir = os.path.join(ckpt_dir, 'uni')
        os.makedirs(local_dir, exist_ok=True)
        ckpt = os.path.join(local_dir, 'pytorch_model.bin')
        if not os.path.exists(ckpt):  # requires `huggingface-cli login` + access to MahmoodLab/UNI
            hf_hub_download('MahmoodLab/UNI', filename='pytorch_model.bin', local_dir=local_dir)
        model = timm.create_model('vit_large_patch16_224', img_size=224, patch_size=16,
                                  init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        return model, True
    if name == 'ctranspath':
        from TransPath.ctran import ctranspath
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(os.path.join(FEATS, 'TransPath', 'ctranspath.pth'), map_location='cpu')
        model.load_state_dict(td['model'], strict=True)
        return model, True
    if name == 'retccl':
        import ResNet
        model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        model.fc = nn.Identity()
        model.load_state_dict(torch.load(os.path.join(FEATS, 'best_ckpt.pth'), map_location='cpu'), strict=True)
        return model, True
    if name == 'simclr':
        import torchvision
        model = torchvision.models.resnet18(weights=None)
        state = torch.load(os.path.join(FEATS, 'tenpercent_resnet18.ckpt'), map_location='cpu')['state_dict']
        state = {k.replace('model.', '').replace('resnet.', ''): v for k, v in state.items()}
        model_dict = model.state_dict()
        model_dict.update({k: v for k, v in state.items() if k in model_dict})
        model.load_state_dict(model_dict)
        model.fc = nn.Sequential()
        return model, True
    if name == 'lunit':
        from timm.models.vision_transformer import VisionTransformer
        url = ('https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/'
               'pretrained-weights/dino_vit_small_patch8_ep200.torch')
        model = VisionTransformer(img_size=224, patch_size=8, embed_dim=384, num_heads=6, num_classes=0)
        print(model.load_state_dict(torch.hub.load_state_dict_from_url(url, progress=False)))
        return model, True
    if name == 'hipt4k':
        from HIPT_4K.hipt_4k import HIPT_4K
        return HIPT_4K(), True
    raise ValueError(name)


def compute_w_loader(file_path, output_path, wsi, model, batch_size, num_workers,
                     custom_downsample, target_patch_size, print_every=20):
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=True,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if device.type == 'cuda' else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_features, **kwargs)
    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}'.format(count, len(loader)))
            features = model(batch.to(device, non_blocking=True)).cpu().numpy()
            save_hdf5(output_path, {'features': features, 'coords': coords}, attr_dict=None, mode=mode)
            mode = 'a'
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('--model', type=str, required=True, choices=BACKBONES)
    parser.add_argument('--data_h5_dir', type=str, required=True, help='folder containing patches/*.h5 coordinates')
    parser.add_argument('--data_slide_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True, help='csv with a slide_id column')
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--custom_downsample', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join(ROOT, 'assets', 'ckpts'))
    parser.add_argument('--no_auto_skip', action='store_true', default=False)
    args = parser.parse_args()

    bags = Dataset_All_Bags(args.csv_path)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    done = set(os.listdir(os.path.join(args.feat_dir, 'pt_files')))

    model, _ = build_model(args.model, args.ckpt_dir)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    for i in range(len(bags)):
        slide_id = bags[i]
        if slide_id.endswith(args.slide_ext):
            slide_id = slide_id[:-len(args.slide_ext)]
        print('\nprogress: {}/{}  {}'.format(i, len(bags), slide_id))
        if not args.no_auto_skip and slide_id + '.pt' in done:
            print('skipped (already extracted)')
            continue
        h5_in = os.path.join(args.data_h5_dir, 'patches', slide_id + '.h5')
        if not os.path.exists(h5_in):
            print('WARNING: no patch file {}, skipped'.format(h5_in))
            continue
        h5_out = os.path.join(args.feat_dir, 'h5_files', slide_id + '.h5')
        wsi = openslide.open_slide(os.path.join(args.data_slide_dir, slide_id + args.slide_ext))
        t0 = time.time()
        compute_w_loader(h5_in, h5_out, wsi, model, args.batch_size, args.num_workers,
                         args.custom_downsample, args.target_patch_size)
        with h5py.File(h5_out, 'r') as f:
            features = torch.from_numpy(f['features'][:])
            print('features {} | coords {} | {:.1f}s'.format(tuple(features.shape), f['coords'].shape, time.time() - t0))
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', slide_id + '.pt'))


if __name__ == '__main__':
    main()
