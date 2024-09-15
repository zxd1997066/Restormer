## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##--------------------------------------------------------------
##------- Demo file to test Restormer on your own images---------
## Example usage on directory containing several images:   python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/' --result_dir './demo/restored/'
## Example usage on a image directly: python demo.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'
##--------------------------------------------------------------

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import time
from pdb import set_trace as stx
import numpy as np

parser = argparse.ArgumentParser(description='Test Restormer on your own images')
parser.add_argument('--input_dir', default='./demo/degraded/', type=str, help='Directory of input images or path of single image')
parser.add_argument('--result_dir', default='./demo/restored/', type=str, help='Directory for restored results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Motion_Deblurring',
                                                                                    'Single_Image_Defocus_Deblurring',
                                                                                    'Deraining',
                                                                                    'Real_Denoising',
                                                                                    'Gaussian_Gray_Denoising',
                                                                                    'Gaussian_Color_Denoising'])
parser.add_argument('--precision', default='float32', type=str, help='')
parser.add_argument('--num_iter', default=20, type=int, help='')
parser.add_argument('--num_warmup', default=5, type=int, help='')
parser.add_argument('--channels_last', default=1, type=int, help='')
parser.add_argument('--ipex', default=False, action='store_true', help='use or not')
parser.add_argument('--jit', default=False, action='store_true', help='use or not')
parser.add_argument('--profile', default=False, action='store_true', help='use or not')
parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")

args = parser.parse_args()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def get_weights_and_parameters(task, parameters):
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] =  1
        parameters['out_channels'] =  1
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters

task    = args.task
inp_dir = args.input_dir
out_dir = os.path.join(args.result_dir, task)

os.makedirs(out_dir, exist_ok=True)

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']

if any([inp_dir.endswith(ext) for ext in extensions]):
    files = [inp_dir]
else:
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(inp_dir, '*.'+ext)))
    files = natsorted(files)

if len(files) == 0:
    raise Exception(f'No files found at {inp_dir}')

# Get model weights and parameters
parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
weights, parameters = get_weights_and_parameters(task, parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

img_multiple_of = 8

print(f"\n ==> Running {task} with weights {weights}\n ")

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'Restormer-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

for file_ in tqdm(files):
    # torch.cuda.ipc_collect()
    # torch.cuda.empty_cache()
    if task == 'Gaussian_Gray_Denoising':
        img = load_gray_img(file_)
    else:
        img = load_img(file_)

    input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0)

    # Pad the input if not_multiple_of 8
    h,w = input_.shape[2], input_.shape[3]
    H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-h if h%img_multiple_of!=0 else 0
    padw = W-w if w%img_multiple_of!=0 else 0
    input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
    if args.triton_cpu:
        print("run with triton cpu backend")
        import torch._inductor.config
        torch._inductor.config.cpu_backend="triton"
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        input_ = input_.contiguous(memory_format=torch.channels_last)
        print("Running NHWC ...")
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})
    if args.ipex:
        model.eval()
        import intel_extension_for_pytorch as ipex
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        elif args.precision == "float32":
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        print("Running IPEX ...")
    if args.jit:
        with torch.no_grad():
            try:
                model = torch.jit.script(model)
                print("Running jit.script ...")
            except:
                model = torch.jit.trace(model, input_, check_trace=False, strict=False)
                print("Running jit.trace ...")
            if args.ipex:
                try:
                    model = torch.jit.freeze(model)
                    print("Running jit.freeze ...")
                except:
                    print("Failed to run jit.freeze ...")

    total_time = 0.0
    total_sample = 0
    batch_time_list = []
    with torch.no_grad():
        if args.profile:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=int((args.num_iter)/2),
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                if args.precision == "bfloat16":
                    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
                        for i in range(args.num_iter):
                            tic = time.time()
                            input_ = input_.cuda() if torch.cuda.is_available() else input_
                            restored = model(input_)
                            p.step()
                            input_ = input_.cpu()
                            toc = time.time()
                            elapsed = toc - tic
                            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                            if i >= args.num_warmup:
                                total_time += elapsed
                                total_sample += 1
                                batch_time_list.append((toc - tic) * 1000)
                elif args.precision == "float16":
                    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
                        for i in range(args.num_iter):
                            tic = time.time()
                            input_ = input_.cuda() if torch.cuda.is_available() else input_
                            restored = model(input_)
                            p.step()
                            input_ = input_.cpu()
                            toc = time.time()
                            elapsed = toc - tic
                            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                            if i >= args.num_warmup:
                                total_time += elapsed
                                total_sample += 1
                                batch_time_list.append((toc - tic) * 1000)
                else:
                    for i in range(args.num_iter):
                        tic = time.time()
                        input_ = input_.cuda() if torch.cuda.is_available() else input_
                        restored = model(input_)
                        p.step()
                        input_ = input_.cpu()
                        toc = time.time()
                        elapsed = toc - tic
                        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                        if i >= args.num_warmup:
                            total_time += elapsed
                            total_sample += 1
                            batch_time_list.append((toc - tic) * 1000)
        else:
            if args.precision == "bfloat16":
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
                    for i in range(args.num_iter):
                        tic = time.time()
                        input_ = input_.cuda() if torch.cuda.is_available() else input_
                        restored = model(input_)
                        input_ = input_.cpu()
                        toc = time.time()
                        elapsed = toc - tic
                        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                        if i >= args.num_warmup:
                            total_time += elapsed
                            total_sample += 1
                            batch_time_list.append((toc - tic) * 1000)
            elif args.precision == "float16":
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
                    for i in range(args.num_iter):
                        tic = time.time()
                        input_ = input_.cuda() if torch.cuda.is_available() else input_
                        restored = model(input_)
                        input_ = input_.cpu()
                        toc = time.time()
                        elapsed = toc - tic
                        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                        if i >= args.num_warmup:
                            total_time += elapsed
                            total_sample += 1
                            batch_time_list.append((toc - tic) * 1000)
            else:
                for i in range(args.num_iter):
                    tic = time.time()
                    input_ = input_.cuda() if torch.cuda.is_available() else input_
                    restored = model(input_)
                    input_ = input_.cpu()
                    toc = time.time()
                    elapsed = toc - tic
                    print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                    if i >= args.num_warmup:
                        total_time += elapsed
                        total_sample += 1
                        batch_time_list.append((toc - tic) * 1000)

    print("\n", "-"*20, "Summary", "-"*20)
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference latency:\t {:.3f} ms".format(latency))
    print("inference Throughput:\t {:.2f} samples/s".format(throughput))
    # P50
    batch_time_list.sort()
    p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
    p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
    p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
    print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
            % (p50_latency, p90_latency, p99_latency))

    break

    restored = torch.clamp(restored, 0, 1)

    # Unpad the output
    restored = restored[:,:,:h,:w]

    restored = restored.permute(0, 2, 3, 1).detach().numpy()
    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    # stx()
    if task == 'Gaussian_Gray_Denoising':
        save_gray_img((os.path.join(out_dir, f+'.png')), restored)
    else:
        save_img((os.path.join(out_dir, f+'.png')), restored)

    break

# print(f"\nRestored images are saved at {out_dir}")
