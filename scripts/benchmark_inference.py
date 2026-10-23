"""Benchmark the repository's inference paths, without downloading checkpoints.

Random dense weights/features: measures architecture/runtime cost, NOT accuracy.
Example: python scripts/benchmark_inference.py --repo . --mode ntk --batch 1 --device cuda --count-flops
Use separate processes for each case. CPU defaults; CUDA also supported.

Modes: zeroshot (plain CLIP), global_ridge / global_mlp (steer on the final feature),
block_smoothed (steer block_ridge on the 13 residual blocks), block_linear (steer block_ridge
on the ~50 linear-operation sites, as in block_granularity="linear"), ntk (linearized_ntk
forward as used for the fine-tuned checkpoints), ntk_single (diagnostic, one JVP instead of two).
Every timed sample is paired with a plain zero-shot forward (baseline) in alternating order.
"""
import argparse
import ast
import contextlib
import gc
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--mode', choices=['zeroshot', 'global_ridge', 'global_mlp', 'block_smoothed', 'block_linear', 'ntk', 'ntk_single'], required=True)
p.add_argument('--model', default='ViT-B-16')
p.add_argument('--batch', type=int, default=1)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
p.add_argument('--warmup', type=int, default=2)
p.add_argument('--repeat', type=int, default=None, help='Default: 7 at batch 1, 3 for larger batches')
p.add_argument('--count-flops', action='store_true')
p.add_argument('--output', type=Path)
a = p.parse_args()
if a.repeat is None:
    a.repeat = 7 if a.batch == 1 else 3
sys.path.insert(0, str(a.repo.resolve() / 'src'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import psutil
from open_clip.model import _build_vision_tower
from merge_and_rebase.models.openclip_classifier import OpenClipClassifier, zero_shot_logits_from_features
from merge_and_rebase.models.forward_modes import get_forward_mode
from merge_and_rebase.rebase.methods import steer
from merge_and_rebase.utils.linearization import forward_ad_safe_attention_context

torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(20260925)
cfg = open_clip.get_model_config(a.model)
d, h = cfg['embed_dim'], 1024

class VisualOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _build_vision_tower(d, cfg['vision_cfg'])
    def encode_image(self, images):
        return self.visual(images)

model = VisualOnly().to(a.device).eval()
clf = OpenClipClassifier(model, None, None).eval()
clf._zs_text_features = F.normalize(torch.randn(100, d, device=a.device), dim=-1)
vp = sum(t.numel() for t in model.parameters())
extra_bytes = 0
ctx = contextlib.nullcontext()
granularity = 'residual'
num_correction_blocks = None
size = cfg['vision_cfg']['image_size']

def repository_closure(index, **bindings):
    # Reuse the exact nested correction_fn source, bypassing offline prepare().
    tree = ast.parse(Path(steer.__file__).read_text())
    nodes = sorted((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'correction_fn'), key=lambda n: n.lineno)
    ns = dict(vars(steer))
    ns.update(bindings)
    module = ast.Module(body=[nodes[index]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), steer.__file__, 'exec'), ns)
    return ns['correction_fn']


if a.mode == 'global_ridge':
    coef = torch.randn(d, d, dtype=torch.float64, device=a.device) * .001
    correction = repository_closure(0, coefficient=coef)
    extra_bytes = coef.numel() * coef.element_size()
elif a.mode == 'global_mlp':
    mlp = steer._ResidualMLP(d, h, d).double().to(a.device).eval()
    correction = repository_closure(1, model=mlp)
    extra_bytes = sum(t.numel() * t.element_size() for t in mlp.parameters())
elif a.mode == 'block_smoothed':
    layers, width = cfg['vision_cfg']['layers'], cfg['vision_cfg']['width']
    # Fitted coefficients have these same shapes for independent/smoothed fits.
    coefs = [torch.randn(width, d, dtype=torch.float64, device=a.device) * .001 for _ in range(layers)]
    coefs.append(torch.randn(d, d, dtype=torch.float64, device=a.device) * .001)
    correction = repository_closure(2, coefficients=coefs, num_target_residual=layers,
        num_source_residual_blocks=layers, block_group_strategy='concat', block_granularity='residual',
        num_target_blocks_total=layers + 1, linear_site_groups=None)
    extra_bytes = sum(t.numel() * t.element_size() for t in coefs)
    num_correction_blocks = len(coefs)
elif a.mode == 'block_linear':
    # Equal-depth source/target (as ViT-B/16 -> ViT-B/16): one coefficient per linear site.
    # Coefficient input widths come from one real capture, so they match what prepare() fits.
    granularity = 'linear'
    groups = steer._linear_site_groups(model.visual, model.visual)
    capture_context, final_ids, num_sites = steer._activation_capture_for_visual(model.visual, 'linear')
    with torch.no_grad(), capture_context as capture:
        probe = F.normalize(model.encode_image(torch.randn(1, 3, size, size, device=a.device)), dim=-1)
    raw = steer._complete_captured_activations(capture, final_output=probe, final_ids=final_ids, expected_blocks=num_sites)
    grouped = steer._group_linear_site_activations(raw, groups, 'concat')
    coefs = [torch.randn(grouped[b].shape[1], d, dtype=torch.float64, device=a.device) * .001 for b in range(len(groups))]
    correction = repository_closure(2, coefficients=coefs, num_target_residual=None,
        num_source_residual_blocks=None, block_group_strategy='concat', block_granularity='linear',
        num_target_blocks_total=num_sites, linear_site_groups=groups)
    extra_bytes = sum(t.numel() * t.element_size() for t in coefs)
    num_correction_blocks = len(coefs)
    del probe, raw, grouped
elif a.mode.startswith('ntk'):
    base = {k: t.detach().clone() for k, t in model.state_dict().items()}
    with torch.no_grad():
        for t in model.parameters():
            t.add_(torch.randn_like(t) * 1e-4)
    get_forward_mode('linearized_ntk').bind(clf=clf, base_sd=base, strict_load=True)
    del base
    lin = clf._linearized_module
    extra_bytes = sum(t.numel() * t.element_size() for t in lin.ref_module.parameters())
    extra_bytes += sum(t.numel() * t.element_size() for t in lin.theta0)

if a.mode in {'global_ridge', 'global_mlp', 'block_smoothed', 'block_linear'}:
    prepared = dict(correction_fn=correction, stage_2_strategy='block_ridge' if a.mode.startswith('block') else a.mode, block_granularity=granularity)
    ctx = steer.steer_correction_context(clf, prepared, alpha=1.)

x = torch.randn(a.batch, 3, size, size, device=a.device)
original_encode = model.encode_image

def run():
    if a.mode == 'ntk_single':
        # Diagnostic: same normalized output, omit unused first raw-feature JVP.
        feat = clf._linearized_module.forward(current_module=model.visual, args=(x,), output_transform=F.normalize, post_transform=F.normalize)
        return zero_shot_logits_from_features(clf, feat)
    return clf(x)

def sync():
    if a.device == 'cuda':
        torch.cuda.synchronize()

def baseline():
    active = model.encode_image
    model.encode_image = original_encode
    try:
        return OpenClipClassifier.forward(clf, x)
    finally:
        model.encode_image = active


def count_flops(fn):
    # Force countable attention only in the diagnostic, outside timings.
    from torch.utils.flop_counter import FlopCounterMode
    with forward_ad_safe_attention_context(x.device), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops(), {str(k): v for k, v in fc.get_flop_counts()['Global'].items()}


proc = psutil.Process()
with torch.no_grad(), ctx:
    for _ in range(a.warmup):
        baseline()
        run()
    sync()
    gc.collect()
    rss_start = proc.memory_info().rss
    rss_peak = [rss_start]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.005):
            rss_peak[0] = max(rss_peak[0], proc.memory_info().rss)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    if a.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        cuda_allocated_start = torch.cuda.memory_allocated()
    samples = []
    baseline_samples = []
    try:
        for i in range(a.repeat):
            # Alternate order to limit drift/thermal/order effects.
            pair = [(run, samples), (baseline, baseline_samples)]
            if i % 2:
                pair.reverse()
            for fn, dest in pair:
                sync()
                start = time.perf_counter()
                out = fn()
                sync()
                dest.append((time.perf_counter() - start) * 1000)
        finite = bool(torch.isfinite(out).all())
    finally:
        stop.set()
        thread.join()
    result = dict(mode=a.mode, model=a.model, batch=a.batch, threads=a.threads, device=a.device,
        torch=torch.__version__, open_clip=open_clip.__version__, python=platform.python_version(),
        platform=platform.platform(), random_weights=True, classes=100, visual_dtype='float32', correction_dtype='float64',
        warmup=a.warmup, repeat=a.repeat, median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples),
        samples_ms=samples, visual_parameters=vp, visual_parameter_bytes=vp*4,
        baseline_samples_ms=baseline_samples, baseline_median_ms=statistics.median(baseline_samples),
        paired_ratio_median=statistics.median([s/b for s,b in zip(samples,baseline_samples)]),
        extra_persistent_bytes=extra_bytes, head_bytes=100*d*4, num_correction_blocks=num_correction_blocks,
        rss_after_warmup_bytes=rss_start, rss_peak_sampled_bytes=rss_peak[0], finite=finite)
    if a.device == 'cuda':
        result['gpu'] = torch.cuda.get_device_name()
        result['cuda_allocated_after_warmup_bytes'] = cuda_allocated_start
        # Peak over the timed loop, covering both run() and baseline() samples.
        result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        for name, fn in (('run', run), ('baseline', baseline)):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            fn()
            sync()
            result[f'cuda_peak_{name}_bytes'] = torch.cuda.max_memory_allocated()
            result[f'cuda_peak_{name}_activation_bytes'] = torch.cuda.max_memory_allocated() - before
    if a.mode == 'ntk_single':
        result['single_vs_repository_max_abs_error'] = float((run() - clf(x)).abs().max())
    if a.count_flops:
        try:
            result['counted_flops_math_attention'], result['flop_ops'] = count_flops(run)
            result['baseline_counted_flops_math_attention'], result['baseline_flop_ops'] = count_flops(baseline)
            # Achieved throughput: counted FLOPs over the median wall time of the timed (SDPA) path.
            result['achieved_tflops'] = result['counted_flops_math_attention'] / (result['median_ms'] / 1e3) / 1e12
            result['baseline_achieved_tflops'] = result['baseline_counted_flops_math_attention'] / (result['baseline_median_ms'] / 1e3) / 1e12
            result['flops_ratio_vs_baseline'] = result['counted_flops_math_attention'] / result['baseline_counted_flops_math_attention']
        except Exception as e:
            result['flop_count_error'] = repr(e)
print(json.dumps(result, indent=2))
if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
