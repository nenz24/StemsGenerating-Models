#!/usr/bin/env python3
"""Export the complete 24-layer pretrained denoiser for one velocity prediction.

Synthetic 25-frame latent input; no source-audio conditioning or diffusion loop.
FP16 stored parameters, FP32 calculations, dual references with unchanged bounds.
"""
import argparse
import ast
import __future__
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

from export_coreai_decoder import ROOT, checkpoint_path, sha256
from export_coreai_dit_block import records, verified_source, load_upstream, convert_storage16, precision_metrics

INPUT_SHAPES = {'noise': [1, 25, 64], 'condition': [1, 64, 2048], 'context': [1, 25, 128],
                'timestep': [1], 'timestep_r': [1]}
OUTPUT_SHAPE = [1, 25, 64]


def load_full_upstream(paths):
    cfg, layer_type, rotary_type, _ = load_upstream(paths['modeling_acestep_v15_base.py'], paths['config.json'],
                                                   paths['configuration_acestep_v15.py'])
    import torch
    import torch.nn.functional as F
    from transformers.modeling_utils import PreTrainedModel
    from transformers.utils import auto_docstring, logging
    namespace = sys.modules[layer_type.__module__]
    namespace.__dict__.update(math=math, F=F, PreTrainedModel=PreTrainedModel, AceStepConfig=type(cfg),
                              auto_docstring=auto_docstring, logger=logging.get_logger(namespace.__name__),
                              Qwen3RotaryEmbedding=rotary_type)
    source = paths['modeling_acestep_v15_base.py']
    tree = ast.parse(source.read_text(), filename=str(source))
    names = {'TimestepEmbedding', 'AceStepPreTrainedModel', 'Lambda', 'AceStepDiTModel'}
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(nodes) != 4:raise ValueError('Pinned full-denoiser classes missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace.__dict__)
    return cfg, namespace.AceStepDiTModel, rotary_type


def verify_bundle(folder, record):
    ref = json.loads((folder/'reference.json').read_text())
    if (ref.get('schema_version') != 1 or ref.get('profile') != 'denoiser'
            or ref.get('asset_file') != 'denoiser.aimodel' or ref.get('function_name') != 'main'
            or ref.get('output_name') != 'velocity' or ref.get('input_shapes') != INPUT_SHAPES
            or ref.get('output_shape') != OUTPUT_SHAPE or ref.get('layer_count') != 24
            or ref.get('checkpoint_prefix') != 'decoder.' or ref.get('latent_frames') != 25
            or ref.get('weight_storage_dtype') != 'float16' or ref.get('compute_dtype') != 'float32'
            or ref.get('scalar_type') != 'float32' or ref.get('byte_order') != 'little'
            or ref.get('checkpoint', {}).get('digest') != record['digest']
            or ref.get('checkpoint_full_sha256_verified') is not True
            or ref.get('original_fp32_reference_file') != 'velocity-fp32-reference.f32le'
            or ref.get('storage16_vs_original_fp32', {}).get('parity_passed') is not True):
        raise ValueError('Unexpected complete-denoiser contract')
    hashes = ref['artifact_sha256']
    required = {f'{name}.f32le' for name in INPUT_SHAPES} | {'velocity-reference.f32le','velocity-fp32-reference.f32le',
                'denoiser.aimodel/main.mlirb','denoiser.aimodel/main.hash','denoiser.aimodel/metadata.json'}
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != '.DS_Store' and p != folder/'reference.json'}
    if not required.issubset(hashes) or set(hashes) != actual:raise ValueError('Incomplete denoiser manifest')
    for name, digest in hashes.items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != digest:raise ValueError(f'Denoiser integrity mismatch: {name}')
    for name, shape in [(f'{n}.f32le',s) for n,s in INPUT_SHAPES.items()] + [
            ('velocity-reference.f32le',OUTPUT_SHAPE),('velocity-fp32-reference.f32le',OUTPUT_SHAPE)]:
        if (folder/name).stat().st_size != math.prod(shape)*4:raise ValueError('Denoiser binary size mismatch')
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT/'prototypes/ios27-coreai/ModelCandidates/denoiser-pretrained')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    publisher = records();record = publisher['model.safetensors']
    if args.verify_only:
        verify_bundle(args.out, record);print('Existing full denoiser: VERIFIED (no download/re-export)');return
    if args.out.exists():raise FileExistsError(f'Output already exists: {args.out}; preserved')
    print('Preparing full 24-layer denoiser; cached Base checkpoint reused (4.79 GB if missing).',flush=True)
    checkpoint = checkpoint_path(record,args.checkpoint)
    print('Full publisher checkpoint SHA-256: VERIFIED',flush=True)
    sources = ROOT/'.runtime-cache/coreai-generator-source-publisher';sources.mkdir(parents=True,exist_ok=True)
    paths = {name:verified_source(publisher[name],sources) for name in [
        'modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json']}
    cfg, kind, rotary_type = load_full_upstream(paths)
    if cfg.num_hidden_layers != 24 or cfg.patch_size != 2 or cfg.in_channels != 192:
        raise ValueError('Pinned denoiser configuration changed')
    import numpy as np
    import torch
    from safetensors import safe_open
    from coreai_torch import TorchConverter,get_decomp_table
    torch.set_num_threads(2)
    with torch.device('meta'):denoiser = kind(cfg).eval()
    with safe_open(str(checkpoint),framework='pt',device='cpu') as source:
        state = {key[len('decoder.'):]:source.get_tensor(key) for key in source.keys() if key.startswith('decoder.')}
    denoiser.load_state_dict(state,strict=True,assign=True)
    del state
    # inv_freq is an analytical, nonpersistent buffer, not a checkpoint tensor.
    denoiser.rotary_emb = rotary_type(cfg,device='cpu')
    denoiser.to(device='cpu',dtype=torch.float32).requires_grad_(False)
    print('All decoder tensors loaded strictly; parameters:',sum(p.numel() for p in denoiser.parameters()),flush=True)

    class Velocity(torch.nn.Module):
        def __init__(self,denoiser):super().__init__();self.denoiser=denoiser
        def forward(self,noise,condition,context,timestep,timestep_r):
            return self.denoiser(hidden_states=noise,encoder_hidden_states=condition,context_latents=context,
                                  timestep=timestep,timestep_r=timestep_r,attention_mask=None,encoder_attention_mask=None,
                                  use_cache=False,output_attentions=False)[0]

    model = Velocity(denoiser).eval()
    generator = torch.Generator(device='cpu').manual_seed(0)
    values = {'noise':torch.randn(INPUT_SHAPES['noise'],generator=generator),
              'condition':torch.randn(INPUT_SHAPES['condition'],generator=generator)*0.1,
              'context':torch.randn(INPUT_SHAPES['context'],generator=generator)*0.1,
              'timestep':torch.tensor([0.5]),'timestep_r':torch.tensor([0.25])}
    inputs = tuple(values[name] for name in INPUT_SHAPES)
    with torch.no_grad():original=model(*inputs).detach().clone()
    if list(original.shape)!=OUTPUT_SHAPE or not torch.isfinite(original).all():raise ValueError('Invalid full FP32 reference')
    print('Original FP32 full velocity reference: finite',flush=True)
    model=convert_storage16(model)
    with torch.no_grad():expected=model(*inputs).detach()
    comparison=precision_metrics(expected,original)
    print('Stored FP16 / FP32 math versus original:',comparison,flush=True)
    if expected.dtype!=torch.float32 or not comparison['parity_passed']:
        raise ValueError('Full denoiser does not retain the original FP32 parity bound; baselines preserved')
    stage_root=ROOT/'.runtime-cache/coreai-denoiser-staging';stage_root.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='denoiser-',dir=stage_root))
    print('Staging (retained on failure):',stage,flush=True)
    for name,value in values.items():value.numpy().astype('<f4',copy=False).tofile(stage/f'{name}.f32le')
    expected.numpy().astype('<f4',copy=False).tofile(stage/'velocity-reference.f32le')
    original.numpy().astype('<f4',copy=False).tofile(stage/'velocity-fp32-reference.f32le')
    print('Tracing and exporting the COMPLETE 24-layer denoiser...',flush=True)
    exported=torch.export.export(model,inputs).run_decompositions(get_decomp_table())
    program=TorchConverter().add_exported_program(exported,input_names=list(INPUT_SHAPES),output_names=['velocity']).to_coreai()
    program.save_asset(stage/'denoiser.aimodel')
    ref={'schema_version':1,'profile':'denoiser','asset_file':'denoiser.aimodel','function_name':'main','output_name':'velocity',
         'input_shapes':INPUT_SHAPES,'output_shape':OUTPUT_SHAPE,'scalar_type':'float32','byte_order':'little',
         'layer_count':24,'checkpoint_prefix':'decoder.','latent_frames':25,'latent_rate_hz':25,'patch_size':2,
         'parameter_count':sum(p.numel() for p in model.parameters()),'weight_storage_dtype':'float16','compute_dtype':'float32',
         'original_fp32_reference_file':'velocity-fp32-reference.f32le','storage16_vs_original_fp32':comparison,
         'checkpoint':{k:record[k] for k in ['repo_id','revision','filename','size','digest']},'checkpoint_full_sha256_verified':True,
         'publisher_source_digests':{n:publisher[n]['digest'] for n in paths},'reference_backend':'PyTorch CPU Float32 / publisher eager attention',
         'input_provenance':'Seed-0 synthetic noise/context/condition; timestep=0.5, timestep_r=0.25; not recorded audio or complete conditioning',
         'scope':'Complete 24-layer velocity forward with projections/time embeddings, odd-length pad/crop; no diffusion loop, CFG or generated instrument',
         'reference_peak':float(expected.abs().max()),'reference_rms':float(np.sqrt(np.mean(expected.numpy().astype(np.float64)**2))),
         'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
         'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','coreai-torch','coreai-core']},
         'artifact_sha256':{str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()},
         'coreai_device_inference_verified':False,'runtime_ram_savings_verified':False}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n')
    verify_bundle(stage,record)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(f'Destination appeared; preserve staging {stage}')
    shutil.move(str(stage),str(args.out))
    print('Prepared full pretrained denoiser:',args.out,flush=True)
    print('iPhone load, dual parity, peak memory and latency: NOT YET VERIFIED',flush=True)


if __name__=='__main__':main()
