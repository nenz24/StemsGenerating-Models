#!/usr/bin/env python3
"""Export actual Qwen3 + ACE ConditionEncoder in bounded sequential assets.

Tokenization is host-side for one fixed bass prompt, not arbitrary prompt entry
on iPhone. Every neural computation is exported, including both embedding-table
lookups, all 28 Qwen blocks and all 12 ACE lyric/timbre blocks. No music output.
"""
import argparse
import ast
import __future__
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile

from export_coreai_decoder import ROOT, checkpoint_path, sha256
from export_coreai_dit_block import records, verified_source, convert_storage16, precision_metrics
from export_coreai_denoiser import load_full_upstream
from export_coreai_source_context import verified_handler_source

QWEN_DIGEST='0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
BASE_DIGEST='4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b'
SILENCE_DIGEST='a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b'
PROFILE='bass-conditioning'


def stage_plan(text_tokens,lyric_tokens):
    if not (1<=text_tokens<=256 and 1<=lyric_tokens<=2048):raise ValueError('Invalid token lengths')
    plan=[]
    def add(asset,inputs,outputs,bindings=None):
        plan.append({'asset_file':asset+'.aimodel','input_shapes':inputs,'output_shapes':outputs,
                     'output_bindings':bindings or {k:k for k in outputs}})
    add('token-embedding',{'text_ids':[1,text_tokens],'lyric_ids':[1,lyric_tokens]},
        {'text_hidden':[1,text_tokens,1024],'lyric_hidden':[1,lyric_tokens,1024]})
    for i in range(0,28,2):
        add(f'qwen-{i:02d}-{i+1:02d}',{'text_hidden':[1,text_tokens,1024]},
            {'hidden_out':[1,text_tokens,1024]},{'hidden_out':'text_hidden'})
    add('qwen-norm',{'text_hidden':[1,text_tokens,1024]}, {'text_embeddings':[1,text_tokens,1024]})
    add('lyric-front',{'lyric_hidden':[1,lyric_tokens,1024]}, {'hidden_out':[1,lyric_tokens,2048]},{'hidden_out':'lyric_hidden'})
    for i in range(0,8,2):
        add(f'lyric-{i:02d}-{i+1:02d}',{'lyric_hidden':[1,lyric_tokens,2048]},
            {'hidden_out':[1,lyric_tokens,2048]},{'hidden_out':'lyric_hidden'})
    add('lyric-norm',{'lyric_hidden':[1,lyric_tokens,2048]}, {'lyric_encoded':[1,lyric_tokens,2048]})
    add('timbre-front',{'reference_audio':[1,750,64]}, {'timbre_hidden':[1,750,2048]})
    for i in range(0,4,2):
        add(f'timbre-{i:02d}-{i+1:02d}',{'timbre_hidden':[1,750,2048]},
            {'hidden_out':[1,750,2048]},{'hidden_out':'timbre_hidden'})
    add('timbre-norm',{'timbre_hidden':[1,750,2048]}, {'timbre_encoded':[1,1,2048]})
    add('condition-pack',{'text_embeddings':[1,text_tokens,1024],'lyric_encoded':[1,lyric_tokens,2048],'timbre_encoded':[1,1,2048]},
        {'condition':[1,text_tokens+lyric_tokens+1,2048],'condition_mask':[1,text_tokens+lyric_tokens+1]})
    return plan


def format_prompts():
    """Execute pinned formatting methods/constants, retaining empty baseline lyrics."""
    constants_path,_=verified_handler_source('constants.py')
    constants={}
    exec(compile(ast.parse(constants_path.read_text()),str(constants_path),'exec'),constants)
    ns={}
    for name,kind,methods in [('prompt_utils.py','PromptMixin',{'_format_instruction','_format_lyrics'}),
                              ('metadata_utils.py','MetadataMixin',{'_build_metadata_dict','_dict_to_meta_string'})]:
        path,_=verified_handler_source(name);tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==kind)
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
        if len(cls.body)!=len(methods):raise ValueError('Pinned formatter missing')
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns)
    prompt=ns['PromptMixin']();meta=ns['MetadataMixin']()
    preset=json.loads((ROOT/'configs/electric-bass.json').read_text())
    instruction=prompt._format_instruction('Generate the bass track based on the audio context:')
    metadata=meta._dict_to_meta_string(meta._build_metadata_dict(None,'','',5.12))
    caption_prompt=constants['SFT_GEN_PROMPT'].format(instruction,preset['caption'],metadata)
    lyrics_prompt=prompt._format_lyrics('','unknown')
    return {'instruction':instruction,'caption':preset['caption'],'metadata':metadata,
            'text_prompt':caption_prompt,'lyric_prompt':lyrics_prompt,'lyrics':'','vocal_language':'unknown',
            'preset_sha256':sha256(ROOT/'configs/electric-bass.json'),'audio_seconds':5.12}


def load_condition_encoder(paths):
    """Load actual pinned classes, including packing/unpacking, for host reference."""
    cfg,_,_=load_full_upstream(paths)
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.utils import can_return_tuple
    import sys
    ns=sys.modules['coreai_pinned_base_layers']
    ns.__dict__.update(BaseModelOutput=BaseModelOutput,can_return_tuple=can_return_tuple)
    names={'AceStepEncoderLayer','AceStepLyricEncoder','AceStepTimbreEncoder','AceStepConditionEncoder','pack_sequences'}
    tree=ast.parse(paths['modeling_acestep_v15_base.py'].read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if len(nodes)!=len(names):raise ValueError('ConditionEncoder source missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(paths['modeling_acestep_v15_base.py']),'exec',
                 flags=__future__.annotations.compiler_flag),ns.__dict__)
    with torch.device('meta'):model=ns.AceStepConditionEncoder(cfg).eval()
    return cfg,model,ns.create_4d_mask


def verify_bundle(folder):
    ref=json.loads((folder/'reference.json').read_text()); nt=ref.get('text_tokens',0);nl=ref.get('lyric_tokens',0)
    plan=stage_plan(nt,nl)
    if (ref.get('schema_version')!=1 or ref.get('profile')!=PROFILE or ref.get('function_name')!='main'
            or nt!=105 or nl!=11
            or ref.get('stages')!=plan or ref.get('qwen_layer_count')!=28 or ref.get('lyric_layer_count')!=8
            or ref.get('timbre_layer_count')!=4 or ref.get('batch_size')!=1
            or ref.get('qwen_checkpoint_digest')!=QWEN_DIGEST or ref.get('base_checkpoint_digest')!=BASE_DIGEST
            or ref.get('silence_digest')!=SILENCE_DIGEST or ref.get('all_checkpoint_hashes_verified') is not True
            or ref.get('scalar_type')!='float32' or ref.get('byte_order')!='little'
            or ref.get('weight_storage_dtype')!='float16' or ref.get('compute_dtype')!='float32'
            or ref.get('absolute_tolerance')!=1e-4 or ref.get('relative_tolerance')!=1e-3
            or ref.get('relative_rmse_tolerance')!=1e-3 or ref.get('all_valid_single_reference_specialization') is not True
            or ref.get('reference_audio_mode')!='publisher_silence_750' or ref.get('tokenization_on_device') is not False
            or ref.get('host_partition_vs_original_fp32',{}).get('parity_passed') is not True
            or ref.get('host_storage16_vs_original_fp32',{}).get('parity_passed') is not True
            or ref.get('prompts')!=format_prompts()):
        raise ValueError('Unexpected bass-conditioning contract; preserve bundle')
    shapes={**plan[0]['input_shapes'],'reference_audio':[1,750,64]}
    required={f'{n}.f32le':s for n,s in shapes.items()}
    for n,s in {'text_embeddings':[1,nt,1024],**plan[-1]['output_shapes']}.items():
        for kind in ['reference','fp32-reference']:required[f'{n}-{kind}.f32le']=s
    native={f'{s["asset_file"]}/{n}' for s in plan for n in ['main.hash','main.mlirb','metadata.json']}
    hashes=ref['artifact_sha256'];actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='.DS_Store' and p!=folder/'reference.json'}
    if not (set(required)|native).issubset(hashes) or set(hashes)!=actual:raise ValueError('Incomplete conditioning bundle')
    for n,digest in hashes.items():
        path=(folder/n).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path)!=digest:raise ValueError(f'Conditioning hash mismatch: {n}')
    for n,shape in required.items():
        data=(folder/n).read_bytes()
        if len(data)!=4*math.prod(shape):raise ValueError(f'Conditioning binary size mismatch: {n}')
        vals=[v[0] for v in struct.iter_unpack('<f',data)]
        if not all(math.isfinite(v) for v in vals):raise ValueError(f'Nonfinite conditioning input: {n}')
        if n.endswith('_ids.f32le') and not all(v==int(v) and 0<=v<151669 for v in vals):raise ValueError('Invalid token ID')
        if n.startswith('condition_mask') and not all(v==1 for v in vals):raise ValueError('Expected all-valid mask')
    return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'prototypes/ios27-coreai/ModelCandidates/bass-conditioning')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:
        verify_bundle(args.out);print('Bass-conditioning bundle VERIFIED; no re-export');return
    if args.out.exists():raise FileExistsError(f'Output already exists: {args.out}; preserved')
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer,Qwen3Config,Qwen3Model
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from coreai_torch import TorchConverter,get_decomp_table
    torch.set_num_threads(2)
    manifest=json.loads((ROOT/'configs/assets-base.json').read_text())
    qrecords={r['filename']:r for r in manifest['files'] if r['filename'].startswith('Qwen3-Embedding-0.6B/')}
    source_dir=ROOT/'.runtime-cache/coreai-bass-conditioning-sources';(source_dir/'Qwen3-Embedding-0.6B').mkdir(parents=True,exist_ok=True)
    for n in ['config.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json']:
        verified_source(qrecords['Qwen3-Embedding-0.6B/'+n],source_dir)
    tokenizer_record=qrecords['Qwen3-Embedding-0.6B/tokenizer.json'];tok_path=checkpoint_path(tokenizer_record)
    tok_destination=source_dir/tokenizer_record['filename']
    if tok_destination.exists():
        if sha256(tok_destination)!=tokenizer_record['digest']:raise ValueError('Existing tokenizer mismatch; preserved')
    else:shutil.copyfile(tok_path,tok_destination)
    tokenizer=AutoTokenizer.from_pretrained(source_dir/'Qwen3-Embedding-0.6B',local_files_only=True)
    prompts=format_prompts()
    text=tokenizer(prompts['text_prompt'],padding='longest',truncation=True,max_length=256,return_tensors='pt')
    lyric=tokenizer(prompts['lyric_prompt'],padding='longest',truncation=True,max_length=2048,return_tensors='pt')
    if not text.attention_mask.all() or not lyric.attention_mask.all():raise ValueError('Single-example all-valid specialization only')
    nt,nl=text.input_ids.shape[1],lyric.input_ids.shape[1];plan=stage_plan(nt,nl)
    print(f'Bass conditioning: {nt} text / {nl} lyric tokens; {len(plan)} assets; all 40 transformer layers.',flush=True)
    qpath=checkpoint_path(qrecords['Qwen3-Embedding-0.6B/model.safetensors'])
    publisher=records();base_path=checkpoint_path(publisher['model.safetensors']);silence_path=checkpoint_path(publisher['silence_latent.pt'])
    if qrecords['Qwen3-Embedding-0.6B/model.safetensors']['digest']!=QWEN_DIGEST or publisher['model.safetensors']['digest']!=BASE_DIGEST:raise ValueError('Unexpected checkpoints')
    qcfg=Qwen3Config(**json.loads((source_dir/'Qwen3-Embedding-0.6B/config.json').read_text()));qcfg._attn_implementation='eager';qcfg.use_cache=False
    with torch.device('meta'):qwen=Qwen3Model(qcfg).eval()
    with safe_open(str(qpath),framework='pt',device='cpu') as w:qstate={k:w.get_tensor(k) for k in w.keys()}
    qwen.load_state_dict(qstate,strict=True,assign=True);del qstate
    # Rotary frequencies are analytical, nonpersistent buffers, not checkpoint
    # parameters. Recreate them on CPU after constructing learned modules on meta.
    qwen.rotary_emb=Qwen3RotaryEmbedding(config=qcfg,device='cpu')
    qwen.to(device='cpu',dtype=torch.float32).requires_grad_(False)
    publisher_source_dir=ROOT/'.runtime-cache/coreai-generator-source-publisher'
    publisher_source_dir.mkdir(parents=True,exist_ok=True)
    paths={n:verified_source(publisher[n],publisher_source_dir) for n in ['modeling_acestep_v15_base.py','configuration_acestep_v15.py','config.json']}
    cfg,encoder,create_mask=load_condition_encoder(paths)
    with safe_open(str(base_path),framework='pt',device='cpu') as w:estate={k[len('encoder.'):]:w.get_tensor(k) for k in w.keys() if k.startswith('encoder.')}
    encoder.load_state_dict(estate,strict=True,assign=True);del estate
    encoder.lyric_encoder.rotary_emb=Qwen3RotaryEmbedding(config=cfg,device='cpu')
    encoder.timbre_encoder.rotary_emb=Qwen3RotaryEmbedding(config=cfg,device='cpu')
    encoder.to(device='cpu',dtype=torch.float32).requires_grad_(False)
    # The pinned loader transposes publisher [B,C,T] before infer_refer_latent
    # selects [:,:750,:] for the no-reference branch.
    silence=torch.load(silence_path,map_location='cpu',weights_only=True).transpose(1,2).float()
    reference_audio=silence[:,:750,:].contiguous()
    if list(reference_audio.shape)!=[1,750,64]:raise ValueError('Unexpected publisher silence latent shape')
    # Original full implementations: no partitioning or replacement for the gold.
    with torch.no_grad():
        gold_text=qwen(input_ids=text.input_ids,lyric_attention_mask=None).last_hidden_state
        gold_lyric=qwen.embed_tokens(lyric.input_ids)
        gold_cond,gold_mask=encoder(text_hidden_states=gold_text,text_attention_mask=text.attention_mask.bool(),
            lyric_hidden_states=gold_lyric,lyric_attention_mask=lyric.attention_mask.bool(),
            refer_audio_acoustic_hidden_states_packed=reference_audio,refer_audio_order_mask=torch.tensor([0]))
    print('Original Qwen + complete ConditionEncoder gold computed.',flush=True)
    class Embedding(torch.nn.Module):
        def __init__(self):super().__init__();self.embedding=qwen.embed_tokens
        def forward(self,text_ids,lyric_ids):
            return (torch.nn.functional.embedding(text_ids.long(),self.embedding.weight.float()),
                    torch.nn.functional.embedding(lyric_ids.long(),self.embedding.weight.float()))
    class Unary(torch.nn.Module):
        def __init__(self,layer,first=False):super().__init__();self.layer=layer;self.first=first
        def forward(self,hidden):
            x=self.layer(hidden)
            return x[:,:1,:] if self.first else x
    class Pair(torch.nn.Module):
        def __init__(self,layers,rotary,length,qwen_kind):
            super().__init__();self.layers=torch.nn.ModuleList(layers);self.qwen_kind=qwen_kind
            hidden=torch.zeros(1,length,1024 if qwen_kind else 2048)
            positions=torch.arange(length).unsqueeze(0)
            cos,sin=rotary(hidden,positions)
            self.register_buffer('cos',cos);self.register_buffer('sin',sin)
            self.register_buffer('positions',positions)
            self.register_buffer('full_mask',create_mask(length,torch.float32,'cpu',is_causal=qwen_kind))
            self.register_buffer('local_mask',create_mask(length,torch.float32,'cpu',sliding_window=128,is_sliding_window=True,is_causal=False))
        def forward(self,hidden):
            for layer in self.layers:
                if self.qwen_kind:
                    hidden=layer(hidden_states=hidden,position_embeddings=(self.cos,self.sin),attention_mask=self.full_mask,use_cache=False)
                else:
                    mask=self.local_mask if layer.attention_type=='sliding_attention' else self.full_mask
                    hidden=layer(hidden,(self.cos,self.sin),mask,self.positions)[0]
            return hidden
    class Pack(torch.nn.Module):
        def __init__(self):super().__init__();self.projector=encoder.text_projector
        def forward(self,text_embeddings,lyric_encoded,timbre_encoded):
            condition=torch.cat([lyric_encoded,timbre_encoded,self.projector(text_embeddings)],dim=1)
            return condition,torch.ones((1,nt+nl+1),dtype=torch.float32)
    modules=[Embedding()]
    modules += [Pair(list(qwen.layers[i:i+2]),qwen.rotary_emb,nt,True) for i in range(0,28,2)]
    modules += [Unary(qwen.norm),Unary(encoder.lyric_encoder.embed_tokens)]
    modules += [Pair(list(encoder.lyric_encoder.layers[i:i+2]),encoder.lyric_encoder.rotary_emb,nl,False) for i in range(0,8,2)]
    modules += [Unary(encoder.lyric_encoder.norm),Unary(encoder.timbre_encoder.embed_tokens)]
    modules += [Pair(list(encoder.timbre_encoder.layers[i:i+2]),encoder.timbre_encoder.rotary_emb,750,False) for i in range(0,4,2)]
    modules += [Unary(encoder.timbre_encoder.norm,first=True),Pack()]
    assert len(modules)==len(plan)
    initial={'text_ids':text.input_ids.float(),'lyric_ids':lyric.input_ids.float(),'reference_audio':reference_audio}
    def forward_pipeline():
        state=dict(initial)
        with torch.no_grad():
            for module,s in zip(modules,plan):
                args=tuple(state[n] for n in s['input_shapes']);result=module(*args)
                if not isinstance(result,tuple):result=(result,)
                for (name,shape),value in zip(s['output_shapes'].items(),result):
                    if list(value.shape)!=shape:raise ValueError('Stage output shape mismatch')
                    state[s['output_bindings'][name]]=value.contiguous()
        return {k:state[k] for k in ['text_embeddings','condition','condition_mask']}
    partition=forward_pipeline();gold={'text_embeddings':gold_text,'condition':gold_cond,'condition_mask':gold_mask.float()}
    part_metrics={n:precision_metrics(partition[n],gold[n]) for n in gold}
    if not all(v['parity_passed'] for v in part_metrics.values()):raise ValueError(f'FP32 partition parity failed: {part_metrics}')
    for module in modules:convert_storage16(module);module.eval().requires_grad_(False)
    storage=forward_pipeline();storage_metrics={n:precision_metrics(storage[n],gold[n]) for n in gold}
    if not all(v['parity_passed'] for v in storage_metrics.values()):raise ValueError(f'Storage16 whole-pipeline drift: {storage_metrics}')
    print('Partition and storage16 dual host parity PASS; exporting native assets.',flush=True)
    parent=ROOT/'.runtime-cache/coreai-bass-conditioning-staging';parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='bass-',dir=parent))
    for n,v in initial.items():v.detach().numpy().astype('<f4').tofile(stage/f'{n}.f32le')
    for n in gold:
        gold[n].detach().numpy().astype('<f4').tofile(stage/f'{n}-fp32-reference.f32le')
        storage[n].detach().numpy().astype('<f4').tofile(stage/f'{n}-reference.f32le')
    state=dict(initial)
    for i,(module,s) in enumerate(zip(modules,plan),1):
        print(f'Export {i}/{len(plan)}: {s["asset_file"]}',flush=True)
        args_tuple=tuple(state[n] for n in s['input_shapes'])
        with torch.no_grad():out=module(*args_tuple)
        if not isinstance(out,tuple):out=(out,)
        exported=torch.export.export(module,args_tuple).run_decompositions(get_decomp_table())
        program=TorchConverter().add_exported_program(exported,input_names=list(s['input_shapes']),output_names=list(s['output_shapes'])).to_coreai()
        program.save_asset(stage/s['asset_file'])
        for (name,shape),v in zip(s['output_shapes'].items(),out):state[s['output_bindings'][name]]=v.contiguous()
        del exported,program;gc.collect()
    hashes={str(p.relative_to(stage)):sha256(p) for p in stage.rglob('*') if p.is_file()}
    ref={'schema_version':1,'profile':PROFILE,'function_name':'main','batch_size':1,'text_tokens':nt,'lyric_tokens':nl,
         'qwen_layer_count':28,'lyric_layer_count':8,'timbre_layer_count':4,'stages':plan,'scalar_type':'float32','byte_order':'little',
         'qwen_checkpoint_digest':QWEN_DIGEST,'base_checkpoint_digest':BASE_DIGEST,'silence_digest':SILENCE_DIGEST,
         'all_checkpoint_hashes_verified':True,'weight_storage_dtype':'float16','compute_dtype':'float32',
         'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
         'prompts':prompts,'tokenization_on_device':False,'reference_audio_mode':'publisher_silence_750',
         'all_valid_single_reference_specialization':True,'host_partition_vs_original_fp32':{'parity_passed':True,'outputs':part_metrics},
         'host_storage16_vs_original_fp32':{'parity_passed':True,'outputs':storage_metrics},'artifact_sha256':hashes,
         'native_asset_bytes':sum(p.stat().st_size for p in stage.rglob('*') if p.is_file() and '.aimodel/' in str(p)),
         'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','coreai-torch','coreai-core']},
         'scope':'All neural text/lyric/timbre conditioning on device; fixed host-tokenized bass prompt; no source-context, denoiser, CFG or audio output'}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n');verify_bundle(stage)
    args.out.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(stage),str(args.out))
    print(f'Ready: {args.out}; {ref["native_asset_bytes"]/1e9:.3f} GB native assets. Rebuild app.',flush=True)


if __name__=='__main__':main()
