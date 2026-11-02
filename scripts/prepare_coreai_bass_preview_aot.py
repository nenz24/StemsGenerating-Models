#!/usr/bin/env python3
"""Compile existing bass preview assets for h18p with Apple's installed CLI.

No PyTorch export, reference recomputation or checkpoint download. Native AOT
compilation runs only on the Mac; package/command/integrity tests are not native
inference. Original model and reference bundles are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time

from export_coreai_decoder import ROOT,sha256
from export_coreai_bass_preview import stage_plan,verify_bundle as verify_preview

SOURCE=ROOT/'prototypes/ios27-coreai/ModelCandidates/bass-preview-512'
OUTPUT=ROOT/'prototypes/ios27-coreai/ModelCandidates/bass-preview-512-aot'


def asset_names():return [*[s['asset_file'] for s in stage_plan()],'apg-euler.aimodel','decoder.aimodel']


def expected_assets(architecture):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',architecture):raise ValueError('Invalid architecture identifier')
    return {name:f'{Path(name).stem}.{architecture}.aimodelc' for name in asset_names()}


def compile_command(source,output,architecture):
    # These flags and the value `none` come from the user's Metal 27B47 help.
    # There is NO `cpu`/`cpuOnly` preferred-compute value in that tool.
    expected_assets(architecture)
    return ['xcrun','coreai-build','compile',str(source),'--output',str(output),
            '--platform','iOS','--min-deployment-version','27.0','--preferred-compute','none',
            '--architecture',architecture]


def file_hashes(folder):
    folder=Path(folder).resolve();hashes={}
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.name=='.DS_Store':continue
        if not p.resolve().is_relative_to(folder):raise ValueError('External file link in compiled package')
        hashes[p.relative_to(folder).as_posix()]=sha256(p)
    return hashes


def payload_hashes(path):
    path=Path(path)
    if path.is_file():return {'@file':sha256(path)}
    if path.is_dir():return file_hashes(path)
    raise ValueError(f'Compiled artifact absent: {path}')


def standalone_result(path,work):
    """Dereference compiler links before hashing/copying into the app bundle."""
    if not path.exists():raise ValueError(f'Compiled artifact absent: {path}')
    linked=path.is_symlink() or (path.is_dir() and any(p.is_symlink() for p in path.rglob('*')))
    if not linked:return path
    parent=work/'standalone';parent.mkdir();destination=parent/path.name
    if path.is_dir():shutil.copytree(path,destination)
    else:shutil.copyfile(path,destination)
    return destination


def verified_work_result(work,identity):
    receipt_path=work/'receipt.json'
    if not receipt_path.exists():return None
    receipt=json.loads(receipt_path.read_text());path=(work/receipt.get('relative_path','')).resolve()
    if (receipt.get('identity')!=identity or not path.is_relative_to(work.resolve())
            or payload_hashes(path)!=receipt.get('payload_sha256')):
        raise ValueError(f'Existing AOT work cache changed: {work}; preserved')
    return path


def verify_aot_bundle(folder,source=SOURCE,architecture='h18p'):
    folder=Path(folder);ref=json.loads((folder/'reference.json').read_text())
    if (ref.get('schema_version')!=1 or ref.get('profile')!='bass-preview-512-aot'
            or ref.get('architecture')!=architecture or ref.get('platform')!='iOS'
            or ref.get('min_deployment_version')!='27.0' or ref.get('preferred_compute')!='none'
            or ref.get('runtime_policy')!='default' or ref.get('assets')!=expected_assets(architecture)
            or ref.get('source_reference_sha256')!=sha256(Path(source)/'reference.json')
            or ref.get('compiled_asset_count')!=16):
        raise ValueError('AOT contract/architecture/source mismatch; original assets preserved')
    actual=file_hashes(folder);actual.pop('reference.json',None)
    hashes=ref.get('artifact_sha256',{})
    if not hashes or actual!=hashes:raise ValueError('AOT package integrity mismatch')
    for name in ref['assets'].values():
        p=folder/name
        if not p.exists() or not payload_hashes(p):raise ValueError('Missing/empty compiled variant')
    if any(not any(n==v or n.startswith(v+'/') for v in ref['assets'].values()) for n in actual):
        raise ValueError('Unexpected file in AOT package')
    return ref


def publish_bundle(compiled,source,out,architecture,compiler):
    """Package actual CLI results; never creates an executable/native fixture."""
    if out.exists():raise FileExistsError(f'Existing AOT output preserved: {out}')
    expected=expected_assets(architecture)
    if set(compiled)!=set(expected):raise ValueError('Incomplete set of compiled models')
    staging=ROOT/'.runtime-cache/coreai-bass-preview-aot-staging';staging.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='package-',dir=staging))
    for name,variant in expected.items():
        path=Path(compiled[name]);destination=stage/variant
        if path.is_dir():shutil.copytree(path,destination)
        elif path.is_file():shutil.copyfile(path,destination)
        else:raise ValueError(f'Missing compiler result: {name}')
    hashes=file_hashes(stage)
    ref={'schema_version':1,'profile':'bass-preview-512-aot','architecture':architecture,'platform':'iOS',
         'min_deployment_version':'27.0','preferred_compute':'none','runtime_policy':'default',
         'source_reference_sha256':sha256(source/'reference.json'),'assets':expected,'compiled_asset_count':16,
         'artifact_sha256':hashes,'compiler':compiler,
         'compiled_bytes':sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()),
         'scope':'Mac AOT only. iPhone specialization/inference, numerical parity, actual placement and memory not measured.'}
    (stage/'reference.json').write_text(json.dumps(ref,indent=2)+'\n')
    verify_aot_bundle(stage,source,architecture)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(f'Destination appeared; preserve staging at {stage}')
    shutil.move(str(stage),str(out));return ref


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bundle',type=Path,default=SOURCE)
    parser.add_argument('--out',type=Path,default=OUTPUT)
    parser.add_argument('--architecture',default='h18p')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();expected=expected_assets(args.architecture)
    if args.verify_only or args.out.exists():
        try:ref=verify_aot_bundle(args.out,args.source_bundle,args.architecture)
        except Exception as error:
            raise ValueError(f'Existing AOT output invalid or stale: {args.out}; preserved. Use a new --out or move the old bundle outside app resources.') from error
        print('AOT package integrity VERIFIED; existing assets retained; architecture:',ref['architecture'])
        print('Explicit bass-aot diagnostic uses runtime default (compiler preference none); auto stays source CPU-only after the reported Metal compiler crash.');return
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        parser.error('Run native compilation on an Apple Silicon Mac with installed Metal Toolchain; no Linux compiler emulation')
    if not (args.source_bundle/'reference.json').is_file():
        parser.error('Existing bass-preview-512 bundle required; run prepare_coreai_bass_preview.sh first')
    verify_preview(args.source_bundle)
    find=subprocess.run(['xcrun','--find','coreai-build'],capture_output=True,text=True)
    if find.returncode:
        raise RuntimeError('coreai-build unavailable. Install Metal Toolchain using xcodebuild -downloadComponent MetalToolchain')
    tool=Path(find.stdout.strip()).resolve()
    help_result=subprocess.run(['xcrun','coreai-build','compile','--help'],capture_output=True,text=True,check=True)
    help_text=help_result.stdout+help_result.stderr
    for option in ['--platform','--min-deployment-version','--preferred-compute','--architecture','--output']:
        if option not in help_text:raise RuntimeError(f'Installed tool missing {option}; send compile --help output')
    if not re.search(r'values:\s*gpu,\s*neural-engine,\s*none',help_text):
        raise RuntimeError('Compute choices changed; send current compile --help; no guessed compute policy')
    version=subprocess.run(['xcodebuild','-version'],capture_output=True,text=True,check=True).stdout.strip()
    compiler={'executable_sha256':sha256(tool),'help_sha256':hashlib.sha256(help_text.encode()).hexdigest(),
              'xcode_version':version,'preferred_compute':'none','native_compilation_invoked':True}
    identity=hashlib.sha256(json.dumps({'source':sha256(args.source_bundle/'reference.json'),'compiler':compiler,
                                      'architecture':args.architecture},sort_keys=True).encode()).hexdigest()
    root=ROOT/'.runtime-cache/coreai-bass-preview-aot-work'/identity[:24];root.mkdir(parents=True,exist_ok=True)
    results={}
    for index,(name,variant) in enumerate(expected.items(),1):
        work=root/Path(name).stem;work.mkdir(exist_ok=True);receipt_path=work/'receipt.json'
        path=verified_work_result(work,identity)
        if path is not None:
            print(f'AOT {index}/16 {name}: VERIFIED cached compiler result',flush=True)
        else:
            attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=work));output=attempt/'output';output.mkdir()
            print(f'AOT {index}/16 {name} -> {variant}; architecture {args.architecture}, compute preference none',flush=True)
            print('Work directory retained on failure:',attempt,flush=True);started=time.monotonic()
            subprocess.run(compile_command(args.source_bundle/name,output,args.architecture),check=True)
            path=standalone_result(output/variant,attempt)
            hashes=payload_hashes(path)
            if not hashes:raise ValueError(f'Compiler returned an empty artifact: {path}')
            receipt={'identity':identity,'relative_path':str(path.relative_to(work)),'payload_sha256':hashes}
            temp=work/'receipt.pending.json';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(receipt_path)
            print(f'AOT {index}/16 complete in {time.monotonic()-started:.1f} seconds',flush=True)
        results[name]=path
    ref=publish_bundle(results,args.source_bundle,args.out,args.architecture,compiler)
    print(f'Ready: {args.out}; {ref["compiled_bytes"]/1e9:.3f} GB compiled; only {args.architecture}',flush=True)
    print('AOT retained on Mac. Auto app profile uses source CPU-only after the reported Metal compiler crash; bass-aot remains an explicit diagnostic. Device validation pending.')


if __name__=='__main__':main()
