"""CPU-only transfer of publisher-verified ACE-Step assets between Colab VMs."""
import argparse
from contextlib import suppress
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from asset_progress import FileProgress
from model_assets import asset_path, verify

ROOT = Path(__file__).resolve().parents[1]
HEADER = 'cache_manifest.json'


def manifest_data(path):
    path=Path(path);data=json.loads(path.read_text());records={}
    for record in data['files']:
        name=record['destination'];relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or not name.startswith('checkpoints/') or name in records:
            raise ValueError('Invalid or duplicate asset destination')
        if record['size']<0 or record['algorithm'] not in ('sha256','git-blob-sha1'):
            raise ValueError('Invalid asset size/digest algorithm')
        records[name]=record
    return data,records,hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, manifest):
    _,records,digest=manifest_data(manifest);upstream=Path(root).resolve()/'vendor/ACE-Step-1.5'
    present=0;missing=[]
    for name,record in records.items():
        path=asset_path(upstream,name)
        if path.is_file() and path.stat().st_size==record['size']:present+=record['size']
        else:missing.append(name)
    total=sum(r['size'] for r in records.values())
    return {'manifest_sha256':digest,'total_bytes':total,'present_bytes_by_size':present,
            'missing_bytes':total-present,'missing_files':missing,'complete_by_size':not missing,
            'note':'Presence/size is not checksum verification; export/restore verifies publisher digests.'}


def discover(root_base, manifest, preferred=None):
    """Find existing assets without requiring CUDA, an active venv or a model import."""
    base=Path(root_base).resolve();candidates=set()
    if preferred is not None:
        preferred=Path(preferred).expanduser().resolve()
        if preferred.is_dir() and inventory(preferred,manifest)['present_bytes_by_size']>0:return preferred
    for parent in dict.fromkeys((base,base.parent)):
        for folder in (parent,parent/'source',parent/'StemsGenerating-Models',
                       *parent.glob('ace_context_source_*'),*parent.glob('songmap_cuda_*/source')):
            if (folder/'vendor/ACE-Step-1.5').is_dir():candidates.add(folder.resolve())
    ranked=[(inventory(p,manifest)['present_bytes_by_size'],p) for p in candidates]
    ranked=[(size,p) for size,p in ranked if size>0]
    if not ranked:return None
    best=max(size for size,p in ranked);choices=[p for size,p in ranked if size==best]
    if len(choices)!=1:raise ValueError('Several asset caches found; set ROOT_HINT to one of: '+', '.join(map(str,choices)))
    return choices[0]


class CheckedReader:
    def __init__(self, stream, record, progress=None):
        self.stream,self.record,self.progress=stream,record,progress;self.count=0
        self.hasher=hashlib.sha256() if record['algorithm']=='sha256' else hashlib.sha1()
        if record['algorithm']=='git-blob-sha1':self.hasher.update(f"blob {record['size']}\0".encode())

    def read(self, size=-1):
        data=self.stream.read(size);self.count+=len(data);self.hasher.update(data)
        if self.progress is not None:self.progress(self.count)
        return data

    def finish(self):
        if self.count!=self.record['size'] or self.hasher.hexdigest()!=self.record['digest']:
            raise ValueError('Transferred asset checksum mismatch: '+self.record['destination'])


def export(root, archive, manifest):
    root,archive=Path(root).resolve(),Path(archive).expanduser().resolve()
    data,records,digest=manifest_data(manifest)
    upstream=root/'vendor/ACE-Step-1.5';total=sum(r['size'] for r in records.values())
    if archive.exists():raise FileExistsError('Backup already exists; it is preserved: '+str(archive))
    for name,record in records.items():
        p=asset_path(upstream,name)
        if not p.is_file() or p.stat().st_size!=record['size']:
            raise ValueError('Old VM asset missing/wrong size: '+name+'. A CPU reconnection may be a new empty VM.')
    archive.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(archive.parent).free<total+1024**2:
        raise RuntimeError('Insufficient destination disk space for the weight archive')
    partial=archive.with_name(archive.name+'.partial')
    header={'schema_version':1,'model':data['model'],'manifest_sha256':digest,'total_bytes':total,'files':len(records)}
    # Exclusive partial file; a failed copy never becomes a completed backup.
    target=partial.open('xb')  # A pre-existing partial is never deleted or overwritten.
    try:
        with target,tarfile.open(fileobj=target,mode='w|') as tar:
            payload=(json.dumps(header,sort_keys=True)+'\n').encode();info=tarfile.TarInfo(HEADER);info.size=len(payload)
            tar.addfile(info,io.BytesIO(payload))
            for index,(name,record) in enumerate(records.items(),1):
                print(f'Backup [{index}/{len(records)}] {name}',flush=True)
                info=tarfile.TarInfo('weights/'+name);info.size=record['size'];info.mode=0o644
                with asset_path(upstream,name).open('rb') as source:
                    reader=CheckedReader(source,record,FileProgress('Backup + checksum',record['size']))
                    tar.addfile(info,reader);reader.finish()
    except BaseException:
        with suppress(OSError):partial.unlink()
        raise
    if archive.exists():raise FileExistsError('Destination appeared during export; completed partial is preserved')
    partial.rename(archive)
    return dict(header,status='exported',archive=str(archive),requires_gpu=False)


def restore(root, archive, manifest):
    root,archive=Path(root).resolve(),Path(archive).expanduser().resolve()
    data,records,digest=manifest_data(manifest);upstream=root/'vendor/ACE-Step-1.5'
    upstream.mkdir(parents=True,exist_ok=True)
    status=inventory(root,manifest)
    if shutil.disk_usage(upstream).free<status['missing_bytes']+1024**2:
        raise RuntimeError('Insufficient disk space for restored assets')
    seen=set();restored=0;reused=0
    # Stream named regular members only; never extract paths/links from an archive.
    with tarfile.open(archive,'r|*') as tar:
        members=iter(tar);first=next(members,None)
        if first is None or first.name!=HEADER or not first.isfile() or first.size>65536:
            raise ValueError('Not a supported ACE weight cache archive')
        header=json.loads(tar.extractfile(first).read())
        if header.get('schema_version')!=1 or header.get('model')!=data['model'] or header.get('manifest_sha256')!=digest:
            raise ValueError('Weight archive does not match the trusted current manifest')
        for member in members:
            name=member.name.removeprefix('weights/')
            if not member.name.startswith('weights/') or name not in records or name in seen or not member.isfile():
                raise ValueError('Unexpected, duplicate or non-regular weight archive member')
            record=records[name]
            if member.size!=record['size']:raise ValueError('Archive member size differs from trusted manifest')
            target=asset_path(upstream,name);target.parent.mkdir(parents=True,exist_ok=True)
            existing=target.exists()
            if existing:verify(target,record)
            reader=CheckedReader(tar.extractfile(member),record,FileProgress('Restore + checksum',record['size']))
            temporary=None
            try:
                if existing:
                    while reader.read(1024*1024):pass
                else:
                    with tempfile.NamedTemporaryFile(prefix=target.name+'.',suffix='.partial',dir=target.parent,delete=False) as output:
                        temporary=Path(output.name)
                        while True:
                            block=reader.read(1024*1024)
                            if not block:break
                            output.write(block)
                reader.finish()
                if temporary is not None:
                    if target.exists():raise FileExistsError('Asset appeared during restore; existing file preserved')
                    os.replace(temporary,target);restored+=1
                else:reused+=1
            except BaseException:
                if temporary is not None:
                    with suppress(OSError):temporary.unlink()
                raise
            seen.add(name)
    if seen!=set(records):raise ValueError('Incomplete weight archive; verified files are retained for a retry')
    return {'status':'restored','archive':str(archive),'manifest_sha256':digest,
            'restored_files':restored,'reused_files':reused,'total_bytes':sum(r['size'] for r in records.values()),'requires_gpu':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','export','restore'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--manifest',type=Path,default=ROOT/'configs/assets-base.json')
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args()
    if args.action=='inventory':result=inventory(args.root,args.manifest)
    else:
        if args.archive is None:parser.error('--archive is required')
        result=(export if args.action=='export' else restore)(args.root,args.archive,args.manifest)
    print(json.dumps(result,indent=2))
