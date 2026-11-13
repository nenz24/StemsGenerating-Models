"""Stdlib-only local dependency proof for the pinned FP32 source encoder."""
from coreai_drum_reference30_window_contract import VAE_SHA, strict_metric
from package_coreai_drum_reference import ACCEPTED_REPORT_SHA256
import re

PROFILE='source-reference30-window-encoder'
BUNDLE='source-reference30-window-encoder-256'
WINDOW=256
CORE=224
HALO=16
RATIO=1920
INPUT_SHAPES={'audio':[1,2,WINDOW*RATIO],'epsilon':[1,64,WINDOW],'chunk_mask':[1,WINDOW,64]}
OUTPUT_SHAPES={'posterior_mean':[1,64,WINDOW],'posterior_std':[1,64,WINDOW],
               'source_latents':[1,WINDOW,64],'context_latents':[1,WINDOW,128]}
GOLD_FILES={'posterior_mean':'source-mean.f32le','posterior_std':'source-std.f32le',
            'source_latents':'source-latent.f32le','context_latents':'source-context.f32le'}
DATA_SHAPES={'source48.f32le':[1,2,750*RATIO],'source-epsilon.f32le':[1,64,750],
             'source-mask.f32le':[1,750,64],'source-mean.f32le':[1,64,750],
             'source-std.f32le':[1,64,750],'source-latent.f32le':[1,750,64],
             'source-context.f32le':[1,750,128]}
NATIVE_FILES=('main.hash','main.mlirb','metadata.json')
FILES=set(DATA_SHAPES)|{'capture-report.json','original-source-reference.json'}|{
    'source-context.aimodel/'+name for name in NATIVE_FILES}


def tile_plan(total_frames,window_frames=WINDOW):
    if window_frames not in (128,256) or type(total_frames) is not int or total_frames<window_frames:
        raise ValueError('Source requires supported128/256 window and enough global frames')
    core=window_frames-2*HALO
    rows=[]
    starts=[0] if total_frames==window_frames else range(0,total_frames,core)
    for start in starts:
        window=min(max(start-HALO,0),total_frames-window_frames)
        rows.append({'window_start':window,'window_frames':window_frames,'core_start':start,
                     'core_end':total_frames if total_frames==window_frames else min(start+core,total_frames),
                     'crop_start':start-window})
    return rows


def dependency_bounds(tile,total_frames):
    """Reverse every Conv1d, clipping only at genuine global clip boundaries."""
    width=tile['window_frames']
    if (width not in (128,256) or tile['core_end']>total_frames or tile['core_start']<0
            or tile['core_end']<=tile['core_start'] or tile['window_start']<0
            or tile['window_start']+width>total_frames or tile['crop_start']!=tile['core_start']-tile['window_start']):
        raise ValueError('Invalid source tile')
    size=width
    lo,hi=tile['crop_start'],tile['crop_start']+tile['core_end']-tile['core_start']-1
    def clip(a,b,length):
        if a<0 and tile['window_start']!=0:
            raise ValueError('Artificial left encoder padding reaches retained core')
        if b>=length and tile['window_start']+width!=total_frames:
            raise ValueError('Artificial right encoder padding reaches retained core')
        return max(0,a),min(length-1,b)
    lo,hi=clip(lo-1,hi+1,size) # final k3 conv
    for stride in (10,6,4,4,2):
        size*=stride
        lo,hi=clip(lo*stride-stride//2,hi*stride-stride//2+2*stride-1,size)
        for dilation in (9,3,1):
            lo,hi=clip(lo-3*dilation,hi+3*dilation,size)
    lo,hi=clip(lo-3,hi+3,size) # initial k7 conv
    return [tile['window_start']*RATIO+lo,tile['window_start']*RATIO+hi]


def validate_host_checks(checks):
    names={f'core-{i}-{name}' for i in range(4) for name in GOLD_FILES}|{
        f'seam-{i}-{name}' for i in range(1,4) for name in GOLD_FILES}|{'overall-'+name for name in GOLD_FILES}
    if set(checks)!=names:raise ValueError('Incomplete original source host gates')
    for metric in checks.values():strict_metric(metric)


def validate_reference(ref):
    expected={'schema_version':1,'profile':PROFILE,'status':'prepared_source_window_encoder',
              'asset_file':'source-context.aimodel','function_name':'main','latent_frames':750,
              'window_frames':WINDOW,'core_frames':CORE,'halo_frames':HALO,
              'input_shapes':INPUT_SHAPES,'output_shapes':OUTPUT_SHAPES,
              'vae_checkpoint_sha256':VAE_SHA,'capture_report_sha256':ACCEPTED_REPORT_SHA256,
              'weight_storage_dtype':'float32','compute_dtype':'float32','scalar_type':'float32','byte_order':'little',
              'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
              'window_plan':tile_plan(750),'dependency_sample_bounds':[dependency_bounds(t,750) for t in tile_plan(750)],
              'host_numerical_gate_passed':True,'coverage_exactly_once':True,'architecture_guard_passed':True,
              'epsilon_regenerated':False,'native_exports':0,'baseline_modified':False,
              'native_parity':'NOT EVALUATED','native_memory_gate':'NOT EVALUATED','full_song_native_readiness':'NOT EVALUATED'}
    if any(ref.get(k)!=v for k,v in expected.items()):raise ValueError('Source window contract mismatch')
    hashes=ref.get('artifact_sha256',{})
    if (set(hashes)!=FILES or hashes['capture-report.json']!=ACCEPTED_REPORT_SHA256
            or any(not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v) for v in hashes.values())
            or hashes['original-source-reference.json']!=ref.get('original_source_reference_sha256')):
        raise ValueError('Source window artifact identity mismatch')
    validate_host_checks(ref.get('host_checks',{}))
    inputs=ref.get('tile_input_sha256',[])
    if (len(inputs)!=4 or any(set(item)!=set(INPUT_SHAPES) or
            any(not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v) for v in item.values()) for item in inputs)):
        raise ValueError('Missing source tile input identities')
    return ref
