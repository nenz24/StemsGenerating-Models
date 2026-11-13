"""Stdlib-only contract for native generation from host-prepared full-song inputs."""
import re
from prepare_coreai_full_song import schedule
from coreai_instrument_preview30_contract import PRESETS
from coreai_drum_reference30_contract import REFERENCE_SHA256
from coreai_drum_reference30_generation_contract import SAMPLER_BUNDLE,DECODER_BUNDLE

PROFILE='full-song-prepared-generation'
CONFIG='full-song-generation.json'


def validate_inputs(ref):
    plan=schedule(ref['frames32'])
    expected={'schema_version':1,'profile':'full-song-prepared-inputs-v1','status':'prepared_inactive_inputs',
              'instruments':['bass','drums','guitar'],'window_frames':750,'condition_tokens':127,
              'latent_frames':plan['latent_frames'],'target_samples48':plan['target_samples48'],'windows':plan['windows'],
              'preset_sha256':PRESETS,'inference_steps':50,'guidance_scale':7.,'apg_momentum':-.75,
              'apg_norm_threshold':2.5,'apg_projection_dims':[1],'sampler_weight_storage_dtype':'bfloat16',
              'compute_dtype':'float32','vae_dtype':'float32','absolute_tolerance':1e-4,'relative_tolerance':1e-3,
              'relative_rmse_tolerance':1e-3,'baseline_modified':False,'native_exports':0,'profile_activated':False,
              'ready_for_native_execution':False,'source_gain_already_applied':.7,'additional_source_gain':1.}
    if plan['latent_frames']<750 or any(ref.get(k)!=v for k,v in expected.items()):raise ValueError('Full-song input geometry/settings changed')
    if plan['target_samples48']>(2**32-1-50)//8:raise ValueError('Full-song stereo Float32 WAV exceeds RIFF size limit')
    frames=plan['latent_frames']
    shapes={'piano48.f32le':[1,2,frames*1920],'source-context.f32le':[1,frames,128],
            'source-epsilon.f32le':[1,64,frames],'timesteps.f32le':[51],
            **{role+'/'+name:shape for role in PRESETS for name,shape in [('condition.f32le',[1,127,2048]),
               ('condition-mask.f32le',[1,127]),('noise-global.f32le',[1,frames,64])]}}
    extra={'provenance/source-report.json','provenance/context-report.json','provenance/conditioning-report.json'}
    hashes=ref.get('artifact_sha256',{})
    if (ref.get('tensor_shapes')!=shapes or set(hashes)!=set(shapes)|extra
            or any(not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v) for v in hashes.values())):
        raise ValueError('Full-song artifact contract incomplete')
    runs=ref.get('runs',[])
    expected_keys={'instrument','window_index','latent_start','latent_end','condition_sha256','initial_noise_sha256',
                   'source_context_sha256','source_audio_sha256','source_epsilon_sha256'}
    expected_order=[(role,w) for role in PRESETS for w in plan['windows']]
    if len(runs)!=len(expected_order):raise ValueError('Missing full-song runs')
    for row,(role,w) in zip(runs,expected_order):
        if (set(row)!=expected_keys or row['instrument']!=role or row['window_index']!=w['index']
                or row['latent_start']!=w['latent_start'] or row['latent_end']!=w['latent_end']
                or row['condition_sha256']!=hashes[role+'/condition.f32le']
                or any(not isinstance(row[k],str) or not re.fullmatch('[a-f0-9]{64}',row[k]) for k in expected_keys if k.endswith('_sha256'))):
            raise ValueError('Full-song run identities differ')
    return ref


def configuration(job_sha,decoder_sha):
    if any(not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v) for v in (job_sha,decoder_sha)):
        raise ValueError('Invalid full-song configuration hash')
    return {'schema_version':1,'profile':PROFILE,'scope':'prepared-input-full-song-diagnostic',
            'job_bundle':'full-song-inputs-'+job_sha[:16],'job_reference_sha256':job_sha,
            'sampler_bundle':SAMPLER_BUNDLE,'sampler_reference_sha256':REFERENCE_SHA256,
            'decoder_bundle':DECODER_BUNDLE,'decoder_reference_sha256':decoder_sha,
            'native_implementation':'prepared-full-song-v1','source_frontend':'host-prepared context; native source NOT EVALUATED',
            'full_song_strict_parity':'NOT EVALUATED','baseline_modified':False}


def validate_config(ref):
    if ref!=configuration(ref.get('job_reference_sha256'),ref.get('decoder_reference_sha256')):
        raise ValueError('Full-song execution configuration differs')
    return ref
