"""Fixed-input50-step generation contract; no new weights or reference resets."""
from coreai_drum_reference30_contract import BUNDLE as SAMPLER_BUNDLE, REFERENCE_SHA256, CAPTURE_REPORT_SHA256
from coreai_drum_reference30_window_contract import BUNDLE as DECODER_BUNDLE

PROFILE='drums-reference30-generation'
CONFIG='drums-reference30-generation.json'
SOURCE_SHA='9d62d4eccd36188a4170a08ee6a22fcc66b5ad8af3b0753dc6b021a7f507f4a8'
LISTENING_GAIN=0.36943485303214846
SHAPES={'capture/initial-noise.f32le':[1,750,64],'capture/condition.f32le':[1,127,2048],
        'capture/source-context.f32le':[1,750,128],'capture/timesteps.f32le':[51],
        'capture/source48.f32le':[1,2,1440000],
        **{f'capture/step-{s:02d}-latent.f32le':[1,750,64] for s in (1,5,10,25,50)}}
KEEP=set(SHAPES)|{'capture/report.json'}|{a+'/'+n for a in
    ['front.aimodel',*[f'blocks-{i:02d}-{i+1:02d}.aimodel' for i in range(0,24,2)],'tail.aimodel','apg-euler.aimodel']
    for n in ('main.hash','main.mlirb','metadata.json')}


def configuration(decoder_sha):
    return {'schema_version':1,'profile':PROFILE,'scope':'frozen_reference_diagnostic',
        'sampler_bundle':SAMPLER_BUNDLE,'sampler_reference_sha256':REFERENCE_SHA256,
        'decoder_bundle':DECODER_BUNDLE,'decoder_reference_sha256':decoder_sha,
        'capture_report_sha256':CAPTURE_REPORT_SHA256,'source_audio_sha256':SOURCE_SHA,
        'latent_frames':750,'condition_tokens':127,'audio_seconds':30.,'sample_rate':48000,
        'inference_steps':50,'checkpoints':[1,5,10,25,50],'guidance_scale':7.,'shift':1.,
        'apg_momentum':-.75,'apg_norm_threshold':2.5,'apg_projection_dims':[1],
        'sampler_weight_storage_dtype':'bfloat16','compute_dtype':'float32','VAE_dtype':'float32',
        'absolute_tolerance':1e-4,'relative_tolerance':1e-3,'relative_rmse_tolerance':1e-3,
        'reference_listening_gain':LISTENING_GAIN,'piano_mix_gain':.5,'drum_mix_gain':.5,
        'lifecycle':'actual_step1_sequential_then_max15_retained_release_before_window_decoder',
        'feedback':'live_native_latent_and_momentum_no_gold_resets','native_full_trajectory':'NOT EVALUATED',
        'native_music_acceptance':'NOT EVALUATED','baseline_modified':False}


def validate(ref,decoder_sha):
    if ref!=configuration(decoder_sha):raise ValueError('Frozen30 generation configuration mismatch; original settings/gates required')
    return ref
