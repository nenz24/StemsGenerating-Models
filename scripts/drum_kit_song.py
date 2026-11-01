"""Continue an approved kit comparison across the song, reusing approved chunk 0."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import ace_context_chunks as chunks
import drum_probe as drum


def prepare(root,source,output,*,comparison_folder=None,result_archive=None,
            expected_candidate_sha=None,expected_reference_sha=None):
    root,source,output=Path(root).resolve(),Path(source).resolve(),Path(output).resolve()
    if (comparison_folder is None)==(result_archive is None):raise ValueError('Choose approved comparison folder or result ZIP')
    if output.exists():raise ValueError('Output exists; use a new folder to preserve the approved experiment')
    # Inspect metadata before preparing any new plan; do not accept feedback MP3s.
    if comparison_folder is not None:
        old=Path(comparison_folder).resolve()
        comparison=json.loads((old/'comparison.json').read_text())
        report=json.loads((old/'candidate/drum_probe.json').read_text())
        ref=old/'reference.wav';stem=old/'candidate/drums.wav'
        if not ref.is_file() or not stem.is_file():raise ValueError('Approved WAVs missing; use drum_kit_result.zip')
    else:
        with zipfile.ZipFile(result_archive) as z:
            required={'comparison.json','reference.wav','candidate/drum_probe.json','candidate/drums.wav'}
            if not required.issubset(z.namelist()):raise ValueError('Use drum_kit_result.zip, not feedback ZIP')
            if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>256*1024**2:
                raise ValueError('Invalid or oversized result ZIP')
            comparison=json.loads(z.read('comparison.json'));report=json.loads(z.read('candidate/drum_probe.json'))
    if comparison.get('status')!='generated' or comparison.get('target_chunk')!=0 or report.get('status')!='generated':
        raise ValueError('Expected successful chunk-0 reference comparison')
    if report['preset']!=drum.PRESET or comparison['preset']!=drum.PRESET:
        raise ValueError('Approved drum preset differs')
    if comparison['original_source_sha256']!=chunks.sha256(source) or report['input']['original_sha256']!=chunks.sha256(source):
        raise ValueError('Original piano differs from approved comparison')
    if expected_candidate_sha is not None and report['stem_sha256']!=expected_candidate_sha:raise ValueError('Not the approved chunk-0 candidate')
    if expected_reference_sha is not None and comparison['reference_excerpt_sha256']!=expected_reference_sha:raise ValueError('Not the approved reference')
    # A temporary extraction directory outside the original run is retained for
    # audit on failure. Named members only; never extract arbitrary ZIP paths.
    if result_archive is not None:
        import tempfile
        output.parent.mkdir(parents=True,exist_ok=True)
        recovered=Path(tempfile.mkdtemp(prefix='approved_drum_kit_',dir=output.parent))
        with zipfile.ZipFile(result_archive) as z:
            ref=recovered/'reference.wav';stem=recovered/'drums.wav'
            ref.write_bytes(z.read('reference.wav'));stem.write_bytes(z.read('candidate/drums.wav'))
    if chunks.sha256(ref)!=comparison['reference_excerpt_sha256'] or chunks.sha256(stem)!=report['stem_sha256']:
        raise ValueError('Approved reference/candidate WAV changed')
    if comparison['candidate_stem_sha256']!=report['stem_sha256'] or report['reference_audio']['sha256']!=chunks.sha256(ref):
        raise ValueError('Comparison/report reference or candidate hash mismatch')
    plan=chunks.prepare(source,output,context=90,margin=6,fade=.58,instrument='drums',reference_audio=ref)
    plan['status']='preparing_approved_kit';chunks.write_json(output/'plan.json',plan)
    chunk=plan['chunks'][0]
    first=output/'chunk_000/attempt_000';first.mkdir(parents=True)
    shutil.copyfile(stem,first/'drums.wav');chunks.write_json(first/'drum_probe.json',report)
    chunks.verify_chunk(first,plan,chunk)
    drum.single.attach(root)
    check=drum.single.prepare(source,output/'approved_input_check',start=chunk['input_start']/chunks.RATE,
                             duration=(chunk['input_end']-chunk['input_start'])/chunks.RATE)
    if check['source_sha256']!=report['input']['source_sha256']:
        raise ValueError('New plan piano input differs from approved chunk 0; do not regenerate it silently')
    chunks.write_json(output/'progress.json',{'0':'chunk_000/attempt_000'})
    chunks.write_json(output/'approved_kit.json',{
        'status':'prepared','approved_chunk_0_reused':True,
        'approved_chunk_0_sha256':report['stem_sha256'],
        'shared_reference_sha256':chunks.sha256(ref),
        'source_sha256':plan['original_sha256'],
        'new_model_calls_requested':len(plan['chunks'])-1,
        'note':'Same reference bytes across chunks; consistency still needs listening. No EQ or independent kick replacement.'})
    plan['status']='planned';chunks.write_json(output/'plan.json',plan)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    old=p.add_mutually_exclusive_group(required=True)
    old.add_argument('--comparison',type=Path);old.add_argument('--result-zip',type=Path)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    prepare(a.root,a.source,a.out,comparison_folder=a.comparison,result_archive=a.result_zip)
    if a.execute:
        chunks.run_chunks(a.root,a.out/'plan.json');chunks.stitch_run(a.out/'plan.json')
