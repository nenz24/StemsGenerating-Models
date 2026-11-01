"""Two matched 30-second guitar trials: original strumming versus picked notes."""
import argparse
import json
from pathlib import Path
import zipfile

import guitar_probe as guitar


def compare(root, source, output, *, start=77.130625, execute=False):
    root, source, output = [Path(p).resolve() for p in (root, source, output)]
    guitar.single.attach(root)
    from provenance import sha256, write_json
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'source_sha256': sha256(source),
              'start_seconds': start, 'duration_seconds': 30,
              'model_calls_requested': 2,
              'note': 'Same 30-second piano input and generation settings; only caption changes. Compare shorter original against the rejected 90-second run before attributing noise to context length. No full-song regeneration.'}
    write_json(output / 'comparison.json', report)
    try:
        trials = {}
        for name in ('original', 'picked'):
            trials[name] = guitar.generate(root, source, output / name, start=start,
                                           duration=30, execute=execute, variant=name)
            write_json(output / 'comparison.json', dict(report, completed_trials=list(trials)))
        a, b = trials['original'], trials['picked']
        if a['input'] != b['input'] or a['input']['duration'] != 30:
            raise ValueError('Comparison requires the same complete 30-second input')
        if a['engine_sha256'] != b['engine_sha256'] or sha256(source) != report['source_sha256']:
            raise ValueError('Comparison engine or original source changed')
        preset_differences = sorted(k for k in a['preset'].keys() | b['preset'].keys() if a['preset'].get(k) != b['preset'].get(k))
        if preset_differences != ['caption', 'id']:
            raise ValueError('Only the guitar caption/preset id may differ')
        report.update(input_sha256=a['input']['source_sha256'], preset_differences=preset_differences)
        if not execute:
            report.update(status='planned')
            return report
        if a['status'] != 'generated' or b['status'] != 'generated':
            raise ValueError('Both guitar trials must finish')
        if a['settings']['config'] != b['settings']['config']:
            raise ValueError('Guitar generation configs differ')
        pa, pb = a['settings']['params'], b['settings']['params']
        differences = sorted(k for k in pa.keys() | pb.keys() if pa.get(k) != pb.get(k))
        if 'caption' not in differences or set(differences) - {'caption', 'src_audio'}:
            raise ValueError('Unexpected differences between actual guitar settings')
        report.update(status='generated', actual_parameter_differences=differences,
                      original_stem_sha256=a['stem_sha256'], picked_stem_sha256=b['stem_sha256'],
                      musical_quality='unreviewed')
        write_json(output / 'comparison.json', report)
        for filename, full in [('guitar_articulation_feedback.zip', False), ('guitar_articulation_result.zip', True)]:
            with zipfile.ZipFile(output / filename, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(output / 'comparison.json', 'comparison.json')
                for name in ('original', 'picked'):
                    relatives = ['guitar_probe.json', 'prepared/input.json', 'guitar_preview.mp3', 'mix_preview.mp3']
                    if full:
                        relatives += ['guitar.wav', 'mix.wav']
                    for relative in relatives:
                        archive.write(output / name / relative, name + '/' + relative)
        return report
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'comparison.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=float, default=77.130625)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(compare(args.root, args.source, args.out, start=args.start, execute=args.execute), indent=2))
