#!/usr/bin/env python3
"""Store an Apple Team ID in the ignored CoreAI signing include, without editing pbxproj."""
import argparse
import os
from pathlib import Path
import re
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--team', help='10-character Apple signing Team ID')
    source.add_argument('--from-project', type=Path, help='Read the Team from a saved pbxproj')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'prototypes/ios27-coreai/Signing.local.xcconfig')
    args = parser.parse_args()
    if args.team is not None:
        team = args.team.strip()
        if not re.fullmatch(r'[A-Z0-9]{10}', team):
            parser.error('Team ID must contain exactly 10 uppercase letters/digits')
    else:
        text = args.from_project.read_text()
        assignments = re.findall(
            r'\b(?:DEVELOPMENT_TEAM|DevelopmentTeam)\s*=\s*(?:"([^"\r\n]*)"|([^;\s]+))\s*;', text)
        teams = {quoted or bare for quoted, bare in assignments
                 if re.fullmatch(r'[A-Z0-9]{10}', quoted or bare)}
        if len(teams) != 1:
            parser.error('Expected exactly one signing Team in the saved project; '
                         'no settings changed. Use --team with the intended Team ID.')
        team = teams.pop()
    content = '// Local Apple signing account; ignored by Git.\nDEVELOPMENT_TEAM = ' + team + '\n'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=args.out.parent, prefix='.Signing.',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, args.out)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print('Signing configuration saved:', args.out)


if __name__ == '__main__':
    main()
