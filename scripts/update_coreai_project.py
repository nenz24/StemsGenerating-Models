#!/usr/bin/env python3
"""Back up local Xcode project edits, migrate signing, fast-forward fetched main.

Run from the checkout after `git fetch origin main`. Only the CoreAI pbxproj
is restored. Models, caches, other working files and stashes are untouched.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

PROJECT = 'prototypes/ios27-coreai/CoreAIProbe.xcodeproj/project.pbxproj'
SIGNING = 'prototypes/ios27-coreai/Signing.local.xcconfig'
TEAM_PATTERN = r'\b(?:DEVELOPMENT_TEAM|DevelopmentTeam)\s*=\s*(?:"([^"\r\n]*)"|([^;\s]+))\s*;'


def git(root, *args, check=True):
    return subprocess.run(['git', '-C', str(root), *args], check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def update(root):
    root = Path(git(root, 'rev-parse', '--show-toplevel').stdout.decode().strip()).resolve()
    if git(root, 'branch', '--show-current').stdout.strip() != b'main':
        raise ValueError('Stopped: switch to main first; no files changed.')
    if git(root, 'diff', '--name-only', '--diff-filter=U').stdout.strip():
        raise ValueError('Stopped: unresolved conflicts need separate recovery; no files changed.')
    git_dir = Path(git(root, 'rev-parse', '--absolute-git-dir').stdout.decode().strip())
    if (git(root, 'rev-parse', '--verify', '-q', 'MERGE_HEAD', check=False).returncode == 0
            or any((git_dir / name).is_dir() for name in ('rebase-merge', 'rebase-apply'))):
        raise ValueError('Stopped: complete the active merge/rebase first; no files changed.')
    if git(root, 'merge-base', '--is-ancestor', 'HEAD', 'origin/main', check=False).returncode:
        raise ValueError('Stopped: main cannot fast-forward to origin/main; no files changed.')
    shared = git(root, 'show', 'origin/main:prototypes/ios27-coreai/Signing.xcconfig').stdout
    if b'Signing.local.xcconfig' not in shared:
        raise ValueError('Stopped: upstream lacks the local signing include; no files changed.')
    if git(root, 'check-ignore', '--quiet', SIGNING, check=False).returncode:
        raise ValueError('Stopped: local signing is not ignored; no files changed.')
    incoming = set(git(root, 'diff', '--name-only', '-z', 'HEAD', 'origin/main').stdout.split(b'\0'))
    local = set(git(root, 'diff', '--name-only', '-z', 'HEAD').stdout.split(b'\0'))
    overlap = (incoming & local) - {b'', PROJECT.encode()}
    if overlap:
        raise ValueError('Stopped: other edited files would be updated; no files changed: '
                         + ', '.join(p.decode() for p in sorted(overlap)))
    project, signing = root / PROJECT, root / SIGNING
    if not project.is_file() or project.is_symlink() or signing.is_symlink():
        raise ValueError('Stopped: missing project or unexpected symlink; no files changed.')
    original = project.read_bytes()
    teams = {quoted or bare for quoted, bare in re.findall(TEAM_PATTERN, original.decode())
             if re.fullmatch(r'[A-Z0-9]{10}', quoted or bare)}
    if len(teams) > 1:
        raise ValueError('Stopped: multiple project signing Teams; no files changed.')
    # Check this is a tracked project before taking the local edits out of Git.
    staged = git(root, 'show', ':' + PROJECT).stdout
    existing_signing = signing.read_bytes() if signing.exists() else None
    backup = Path(tempfile.mkdtemp(prefix='stems-coreai-project-backup-', dir=root.parent))
    (backup / 'project-local.pbxproj').write_bytes(original)
    (backup / 'project-staged.pbxproj').write_bytes(staged)
    for name, args in [('working.patch', ('diff', '--binary', '--', PROJECT)),
                       ('staged.patch', ('diff', '--cached', '--binary', '--', PROJECT))]:
        (backup / name).write_bytes(git(root, *args).stdout)
    if existing_signing is not None:
        shutil.copy2(signing, backup / 'Signing.local.xcconfig')
    print('Backup:', backup, flush=True)
    if teams:
        # The project override wins over the include in Xcode. Preserve it in
        # the local include, while keeping all other custom signing settings.
        team = next(iter(teams))
        content = existing_signing.decode() if existing_signing is not None else '// Local Apple signing account; ignored by Git.\n'
        override = 'DEVELOPMENT_TEAM = ' + team
        # Preserve every custom setting and append the effective project override.
        if content.splitlines()[-1:] != [override]:
            content += ('\n' if not content.endswith('\n') else '') + override + '\n'
            descriptor, temporary = tempfile.mkstemp(prefix='.Signing.', dir=signing.parent)
            try:
                with os.fdopen(descriptor, 'w') as stream:
                    stream.write(content)
                os.replace(temporary, signing)
            finally:
                Path(temporary).unlink(missing_ok=True)
        print('Project Team preserved in ignored Signing.local.xcconfig.', flush=True)
    else:
        print('No explicit project Team; existing local signing retained.', flush=True)
    git(root, 'restore', '--source=HEAD', '--staged', '--worktree', '--', PROJECT)
    result = git(root, '-c', 'merge.autoStash=false', '-c', 'rebase.autoStash=false',
                 'merge', '--ff-only', 'origin/main', check=False)
    if result.stdout:
        print(result.stdout.decode(), end='', flush=True)
    if result.returncode:
        raise RuntimeError('Fast-forward stopped. Project/signing backup retained at ' + str(backup)
                           + '\n' + result.stderr.decode())
    print('Updated. Other project edits remain in the backup; models/cache/stashes untouched.', flush=True)
    return backup


def main():
    try:
        update(Path.cwd())
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        detail = error.stderr.decode() if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise SystemExit(detail)


if __name__ == '__main__':
    main()
