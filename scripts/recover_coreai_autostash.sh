#!/usr/bin/env bash
# Run from an existing checkout after fetching the signing fix. Back up first;
# repair only a sole CoreAI project conflict, retaining the original stash.
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
project_file="prototypes/ios27-coreai/CoreAIProbe.xcodeproj/project.pbxproj"
unmerged="$(git diff --name-only --diff-filter=U)"
if [ "$unmerged" != "$project_file" ]; then
    printf 'Stopped: expected only the CoreAI project conflict. Send this status for review:\n'
    git status --short
    exit 1
fi
if ! git merge-base --is-ancestor HEAD origin/main; then
    printf 'Stopped: main cannot be fast-forwarded from this checkout. No files changed.\n'
    exit 1
fi
git cat-file -e ":3:$project_file"
git cat-file -e "origin/main:scripts/configure_coreai_signing.py"
backup_dir="$(mktemp -d "$(dirname "$repo_root")/stems-coreai-backup.XXXXXX")"
printf 'Backup directory: %s\n' "$backup_dir"
tar -czf "$backup_dir/coreai-project.tar.gz" prototypes/ios27-coreai
git show ":3:$project_file" > "$backup_dir/project-local.pbxproj"
git show ":2:$project_file" > "$backup_dir/project-updated.pbxproj"
git diff --binary > "$backup_dir/working.patch"
git diff --cached --binary > "$backup_dir/staged.patch"
git stash list --format='%H %s' > "$backup_dir/stashes.txt"
git show origin/main:scripts/configure_coreai_signing.py > "$backup_dir/configure_coreai_signing.py"

# Validate/extract the Team into the backup before changing the checkout.
python3 "$backup_dir/configure_coreai_signing.py" \
    --from-project "$backup_dir/project-local.pbxproj" \
    --out "$backup_dir/Signing.local.xcconfig"

git restore --source=HEAD --staged --worktree -- "$project_file"
cp "$backup_dir/Signing.local.xcconfig" prototypes/ios27-coreai/Signing.local.xcconfig
git -c rebase.autoStash=false -c merge.autoStash=false pull --ff-only
git check-ignore prototypes/ios27-coreai/Signing.local.xcconfig
printf '\nRecovered project; signing is local. Original stash and project backup are retained.\n'
git status --short
