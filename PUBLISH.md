# Create the public repository manually

Preparation did not initialize Git, create a remote repository, commit, or push. This directory is an independent copy of selected files from the `thesis_hybrid_model` working tree, including necessary uncommitted edits and restored helpers. Initializing here creates a single new root commit: no original Git history, branches, worktree link or remote is copied.

The commands below use the new repository name `AugustoBL/Thesis_hybrid_model`; change that argument if the name is already taken. Run each step only after the previous step succeeds. Do not run `git init` or `gh repo create` inside the original checkout.

1. Install GitHub CLI if `gh --version` is not available. On Windows:

```powershell
winget install --id GitHub.cli --exact --source winget
```

Open a fresh PowerShell window after installation, then verify prerequisites and sign in:

```powershell
git --version
git lfs version
gh --version
gh auth login
```

2. Enter the prepared standalone directory, review the migration documents, and verify files:

```powershell
cd 'C:\Users\ajlim\Documents\Git\Thesis_hybrid_model_public'
python scripts/verify_repository.py
```

3. Initialize a fresh repository and stage only this clean copy:

```powershell
git init -b main
git lfs install --local
git add .
git status --short
git diff --cached --stat
git lfs ls-files
```

The included `.gitattributes` sends `.jld2` and `.bson` files to Git LFS, including all five required datasets. Four ensembles exceed GitHub's 100 MiB ordinary-Git file limit. Review the staged files and LFS list before continuing. `.gitignore` excludes fresh runs and locally regenerated figures. No thesis directory should appear in the staged files.

4. Create the initial commit, check LFS consistency, and publish publicly:

```powershell
git commit -m 'Initial standalone thesis experiment release'
git lfs fsck
git log --oneline --all
gh repo create AugustoBL/Thesis_hybrid_model --public --source . --remote origin --push
```

5. Verify the published repository:

```powershell
gh repo view AugustoBL/Thesis_hybrid_model --json nameWithOwner,isPrivate,url
git remote -v
gh repo view AugustoBL/Thesis_hybrid_model --web
```

`isPrivate` should be `false`. Collaborators should install Git LFS and run `git lfs pull` after cloning. The included data and model objects use GitHub LFS storage and download allowance.

References: [GitHub CLI repository creation](https://cli.github.com/manual/gh_repo_create), [GitHub large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [GitHub CLI installation](https://github.com/cli/cli#installation).
