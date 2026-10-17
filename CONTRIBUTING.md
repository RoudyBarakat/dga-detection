# How we work

`main` always holds the reviewed version of the project. Nobody pushes to `main`
directly: every change goes through a branch and a pull request (PR).

## Workflow

1. Start from the latest `main`:
   ```
   git switch main
   git pull
   ```
2. Create a branch for one piece of work, named `<your-name>/<topic>`:
   ```
   git switch -c abdo/related-work
   ```
3. Commit your work on that branch and push it:
   ```
   git add <files>
   git commit -m "Write related work section"
   git push -u origin abdo/related-work
   ```
4. Open a pull request into `main` (on GitHub, or `gh pr create`).
5. The other teammate reviews it. Fix any comments by pushing more commits to the
   same branch.
6. After approval, merge the PR on GitHub and delete the branch.

## Rules

- No direct pushes to `main`; no force-pushes to `main`.
- Every PR needs one approval from the other teammate before merging.
- Keep a PR to one topic (one proposal section, one script, one experiment).
- Do not commit raw data or large files; see `data/raw/README.md`.
- Before opening a PR that changes the proposal, check that it compiles.

## Repository settings (for the owner)

These rules are enforced by GitHub through a ruleset on `main`, defined in
`.github/ruleset-main.json`. Rulesets on private repositories need GitHub Pro
(free with the GitHub Student Developer Pack). To apply or re-apply it:

```
gh api -X POST repos/RoudyBarakat/dga-detection/rulesets --input .github/ruleset-main.json
```

To invite a teammate with write access:

```
gh api -X PUT repos/RoudyBarakat/dga-detection/collaborators/<github-username> -f permission=push
```
