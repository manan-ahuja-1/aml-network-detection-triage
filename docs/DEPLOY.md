# Deploying the demo

The Streamlit app is built and verified locally. **Publishing it is your call** — it means
pushing this repository to a public GitHub remote, so read the pre-flight below before
running anything.

---

## What the demo is

`app/streamlit_app.py` reads **only committed artifacts**: `results/*.json`,
`results/*.parquet` and `results/figures/*.png`. It never trains, never scores, never
retrieves, and never calls the Anthropic API — so it costs nothing per view, needs no API
key, and cannot leak one.

That property is load-bearing and checked:

```bash
make demo                      # runs it locally on http://localhost:8501

# prove it is self-contained — rename away the two gitignored directories it must not need
mv data data.off && mv models models.off
make demo                      # must still render every tab
mv data.off data && mv models.off models
```

`app/requirements.txt` is deliberately **not** the project's `requirements.txt`. It pins
only `streamlit`, `pandas`, `numpy` and `pyarrow`, because the heavy imports in the
modules the app touches (`kb_index`, `triage`) sit inside functions rather than at module
scope. Installing the full pipeline — lightgbm, networkx, chromadb, anthropic, shap,
scikit-learn — would add roughly a gigabyte of wheels and can exceed a free-tier
container's resource limit.

---

## Pre-flight, before anything becomes public

Run all four. The first two are the ones that matter.

**1. No secret has ever been committed — not just isn't now.**

```bash
git log --all --full-history -- .env .env.* credentials.json token.json
# expect: no output

git grep -nI -E 'sk-ant-|ANTHROPIC_API_KEY *= *.+|KAGGLE_KEY *= *.+' $(git rev-list --all) -- \
  ':!docs/*' ':!*.md' | head
# expect: no output. A key committed once lives in the history forever, and rotating it
# is the only fix — a later deletion does not remove it.
```

**2. The files that must stay untracked, are.**

```bash
git status --porcelain --ignored | grep '^!!' | grep -E '\.env|^!! data/|^!! models/|LEARNING_NOTES'
# expect: .env, data/, models/, docs/LEARNING_NOTES.md all listed as ignored
```

`docs/LEARNING_NOTES.md` is your interview preparation. It is gitignored and should stay
that way — it is written to you, not to a reviewer.

**3. The repo is small enough to clone comfortably.**

```bash
git count-objects -vH | grep size-pack
du -sh results/          # the demo's data; expect a few MB, not hundreds
```

**4. Tests pass.**

```bash
make test
```

---

## Publishing

```bash
# 1. Create an EMPTY public repo on GitHub (no README, no .gitignore, no licence —
#    anything it adds will conflict with the history you already have).

# 2. Point this repo at it and push.
git remote add origin https://github.com/<you>/aml-network-triage.git
git branch -M main
git push -u origin main
```

Then at **share.streamlit.io** → *New app*:

| Field | Value |
|---|---|
| Repository | `<you>/aml-network-triage` |
| Branch | `main` |
| Main file path | `app/streamlit_app.py` |

Streamlit Cloud finds `app/requirements.txt` automatically because it sits beside the
entry point. **Do not add any secret** in *Advanced settings → Secrets*; the app needs
none, and adding one would be the only way to leak one.

First build takes a few minutes. After that, every push to `main` redeploys.

---

## Once it is live

Put the URL in three places:

1. **`src/make_readme.py`** — replace `DEMO_URL`, then `make readme`. The link then lives
   in the generated README rather than being hand-edited into it, so it survives the next
   regeneration.
2. **The GitHub repo's About panel** — the "Website" field, which is what shows on your
   profile.
3. **Your CV**, as the link on the project line. Most reviewers will click a live app and
   will never clone a repo.

---

## If the deploy fails

- **`ModuleNotFoundError`** — something the app imports gained a module-level heavy
  import. Find it with `python -c "import sys; sys.path.insert(0,'src'); from agent
  import triage; print([m for m in ('lightgbm','chromadb','anthropic') if m in
  sys.modules])"` and either move the import inside its function or add the package to
  `app/requirements.txt`. Prefer the former.
- **A tab renders "not in this checkout"** — that results file is gitignored or was never
  generated. `results/*.json` must be committed; check `git ls-files results/`.
- **Resource limits on the free tier** — `results/demo_bundle.json` is the largest thing
  the app loads. `src/export_demo.py` warns above 8 MB and the evidence cap can be lowered
  there.
- **The app boots but figures are missing** — `results/figures/*.png` are committed, but
  `results/curve_scores.parquet` (~30 MB) is not, by design. The figures themselves are
  what the app shows; it never redraws them.
