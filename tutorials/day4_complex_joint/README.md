# Single-trial integrative joint modeling

- **Instructor:** Michael D. Nunez
- **When:** Day 4, Thursday July 30, 14:00 — "Joint modeling workshop: using a
  deep neural network to build in covariation"
- **Stack:** Google Colab (nothing to install locally) · BayesFlow 2

## Run

Follow the workshop steps in the companion repository:

```bash
open https://github.com/mdnunez/single_trial_nddm_compare/blob/workshop_demo/workshop_steps.md
```

## Requirements

A browser and a Google account — that is all. The notebook is written for
Google Colab and clones
[`mdnunez/single_trial_nddm_compare`](https://github.com/mdnunez/single_trial_nddm_compare/tree/workshop_demo)
(branch `workshop_demo`) at runtime. Model fitting uses BayesFlow 2 with
networks that were **pre-trained** on the Snellius cluster, so the session does
not train from scratch. The notebook itself notes it depends on Colab's current
runtime and may break as Google updates it.

## Files

- `demo_single_trial_integrative_joint_modeling.ipynb` — the demonstration
  notebook, committed with stored outputs so it renders as a site page. Source
  of truth for the live session is the `workshop_demo` branch linked above.
