# DPUC nuPlan code review notes

## Fixed in this revision

- Corrected nuPlan SQLite object-category loading to join `track.category_token -> category.name` instead of assuming `track.category_name` exists.
- Added quaternion-to-yaw recovery for `ego_pose` rows.
- Reworked preprocessing to build multi-slot candidate structures with beam search instead of singleton-slot pseudo-structures.
- Padded witness-dependent tensors in datasets so support-head training does not break when different actions have different witness counts.
- Updated interface slot feature dimensionality to match the actual engineered slot features.
- Added frozen-support importance weighting, ESS tracking, max normalized weight diagnostics, and fallback re-evaluation with larger retained support.
- Improved greedy retained-support selection with diversity tie-breaking and a rescue insertion for uncovered witnesses.
- Added empty-dataset guards in training code.

## Still simplified relative to the paper

- No official nuPlan map API integration yet, so anchors/route-compatibility are still geometric heuristics rather than map-grounded conflict/merge/route anchors.
- No closed-loop nuPlan simulator integration yet.
- Mixed public/individualized online planning now includes a heuristic DBI-based runtime gate and same-support re-evaluation; a fully learned mixed-interface refresh is still future work.
- Residual laws are still lightweight diagonal Gaussian approximations instead of the full grouped residual design described in the paper.
