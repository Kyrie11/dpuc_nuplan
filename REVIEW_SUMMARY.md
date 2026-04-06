# DPUC nuPlan implementation review summary

## What I checked

- Alignment between the paper's Method + Appendix and the codebase modules.
- nuPlan SQLite loading against the official schema.
- Runtime bugs that would break preprocessing / training / offline evaluation.
- Gaps between the paper design and the current implementation.

## Main issues found and fixed

1. **Agent dimension fallback bug in preprocessing**
   - `sqlite3.Row` does not implement `.get(...)`.
   - This could break `length/width` fallback when a field is absent.
   - Fixed by adding a safe `_get_row_value(...)` helper.

2. **Incomplete agent history extraction**
   - The previous preprocessing only stored the current frame for each agent.
   - This did not match the intended 2s history-window design.
   - Fixed by querying `lidar_box` across the history window using selected `track_token`s.

3. **DBI training crash**
   - `train_dbi()` referenced `batch['weights']`, which does not exist in `DBIDataset`.
   - Fixed by replacing that loss with plain MSE and adding grad clipping.

4. **Selective individualization missing in planner runtime**
   - Added a heuristic DBI-based runtime gate that ranks agents by boundary influence.
   - The planner now performs a mixed public/individualized pass before fallback.
   - This is still a heuristic approximation, not yet the full learned mixed-interface refresh described in the paper.

5. **Conservative evaluation / diagnostics refinement**
   - Added per-action uncertainty margins from ESS and normalized-weight diagnostics.
   - Added a gap-vs-uncertainty trigger for fallback.
   - Offline evaluation now also reports fallback rate and average refined-agent count.

## Still simplified relative to the paper

- No official nuPlan map API grounding yet; anchors and route compatibility are still geometric heuristics.
- No official closed-loop nuPlan simulator integration yet.
- Mixed-interface runtime refresh is now present heuristically, but not yet via a learned interface refresh network.
- Residual laws remain lightweight diagonal Gaussian approximations rather than the full grouped residual design.

## nuPlan schema notes used for validation

- `lidar_box.track_token -> track.token`
- `track.category_token -> category.token`
- `category.name` stores the object class name
- `lidar_pc.ego_pose_token -> ego_pose.token`
- `ego_pose` stores quaternion fields `qw/qx/qy/qz`
- `scenario_tag.lidar_pc_token` links scenario types to a frame

These match the current loader design after the fixes above.
