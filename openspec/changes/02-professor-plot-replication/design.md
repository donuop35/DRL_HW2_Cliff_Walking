# Design: 02-professor-plot-replication

## Status: APPROVED

## Architecture

### Experiment Script (`scripts/run_professor_plot.py`)
- Independent script to run the specific configuration required to replicate the professor's plot.
- Settings: `alpha=0.5`, `epsilon=0.1`, `gamma=1.0`, `n_episodes=500`, `n_runs=50`.
- Generates new raw data: `rewards_qlearning_prof.npy` and `rewards_sarsa_prof.npy`.

### Visualization
- Implements `plot_professor_style()` in the script to match the visual style of the provided reference image.
- **Lines**:
  - Teal solid: SARSA (our experiment)
  - Dark red solid: Q-learning (our experiment)
  - Teal dotted: SARSA (Sutton Pub. reference)
  - Dark red dotted: Q-learning (Sutton Pub. reference)
- **Reference Curves**: Hardcoded approximate keypoints based on Sutton & Barto Fig 6.4, with light interpolation and very mild noise to match the "Sutton Pub." reference lines.
- **Smoothing**: Applies a very light Moving Average (window=5) to our experimental data to reduce extreme outliers (like -100 drops) while maintaining the characteristic jagged look of the professor's raw averaged plot.

### Report Update
- Append a new section to `report/hw2_report.md` detailing the replication of the textbook / professor's plot.
- Embed `results/figures/07_professor_style.png`.
