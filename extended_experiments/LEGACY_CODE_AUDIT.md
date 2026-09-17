# Legacy code audit

Repository revision inspected: `df81462b8d4de673c32795d03f372821a53cec4d`

The inspected `scripts/data/generate_multidim_colebrook_parallel2.py` contains duplicated program bodies. In both copies, `build_coeffs_for_sample` receives `z_star`, assigns its components to the expansion variables, and stores `center = [Q1s, x1s, x2s]`. The resulting `center` is therefore the target state. The current training path concatenates the stored center into `seq_x`.

Consequences:

1. Results produced from those files require a target-leakage audit.
2. A baseline-input ablation using the legacy `center` would not isolate baseline information.
3. The new extended generator does not reuse the legacy `center` or `coeffs` arrays.
4. `audit-legacy` checks existing NPZ files and returns a nonzero exit code when more than 1% of centers match targets.

This finding does not prove that every historical result used the affected generator, because the committed repository does not contain the corresponding NPZ datasets or trained checkpoints. It does mean that the affected headline experiments should be reproduced from leakage-safe data before submission.
