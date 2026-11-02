# JOG ANN – PHENIX cases with the neural-network surrogate

The four PHENIX cases of `validation/jog`, run with the JOG ANN instead of OpenCalphad for the grain-boundary
equilibrium (see `docs/JOG_ANN.md`). The inputs differ from `validation/jog/test_PHENIXpins_point_*` in two lines only:

- `input_thermochemistry_settings.txt`: `fission_products.module = NEURALNETWORK`;
- `input_thermochemistry.txt`: entry 70 of the manifest (`MO3TE4`, Te is not in the system) becomes `OTHER`, the
  matter of the phases without a dedicated network.

`output_gold.txt` and `thermochemistry_output_gold.txt` are the reference outputs of the surrogate.

## Run

```bash
./Allmake.sh --oc                        # the coupling code is compiled with OpenCalphad support
python3 -m testing.runner --jog-ann      # runs and checks the four cases against the gold outputs
```

## Compare with OpenCalphad

```bash
cd validation/jog_ann/PHENIXpins
python3 ../../jog/PHENIXpins/plot_JOG.py     # OpenCalphad, system pressure (validation/jog)
python3 plot_JOG_ANN.py                      # JOG ANN: same plots and results_summary.txt, from validation/jog_ann
# optional: OpenCalphad at 1 bar, the pressure of the training data (slow, about 5-8 min per point)
python3 make_oc_1bar_cases.py /tmp/oc1bar --run
python3 plot_JOG_ANN.py --cases /tmp/oc1bar --out /tmp/oc1bar
python3 compare_JOG_ANN.py [--oc1 /tmp/oc1bar]   # plots/compare_*.png and compare_summary.txt
```

`plot_JOG_ANN.py` runs `validation/jog/PHENIXpins/plot_JOG.py` unchanged, only redirecting the case, plot and summary
folders. The plots based on `phase_sublattice_composition.txt` (radial phase fractions, JOG elemental composition)
are meaningful for OpenCalphad only: the surrogate writes approximate sublattices for the liquid and none for the
condensed phases.
