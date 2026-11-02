# JOG ANN: neural-network alternative to OpenCalphad for the JOG

The JOG ANN replaces the OpenCalphad equilibrium **at the grain boundary** (Ba-Cs-Mo-O-Pd-Rh-Ru-Tc) with a neural
network. Everything downstream (thermochemistry variables, Cs balance, `JOGFormation`) is the standard SCIANTIX code:
the network output is converted into the same `OCOutputData` structure produced by the OpenCalphad coupling.
Author: E. Perugini (MSc thesis, Politecnico di Milano, 2026).

## How to use it (routine simulation)
1. Build SCIANTIX as usual with OpenCalphad (`./Allmake.sh --oc`). The network needs the OpenCalphad data
   structures, so it is available only in builds with the coupling enabled.
2. In `input_thermochemistry_settings.txt` set
   ```
   fission_products.module = NEURALNETWORK
   ```
   The weights are read from `data/JOG_ANN.txt`; nothing else is needed. Optionally,
   `neuralnetwork.path = <file>` selects a different weights file (absolute path, or a name searched first in the test
   folder and then in `data/`).
3. In `input_thermochemistry.txt` replace entry 70 (`MO3TE4`: Te is not in the system) with
   ```
   70 | fission_products | condensed | OTHER | at grain boundary | (g/m3) | 1
   ```
   `OTHER` holds the phases without a dedicated network (Cs oxides, Cs and Ba polymolybdates). The manifest is limited
   to 123 entries, which is why an unused entry is reused instead of adding one.
4. Run SCIANTIX as usual. If the network is evaluated outside its training domain, a single warning is printed and
   the inputs are projected onto the domain. If the ratios among Ba, Mo, Pd, Rh, Ru, Tc at the grain boundary differ
   from those of the training data (Mo:Ba:Tc:Ru:Rh:Pd = 21.9:9.9:5.3:19.8:5.9:18.9) by more than 5%, a single warning
   prints the actual and training ratios.
5. The network is evaluated at every time step: the cache of the previous equilibrium used with OpenCalphad
   (`coupling.*` tolerances) is skipped. `neuralnetwork.use_cache = true` restores it, only to measure its effect.

`matrix.module = NEURALNETWORK` is not available (grain boundary only).

## What the network computes
For given temperature, fuel oxygen potential μ(O₂) and fission-product inventory:
1. A network (5×40) trained to be monotone in X_O (penalty on negative slopes) is inverted by bisection to give
   X_O from μ(O₂) at fixed metal ratios.
2. Nine networks (7×20 ReLU) give the fractions of the stoichiometric phases: Cs₂MoO₄ (α/β by temperature, 841.3 K),
   Ba₃MoO₆, BaMoO₄, Ba₂MoO₅, MoPd₂, BaO (HALITE), BaO₂ (MO2_C11A), MoO₂, BaMoO₃ (PEROVSKITE). They are scaled down,
   if needed, so that no element is used beyond its availability.
3. A softmax network (5×64) splits the remaining moles of each element among HCP_A3, LIQUID, GAS, FCC_A1, BCC_A2,
   HCP_ORD and OTHER. The element balance is exact by construction.
4. The liquid gets approximate sublattice fractions built from its element amounts, because `JOGFormation` uses them
   to tell oxide from metallic liquid and to estimate the oxide-liquid density. If its O atom fraction is ≥ 0.3 the
   liquid is oxide: Mo goes to MoO₄²⁻ anions as far as O allows (4 O per Mo), the remaining Mo is a Mo⁴⁺ cation and
   the remaining O is O²⁻. Otherwise the liquid is metallic (second sublattice VA).

Training domain: X_Cs 0.01–0.8, X_O 1/15–0.8, X_Cs + X_O ≤ 13/15, T 300–2300 K, P = 1 bar.

## Provenance of `data/JOG_ANN.txt`
- md5 `6ab6ba62b332a96c0de01fdf3333d07d`.
- Training dataset: OpenCalphad minimum of G along the sweep paths, checked against pycalphad candidates (lowest G
  wins), same TDB `BaMoO_CsMoO_MoPdRhRuTc_merged.TDB`, 1 bar. 2317 training rows plus held-out sets.
- Partition network trained with a Kullback–Leibler loss. Seed 4 of five, chosen on the validation set only.
- C++ and Python reference implementations agree to 1e-12 on 672 test vectors (engine in `src/coupling/JOGSurrogate.C`).

## Weights file format
Plain text, read by `JOGSurrogate::Model::load`; lines starting with `#` are comments. After the header `JOG_ANN 2`
come the element list, the phase groups, the stoichiometric phases with their atoms per formula and exact formulas,
the solution phases, the α/β temperature of Cs₂MoO₄, the training domain, the fixed ratios among Ba, Mo, Pd, Rh,
Ru, Tc of the training data (`rest_ratios`, in the order of `elements`, 0 for Cs and O) and their relative
tolerance (`rest_ratio_tolerance`). Version 1 files have no ratios and are still accepted. Then one `net` block per network
(9 stoichiometric phases, `PARTITION`, `OPOT`), each with input/output min-max normalization, `clip01` and its
layers (`layer n_in n_out relu|linear`, then `W` row-major with rows = outputs, then `b`). The file ends with `end`.
Retraining with the same architecture only requires replacing this file.

## Known limits
- **Pressure.** The training data are at 1 bar: only the gas phase depends on pressure in this TDB. The proper
  OpenCalphad reference for the network is therefore a run at 1 bar.
- **Sharp phase boundaries.** Errors are larger close to sharp phase boundaries. For example, at the end of cooling of
  the PHENIX validation cases, Cs₂MoO₄ near X_O ≈ 0.44–0.46 (onset of polymolybdates) and Ba₃MoO₆ near X_Cs ≈ 0.185
  are about 25% below OpenCalphad.
- **No thermodynamic quantities.** G, H and μ are not provided. The liquid site fractions are approximate (see above),
  and the condensed phases have no sublattice data, so `phase_sublattice_composition.txt` lists the liquid only.
- **OTHER** has no density in the manifest: SCIANTIX uses its 4 g/cm³ fallback and reports it as "JOG (other phases)".
- **Weights search in `data/`** uses `/proc/self/exe` (Linux). On other systems give `neuralnetwork.path` as an
  absolute path or put the weights file in the test folder.
