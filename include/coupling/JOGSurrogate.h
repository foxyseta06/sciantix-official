//////////////////////////////////////////////////////////////////////////////////////
//  JOGSurrogate (JOG ANN): neural-network surrogate of the OpenCalphad grain-      //
//  boundary equilibrium (Ba-Cs-Mo-O-Pd-Rh-Ru-Tc, 1 bar, 300-2300 K) for the JOG.    //
//  Phase-fraction networks for the stoichiometric phases (number and formulas read  //
//  from the weights file) + one softmax network that partitions the remaining      //
//  moles of each element among the solution phases and "OTHER" (phases without a    //
//  network). A monotone network inverts mu(O2) -> X_O.                             //
//  The element balance is exact by construction.                                    //
//  Core part: depends only on the C++ standard library (testable stand-alone).     //
//                                                                                  //
//  Author: E. Perugini (MSc thesis, Politecnico di Milano), 2026.                  //
//////////////////////////////////////////////////////////////////////////////////////

#ifndef JOG_SURROGATE_H
#define JOG_SURROGATE_H

#include <string>
#include <vector>

namespace JOGSurrogate
{
    /// Dense feed-forward network: ReLU hidden layers, linear output layer.
    /// Inputs are min-max normalized; output y = ymin + y_n (ymax - ymin), optionally clipped to [0, 1].
    struct DenseNetwork
    {
        std::string                      name;
        std::vector<double>              input_min, input_max;
        double                           output_min = 0.0, output_max = 1.0;
        bool                             clip01 = false;
        std::vector<int>                 n_in, n_out;
        std::vector<bool>                relu;
        std::vector<std::vector<double>> weights;  // row-major, rows = outputs
        std::vector<std::vector<double>> biases;

        std::vector<double> evaluate(const std::vector<double>& x) const;
    };

    enum class OxygenInput
    {
        Potential,  // open system: mu(O) imposed (kJ/mol O2), X_O from the monotone OPOT network
        Content     // closed system: X_O from the oxygen amount
    };

    /// Flags (bit field) returned by Model::evaluate
    enum Flags
    {
        MuOBelowRange = 1,  // mu(O) below the network range: X_O set to its minimum
        MuOAboveRange = 2,  // mu(O) above the network range: X_O set to its maximum
        InputClamped  = 4,  // (X_Cs, X_O, T) outside the training domain: network inputs projected on it
        RestRatioOff  = 8   // ratios among Ba, Mo, Pd, Rh, Ru, Tc differ from the training ratios beyond the tolerance
    };

    struct Result
    {
        double                           x_o                  = 0.0;
        double                           x_cs                 = 0.0;
        int                              flags                = 0;
        double                           rest_ratio_deviation = 0.0;  // max relative deviation from the training ratios
        std::vector<double>              rest_ratio;  // actual ratios among the rest metals (elements() order)
        std::vector<std::vector<double>> moles;  // [group][element], moles per mole of system (sum over groups = input)
    };

    class Model
    {
      public:
        /// Reads the weights text file (format in docs/JOG_ANN.md). Returns false and an error message on failure.
        bool load(const std::string& path, std::string& error);
        bool isLoaded() const
        {
            return loaded_;
        }

        /// fractions: element fractions in the order of elements(); in Potential mode the O entry is ignored.
        Result evaluate(double                     temperature,
                        const std::vector<double>& fractions,
                        OxygenInput                mode,
                        double                     oxygen_potential_kj_per_mol_o2) const;

        const std::vector<std::string>& elements() const
        {
            return elements_;
        }
        const std::vector<std::string>& groups() const
        {
            return groups_;
        }
        bool isStoichiometric(int group) const
        {
            return stech_index_[group] >= 0;
        }
        int atomsPerFormula(int group) const
        {
            return stech_index_[group] >= 0 ? stech_atoms_[stech_index_[group]] : 1;
        }
        double alphaBetaTemperature() const
        {
            return alpha_beta_T_;
        }
        const std::vector<double>& restRatios() const
        {
            return rest_ratio_;
        }  // training ratios (elements() order; 0 for Cs, O)
        double restRatioTolerance() const
        {
            return rest_ratio_tolerance_;
        }
        int oxygenIndex() const
        {
            return oxygen_index_;
        }

      private:
        double opot(double y_cs, double x_o, double temperature) const;
        void   reduceStoichiometric(std::vector<double>& f, const std::vector<double>& b) const;

        bool                             loaded_ = false;
        std::vector<std::string>         elements_, groups_, stech_, solution_;
        std::vector<int>                 stech_atoms_, stech_index_, solution_index_;
        std::vector<std::vector<double>> formula_;  // [element][stech phase]: moles of element per mole of atoms
        double                           alpha_beta_T_ = 841.3;
        std::vector<double>              rest_ratio_;                   // empty for format version 1
        double                           rest_ratio_tolerance_ = 0.05;  // relative
        double xcs_min_ = 0, xcs_max_ = 1, xo_min_ = 0, xo_max_ = 1, sum_max_ = 1, T_min_ = 0, T_max_ = 1e9;
        int    oxygen_index_ = -1, caesium_index_ = -1;
        std::vector<DenseNetwork> stech_networks_;
        DenseNetwork              partition_network_, opot_network_;
    };

    /// Model loaded once per path and shared by all calls.
    const Model& sharedModel(const std::string& path);
}  // namespace JOGSurrogate

#endif
