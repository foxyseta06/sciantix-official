//////////////////////////////////////////////////////////////////////////////////////
//  JOGSurrogate: neural-network surrogate of the grain-boundary equilibrium (core). //
//  Same algorithm as the Python reference implementation used for training.        //
//  Author: E. Perugini (MSc thesis, Politecnico di Milano), 2026.                  //
//////////////////////////////////////////////////////////////////////////////////////

#include "JOGSurrogate.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace JOGSurrogate
{
    namespace
    {
        std::string upper(std::string s)
        {
            std::transform(
                s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            return s;
        }

        // Token reader that skips '#' comment lines
        class Reader
        {
          public:
            explicit Reader(std::istream& in)
                : in_(in)
            {
            }
            std::string word()
            {
                std::string w;
                while (in_ >> w)
                {
                    if (!w.empty() && w[0] == '#')
                    {
                        std::string rest;
                        std::getline(in_, rest);
                        continue;
                    }
                    return w;
                }
                throw std::runtime_error("unexpected end of file");
            }
            void expect(const std::string& key)
            {
                const std::string w = word();
                if (w != key)
                    throw std::runtime_error("expected '" + key + "', found '" + w + "'");
            }
            int integer()
            {
                return std::stoi(word());
            }
            double number()
            {
                return std::stod(word());
            }
            std::vector<double> numbers(size_t n)
            {
                std::vector<double> v(n);
                for (auto& x : v)
                    x = number();
                return v;
            }
            std::vector<std::string> names(size_t n)
            {
                std::vector<std::string> v(n);
                for (auto& x : v)
                    x = word();
                return v;
            }

          private:
            std::istream& in_;
        };

        DenseNetwork readNetwork(Reader& r, const std::string& expected_name)
        {
            DenseNetwork net;
            r.expect("net");
            net.name = r.word();
            if (net.name != expected_name)
                throw std::runtime_error("expected network " + expected_name + ", found " + net.name);
            r.expect("input_min");
            net.input_min = r.numbers(3);
            r.expect("input_max");
            net.input_max = r.numbers(3);
            r.expect("output_scale");
            net.output_min = r.number();
            net.output_max = r.number();
            r.expect("clip01");
            net.clip01 = r.integer() != 0;
            r.expect("layers");
            const int n_layers = r.integer();
            for (int l = 0; l < n_layers; ++l)
            {
                r.expect("layer");
                const int         ni = r.integer(), no = r.integer();
                const std::string act = r.word();
                if (act != "relu" && act != "linear")
                    throw std::runtime_error("unknown activation " + act);
                r.expect("W");
                net.weights.push_back(r.numbers(static_cast<size_t>(ni) * no));
                r.expect("b");
                net.biases.push_back(r.numbers(no));
                net.n_in.push_back(ni);
                net.n_out.push_back(no);
                net.relu.push_back(act == "relu");
            }
            return net;
        }
    }  // namespace

    std::vector<double> DenseNetwork::evaluate(const std::vector<double>& x) const
    {
        std::vector<double> h(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            h[i] = (x[i] - input_min[i]) / (input_max[i] - input_min[i]);
        for (size_t l = 0; l < weights.size(); ++l)
        {
            // raw pointers and per-layer constants: the same arithmetic, fast also in unoptimized (-O0) builds
            const int           ni = n_in[l], no = n_out[l];
            const bool          rl = relu[l];
            const double*       W  = weights[l].data();
            const double*       B  = biases[l].data();
            const double*       hp = h.data();
            std::vector<double> o(no);
            double*             op = o.data();
            for (int j = 0; j < no; ++j)
            {
                double        s = 0.0;  // same summation order as numpy (W @ h): products then bias
                const double* w = W + static_cast<size_t>(j) * ni;
                for (int i = 0; i < ni; ++i)
                    s += w[i] * hp[i];
                s += B[j];
                op[j] = (rl && s < 0.0) ? 0.0 : s;
            }
            h.swap(o);
        }
        for (auto& y : h)
        {
            y = output_min + y * (output_max - output_min);
            if (clip01)
                y = std::min(1.0, std::max(0.0, y));
        }
        return h;
    }

    bool Model::load(const std::string& path, std::string& error)
    {
        loaded_ = false;
        std::ifstream in(path);
        if (!in)
        {
            error = "cannot open " + path;
            return false;
        }
        try
        {
            Reader r(in);
            r.expect("JOG_ANN");
            const int version = r.integer();
            if (version != 1 && version != 2)
                throw std::runtime_error("unsupported file version");
            r.expect("elements");
            elements_ = r.names(r.integer());
            r.expect("groups");
            groups_ = r.names(r.integer());
            r.expect("stech");
            stech_ = r.names(r.integer());
            r.expect("stech_atoms");
            stech_atoms_.clear();
            for (size_t j = 0; j < stech_.size(); ++j)
                stech_atoms_.push_back(r.integer());
            r.expect("stech_formula");
            const std::vector<double> f = r.numbers(stech_.size() * elements_.size());
            formula_.assign(elements_.size(), std::vector<double>(stech_.size()));
            for (size_t j = 0; j < stech_.size(); ++j)
                for (size_t e = 0; e < elements_.size(); ++e)
                    formula_[e][j] = f[j * elements_.size() + e];
            r.expect("solution");
            solution_ = r.names(r.integer());
            r.expect("alpha_beta_T");
            alpha_beta_T_ = r.number();
            r.expect("domain");
            xcs_min_ = r.number();
            xcs_max_ = r.number();
            xo_min_  = r.number();
            xo_max_  = r.number();
            sum_max_ = r.number();
            T_min_   = r.number();
            T_max_   = r.number();
            rest_ratio_.clear();
            if (version >= 2)
            {
                // fixed ratios among the metals other than Cs assumed by the training data
                r.expect("rest_ratios");
                rest_ratio_ = r.numbers(elements_.size());
                r.expect("rest_ratio_tolerance");
                rest_ratio_tolerance_ = r.number();
            }
            stech_networks_.clear();
            for (const auto& s : stech_)
                stech_networks_.push_back(readNetwork(r, s == "CS2MOO4" ? "CS2MOO4_S1+CS2MOO4_S2" : s));
            partition_network_ = readNetwork(r, "PARTITION");
            opot_network_      = readNetwork(r, "OPOT");
            r.expect("end");
        }
        catch (const std::exception& e)
        {
            error = path + ": " + e.what();
            return false;
        }
        // indices
        oxygen_index_ = caesium_index_ = -1;
        for (size_t e = 0; e < elements_.size(); ++e)
        {
            if (upper(elements_[e]) == "O")
                oxygen_index_ = static_cast<int>(e);
            if (upper(elements_[e]) == "CS")
                caesium_index_ = static_cast<int>(e);
        }
        stech_index_.assign(groups_.size(), -1);
        solution_index_.assign(groups_.size(), -1);
        for (size_t g = 0; g < groups_.size(); ++g)
        {
            for (size_t j = 0; j < stech_.size(); ++j)
                if (groups_[g] == stech_[j])
                    stech_index_[g] = static_cast<int>(j);
            for (size_t k = 0; k < solution_.size(); ++k)
                if (groups_[g] == solution_[k])
                    solution_index_[g] = static_cast<int>(k);
        }
        const size_t K = solution_.size() + 1;
        if (oxygen_index_ < 0 || caesium_index_ < 0 ||
            partition_network_.n_out.back() != static_cast<int>(elements_.size() * K))
        {
            error = path + ": inconsistent element/partition definition";
            return false;
        }
        loaded_ = true;
        return true;
    }

    double Model::opot(double y_cs, double x_o, double temperature) const
    {
        const std::vector<double> x = {
            std::min(std::max(y_cs, opot_network_.input_min[0]), opot_network_.input_max[0]),
            x_o,
            std::min(std::max(temperature, opot_network_.input_min[2]), opot_network_.input_max[2])};
        return opot_network_.evaluate(x)[0];
    }

    void Model::reduceStoichiometric(std::vector<double>& f, const std::vector<double>& b) const
    {
        // Same as Python "riduci": no element may be used beyond its amount b_e.
        const size_t ne = elements_.size(), ns = stech_.size();
        for (int it = 0; it < 50; ++it)
        {
            std::vector<double> u(ne, 0.0);
            for (size_t e = 0; e < ne; ++e)
                for (size_t j = 0; j < ns; ++j)
                    u[e] += formula_[e][j] * f[j];
            bool any = false;
            for (size_t e = 0; e < ne; ++e)
                any = any || (u[e] > b[e] * (1.0 + 1e-9));
            if (!any)
                break;
            for (size_t e = 0; e < ne; ++e)
            {
                if (!(u[e] > b[e] * (1.0 + 1e-9)))
                    continue;
                for (size_t j = 0; j < ns; ++j)
                    if (formula_[e][j] > 0.0)
                        f[j] *= b[e] / u[e];
            }
        }
    }

    Result Model::evaluate(double                     temperature,
                           const std::vector<double>& fractions,
                           OxygenInput                mode,
                           double                     oxygen_potential_kj_per_mol_o2) const
    {
        if (!loaded_)
            throw std::runtime_error("JOGSurrogate model not loaded");
        const size_t ne = elements_.size(), ns = stech_.size(), K = solution_.size() + 1;
        Result       res;

        // metal fractions (O excluded) and Cs share among metals
        std::vector<double> r(fractions.begin(), fractions.end());
        r[oxygen_index_]  = 0.0;
        double sum_metals = 0.0;
        for (double v : r)
            sum_metals += v;
        for (double& v : r)
            v /= sum_metals;
        const double y_cs = r[caesium_index_];

        // ratios among the rest metals (O and Cs excluded), compared with the training ratios
        if (!rest_ratio_.empty())
        {
            double sum_rest = 0.0;
            for (size_t e = 0; e < ne; ++e)
                if (static_cast<int>(e) != oxygen_index_ && static_cast<int>(e) != caesium_index_)
                    sum_rest += r[e];
            res.rest_ratio.assign(ne, 0.0);
            for (size_t e = 0; e < ne; ++e)
            {
                if (static_cast<int>(e) == oxygen_index_ || static_cast<int>(e) == caesium_index_ || sum_rest <= 0.0)
                    continue;
                res.rest_ratio[e] = r[e] / sum_rest;
                if (rest_ratio_[e] > 0.0)
                    res.rest_ratio_deviation = std::max(res.rest_ratio_deviation,
                                                        std::abs(res.rest_ratio[e] - rest_ratio_[e]) / rest_ratio_[e]);
            }
            if (res.rest_ratio_deviation > rest_ratio_tolerance_)
                res.flags |= RestRatioOff;
        }

        double x_o = 0.0;
        if (mode == OxygenInput::Content)
            x_o = fractions[oxygen_index_] / (sum_metals + fractions[oxygen_index_]);
        else
        {
            double       lo = opot_network_.input_min[1], hi = opot_network_.input_max[1];
            const double mu = oxygen_potential_kj_per_mol_o2;
            if (mu <= opot(y_cs, lo, temperature))
            {
                x_o = lo;
                res.flags |= MuOBelowRange;
            }
            else if (mu >= opot(y_cs, hi, temperature))
            {
                x_o = hi;
                res.flags |= MuOAboveRange;
            }
            else
            {
                for (int it = 0; it < 60; ++it)  // bisection on the monotone OPOT(X_O)
                {
                    const double mid = 0.5 * (lo + hi);
                    if (opot(y_cs, mid, temperature) < mu)
                        lo = mid;
                    else
                        hi = mid;
                }
                x_o = 0.5 * (lo + hi);
            }
        }

        // moles of each element per mole of system
        std::vector<double> b(ne);
        for (size_t e = 0; e < ne; ++e)
            b[e] = r[e] * (1.0 - x_o);
        b[oxygen_index_]  = x_o;
        const double x_cs = y_cs * (1.0 - x_o);
        res.x_o           = x_o;
        res.x_cs          = x_cs;

        // network inputs, projected on the training domain
        std::vector<double> xi = {std::min(std::max(x_cs, xcs_min_), xcs_max_),
                                  std::min(std::max(x_o, xo_min_), xo_max_),
                                  std::min(std::max(temperature, T_min_), T_max_)};
        if (xi[0] + xi[1] > sum_max_)
            xi[0] = std::max(sum_max_ - xi[1], xcs_min_);
        if (xi[0] != x_cs || xi[1] != x_o || xi[2] != temperature)
            res.flags |= InputClamped;

        // stoichiometric phases
        std::vector<double> fs(ns);
        for (size_t j = 0; j < ns; ++j)
            fs[j] = stech_networks_[j].evaluate(xi)[0];
        reduceStoichiometric(fs, b);
        std::vector<double> residual(ne);
        for (size_t e = 0; e < ne; ++e)
        {
            double used = 0.0;
            for (size_t j = 0; j < ns; ++j)
                used += formula_[e][j] * fs[j];
            residual[e] = b[e] - used;
        }

        // partition of the residual (softmax over solution phases + OTHER, for each element)
        const std::vector<double>        z = partition_network_.evaluate(xi);
        std::vector<std::vector<double>> share(ne, std::vector<double>(K));
        for (size_t e = 0; e < ne; ++e)
        {
            double zmax = z[e * K];
            for (size_t k = 1; k < K; ++k)
                zmax = std::max(zmax, z[e * K + k]);
            double s = 0.0;
            for (size_t k = 0; k < K; ++k)
                s += (share[e][k] = std::exp(z[e * K + k] - zmax));
            for (size_t k = 0; k < K; ++k)
                share[e][k] /= s;
        }

        res.moles.assign(groups_.size(), std::vector<double>(ne, 0.0));
        for (size_t g = 0; g < groups_.size(); ++g)
        {
            for (size_t e = 0; e < ne; ++e)
            {
                if (stech_index_[g] >= 0)
                    res.moles[g][e] = fs[stech_index_[g]] * formula_[e][stech_index_[g]];
                else if (solution_index_[g] >= 0)
                    res.moles[g][e] = share[e][solution_index_[g]] * residual[e];
                else
                    res.moles[g][e] = share[e][K - 1] * residual[e];  // OTHER
            }
        }
        return res;
    }

    const Model& sharedModel(const std::string& path)
    {
        static std::map<std::string, Model> models;
        auto                                it = models.find(path);
        if (it == models.end())
        {
            Model       m;
            std::string error;
            if (!m.load(path, error))
            {
                std::cerr << "Error: cannot load the JOG neural network: " << error << std::endl;
                exit(1);
            }
            it = models.emplace(path, m).first;
        }
        return it->second;
    }
}  // namespace JOGSurrogate
