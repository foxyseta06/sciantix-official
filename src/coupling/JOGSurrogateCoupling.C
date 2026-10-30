//////////////////////////////////////////////////////////////////////////////////////
//  JOGSurrogateCoupling: network result -> OCOutputData (see header).              //
//  Author: E. Perugini (MSc thesis, Politecnico di Milano), 2026.                  //
//////////////////////////////////////////////////////////////////////////////////////

#include "JOGSurrogateCoupling.h"
#include "Constants.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <unistd.h>

namespace JOGSurrogateCoupling
{
    namespace
    {
        std::string upper(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            return s;
        }

        const double liquid_oxide_oxygen_fraction = 0.3;  // O atom fraction above which the liquid is treated as oxide
        const double negligible_moles             = 1.0e-14;

        double atomicMassOf(const std::string& element)
        {
            const auto it = thermochemistry_atomic_masses.find(element);
            if (it == thermochemistry_atomic_masses.end())
            {
                std::cerr << "Error: JOG neural network: atomic mass not available for " << element << std::endl;
                exit(1);
            }
            return it->second;
        }
    }  // namespace

    bool fillOutputData(const JOGSurrogate::Model&         model,
                        double                             temperature,
                        const std::vector<InputComponent>& components,
                        JOGSurrogate::OxygenInput          oxygen_input,
                        double                             oxygen_potential_kj_per_mol_o2,
                        OCOutputData&                      output_data,
                        JOGSurrogate::Result*              result)
    {
        const auto& elements = model.elements();
        const int   io       = model.oxygenIndex();

        // input fractions in the network element order
        std::vector<double> fractions(elements.size(), 0.0);
        for (const auto& c : components)
        {
            size_t e = 0;
            while (e < elements.size() && upper(elements[e]) != upper(c.name)) ++e;
            if (e == elements.size())
            {
                std::cerr << "Error: JOG neural network: element " << c.name << " is not a network input" << std::endl;
                return false;
            }
            fractions[e] = c.fraction;
        }
        double sum_metals = 0.0;
        for (size_t e = 0; e < elements.size(); ++e)
            if (static_cast<int>(e) != io) sum_metals += fractions[e];
        if (sum_metals <= 0.0) return false;

        const JOGSurrogate::Result res = model.evaluate(temperature, fractions, oxygen_input, oxygen_potential_kj_per_mol_o2);
        if (result) *result = res;

        // per mole of system -> same normalization as OpenCalphad (metals equal to their input fractions)
        const double scale = sum_metals / (1.0 - res.x_o);

        output_data.solution_phases.clear();
        output_data.components.clear();
        for (size_t g = 0; g < model.groups().size(); ++g)
        {
            OCPhaseData phase;
            for (size_t e = 0; e < elements.size(); ++e)
            {
                const double n = res.moles[g][e] * scale;
                if (n <= negligible_moles) continue;
                phase.elements[elements[e]]       = n;
                phase.element_masses[elements[e]] = n * atomicMassOf(elements[e]);
                phase.moles += n;
                phase.mass += phase.element_masses[elements[e]];
            }
            if (phase.moles <= negligible_moles) continue;

            const std::string& name = model.groups()[g];
            if (name == "GAS")
            {
                phase.form_units                     = phase.moles;
                output_data.solution_phases["gas"]   = phase;
            }
            else if (name == "LIQUID")
            {
                // The network gives the element composition only. JOGFormation needs the two sublattices of the
                // ionic liquid: the second one tells oxide (no VA) from metallic liquid, and the cation and anion
                // fractions give the oxide-liquid density estimate. Approximate site fractions are built from the
                // element amounts:
                //   oxide liquid (O atom fraction >= 0.3): Mo goes to MoO4-2 anions as far as O allows
                //     (4 O per Mo), the remaining Mo is Mo+4 cation, the remaining O is O-2;
                //   metallic liquid: second sublattice VA.
                // These are not CALPHAD site fractions: they only carry the information JOGFormation uses.
                const auto   element_moles = [&](const std::string& el)
                {
                    const auto it = phase.elements.find(el);
                    return it != phase.elements.end() ? it->second : 0.0;
                };
                const double n_oxygen = element_moles(elements[io]);
                const bool   is_oxide = n_oxygen / phase.moles >= liquid_oxide_oxygen_fraction;
                const double n_mo     = element_moles("Mo");
                const double n_moo4   = is_oxide ? std::min(n_mo, n_oxygen / 4.0) : 0.0;
                const double n_o2     = is_oxide ? n_oxygen - 4.0 * n_moo4 : 0.0;

                static const std::map<std::string, std::string> cation_of = {
                    {"Ba", "BA+2"}, {"Cs", "CS+"}, {"Mo", "MO+4"}, {"Pd", "PD+2"}, {"Rh", "RH+3"}, {"Ru", "RU+4"}, {"Tc", "TC+4"}};
                OCSublatticeData cation;
                cation.index            = 1;
                cation.sites            = 1.0;
                cation.phase_moles      = phase.moles;
                cation.phase_form_units = phase.moles;
                cation.phase_instance   = "LIQUID";
                double n_cations        = 0.0;
                for (const auto& el : phase.elements)
                {
                    const auto c = cation_of.find(el.first);
                    if (c == cation_of.end()) continue;
                    const double n = el.first == "Mo" ? el.second - n_moo4 : el.second;
                    if (n <= negligible_moles) continue;
                    cation.composition[c->second] = n;
                    n_cations += n;
                }
                for (auto& c : cation.composition) c.second /= n_cations;
                cation.constituents_count = static_cast<int>(cation.composition.size());

                OCSublatticeData anion;
                anion.index            = 2;
                anion.sites            = 1.0;
                anion.phase_moles      = phase.moles;
                anion.phase_form_units = phase.moles;
                anion.phase_instance   = "LIQUID";
                if (is_oxide && n_moo4 + n_o2 > negligible_moles)
                {
                    if (n_moo4 > negligible_moles) anion.composition["MOO4-2"] = n_moo4 / (n_moo4 + n_o2);
                    if (n_o2 > negligible_moles) anion.composition["O-2"] = n_o2 / (n_moo4 + n_o2);
                }
                else
                    anion.composition["VA"] = 1.0;
                anion.constituents_count = static_cast<int>(anion.composition.size());

                if (n_cations > negligible_moles) phase.sublattices.push_back(cation);
                phase.sublattices.push_back(anion);
                phase.form_units                      = phase.moles;
                output_data.solution_phases["liquid"] = phase;
            }
            else
            {
                std::string species_name = name;
                if (name == "CS2MOO4")
                    species_name = temperature < model.alphaBetaTemperature() ? "CS2MOO4_S1" : "CS2MOO4_S2";
                OCSpeciesData species;
                species.moles          = phase.moles / model.atomsPerFormula(static_cast<int>(g));  // formula units
                species.mass           = phase.mass;
                species.elements       = phase.elements;
                species.element_masses = phase.element_masses;

                OCPhaseData& condensed = output_data.solution_phases["condensed"];
                condensed.moles += phase.moles;
                condensed.form_units += species.moles;
                condensed.mass += phase.mass;
                for (const auto& el : phase.elements) condensed.elements[el.first] += el.second;
                for (const auto& el : phase.element_masses) condensed.element_masses[el.first] += el.second;
                condensed.species[species_name] = species;
            }
        }
        return true;
    }

    std::string resolveWeightsPath(const std::string& name, const std::string& test_path)
    {
        auto exists = [](const std::string& p) { return !p.empty() && std::ifstream(p).good(); };
        if (!name.empty() && name[0] == '/')
        {
            if (exists(name)) return name;
        }
        else
        {
            if (exists(test_path + name)) return test_path + name;
            // data/ folder of the SCIANTIX checkout, searched upward from the executable
            char          buffer[4096];
            const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
            if (length > 0)
            {
                buffer[length]  = '\0';
                std::string dir = buffer;
                for (size_t cut = dir.find_last_of('/'); cut != std::string::npos && cut > 0; cut = dir.find_last_of('/'))
                {
                    dir = dir.substr(0, cut);
                    if (exists(dir + "/data/" + name)) return dir + "/data/" + name;
                }
            }
        }
        std::cerr << "Error: JOG neural network weights not found: " << name
                  << " (searched: absolute path, test folder, data/ of the SCIANTIX checkout)" << std::endl;
        exit(1);
    }
}  // namespace JOGSurrogateCoupling
