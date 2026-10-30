//////////////////////////////////////////////////////////////////////////////////////
//  JOGSurrogateCoupling: converts the JOGSurrogate result into the same            //
//  OCOutputData structure produced by the OpenCalphad coupling, so that all the    //
//  downstream SCIANTIX routines (thermochemistry variables, release, JOG) are      //
//  unchanged.                                                                       //
//  Author: E. Perugini (MSc thesis, Politecnico di Milano), 2026.                  //
//////////////////////////////////////////////////////////////////////////////////////

#ifndef JOG_SURROGATE_COUPLING_H
#define JOG_SURROGATE_COUPLING_H

#include "JOGSurrogate.h"
#include "OCUtilsCoupling.h"

#include <string>
#include <vector>

namespace JOGSurrogateCoupling
{
    /// Fills output_data (per unit of total input content, like OpenCalphad) from the network.
    /// components: SCIANTIX input components (name, fraction); temperature in K; oxygen potential in kJ/mol O2.
    /// Returns false if an input element is unknown to the network. result (optional) receives X_O and flags.
    bool fillOutputData(const JOGSurrogate::Model&         model,
                        double                             temperature,
                        const std::vector<InputComponent>& components,
                        JOGSurrogate::OxygenInput          oxygen_input,
                        double                             oxygen_potential_kj_per_mol_o2,
                        OCOutputData&                      output_data,
                        JOGSurrogate::Result*              result = nullptr);

    /// Weights file: an absolute path is used as is; otherwise it is searched in the test folder and then in the
    /// data/ folder of the SCIANTIX checkout (found upward from the executable). Exits with an error if not found.
    std::string resolveWeightsPath(const std::string& name, const std::string& test_path);
}  // namespace JOGSurrogateCoupling

#endif
