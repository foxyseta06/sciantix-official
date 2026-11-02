//////////////////////////////////////////////////////////////////////////////////////
//       _______.  ______  __       ___      .__   __. .___________. __  ___   ___  //
//      /       | /      ||  |     /   \     |  \ |  | |           ||  | \  \ /  /  //
//     |   (----`|  ,----'|  |    /  ^  \    |   \|  | `---|  |----`|  |  \  V  /   //
//      \   \    |  |     |  |   /  /_\  \   |  . `  |     |  |     |  |   >   <    //
//  .----)   |   |  `----.|  |  /  _____  \  |  |\   |     |  |     |  |  /  .  \   //
//  |_______/     \______||__| /__/     \__\ |__| \__|     |__|     |__| /__/ \__\  //
//                                                                                  //
//  Originally developed by D. Pizzocri & T. Barani                                 //
//                                                                                  //
//  Version: 2.5                                                                    //
//  Year: 2026                                                                      //
//  Authors: D. Pizzocri, G. Zullo, E. Cappellari.                                  //
//                                                                                  //
//////////////////////////////////////////////////////////////////////////////////////

#ifndef THERMOCHEMISTRY_SETTINGS_H
#define THERMOCHEMISTRY_SETTINGS_H

#include <string>
#include <vector>

struct ThermochemistryPhaseSettings
{
    std::string              module;
    std::string              database;
    std::vector<std::string> elements;
    std::vector<std::string> locations;
};

struct ThermochemistrySettings
{
    std::string                  opencalphad_path                    = "";
    bool                         output_phase_sublattice_composition = false;
    ThermochemistryPhaseSettings fission_products;
    ThermochemistryPhaseSettings matrix;

    // Lazy re-equilibration: SetPhaseDiagram() skips the OpenCalphad solve and reuses the
    // last converged (normalized) result whenever the driving conditions have not moved
    // by more than these tolerances since that solve, up to coupling_max_stale_steps.
    double coupling_temperature_tolerance = 5.0;     // K, absolute
    double coupling_composition_tolerance = 1.0e-3;  // relative
    int    coupling_max_stale_steps       = 20;

    // CODE DEVELOPMENT : JOG NEURAL NETWORK (E. Perugini)
    // Weights of the JOG ANN, used when fission_products.module = NEURALNETWORK (grain boundary only).
    std::string neuralnetwork_path = "JOG_ANN.txt";  // absolute, or searched in the test folder and in data/
    // The network is evaluated at every time step: the cache of the previous equilibrium is not used,
    // unless this flag is set (kept only to measure its effect).
    bool neuralnetwork_use_cache = false;
};

ThermochemistrySettings LoadThermochemistrySettings(const std::string& path);

#endif
