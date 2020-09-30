// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "infrastructure/input.hpp"

namespace serac {

namespace input {

std::shared_ptr<axom::inlet::Inlet> initialize(std::shared_ptr<axom::sidre::DataStore> datastore,
                                               const std::string& input_file_path)
{
  // Initialize Inlet
  auto luareader = std::make_shared<axom::inlet::LuaReader>();
  luareader->parseFile(input_file_path);

  // Store inlet data under its own group
  axom::sidre::Group* inlet_root = datastore->getRoot()->createGroup("input_file");
  auto serac_inlet = std::make_shared<axom::inlet::Inlet>(luareader, inlet_root);

  return serac_inlet;
}


}  // namespace input
}  // namespace serac
