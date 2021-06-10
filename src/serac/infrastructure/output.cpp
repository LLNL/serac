// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/output.hpp"

#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

#include "conduit/conduit.hpp"
#include "ascent/ascent.hpp"
#include "axom/sidre.hpp"

namespace serac::output {

void outputFields(axom::sidre::DataStore& datastore, const std::string& file_name_prefix, double time,
                  const Language language)
{
  SLIC_INFO(fmt::format("Outputting field data at time: {}", time));

  // Configure Ascent to extract data
  ascent::Ascent ascent;
  conduit::Node  ascent_opts;
  ascent_opts["mpi_comm"] = MPI_COMM_WORLD;
  // Do not allow Ascent to use a local file to override actions
  ascent_opts["actions_file"] = "";
  ascent.open(ascent_opts);

  std::string output_language = "";
  if (language == Language::JSON) {
    output_language = "json";
  } else if (language == Language::HDF5) {
    output_language = "hdf5";
  } else if (language == Language::YAML) {
    output_language = "yaml";
  }

  conduit::Node extracts;
  extracts["e1/type"]            = "relay";  // relay is the saving extract
  auto [_, rank]                 = serac::getMPIInfo();
  extracts["e1/params/path"]     = fmt::format("{}.{}.{}", file_name_prefix, rank, output_language);
  extracts["e1/params/protocol"] = output_language;

  // Get domain Sidre group
  // TODO: get this from StateManager directly?
  axom::sidre::Group* sidre_root = datastore.getRoot();
  const std::string   coll_name  = "serac_datacoll";
  SLIC_ERROR_IF(!sidre_root->hasGroup(coll_name), "Data Collection was not found when outputting data!");
  axom::sidre::Group* domain_grp = sidre_root->getGroup(coll_name);

  // Add field names to extract field lists
  SLIC_ERROR_IF(!domain_grp->hasGroup("blueprint/fields"), "Data Collection did not have `fields`!");
  axom::sidre::Group* fields_grp = domain_grp->getGroup("blueprint/fields");
  // TODO: get these from input file
  for (axom::sidre::IndexType idx = fields_grp->getFirstValidGroupIndex(); axom::sidre::indexIsValid(idx);
       idx                        = fields_grp->getNextValidGroupIndex(idx)) {
    axom::sidre::Group* curr_field_grp    = fields_grp->getGroup(idx);
    extracts["e1/params/fields"].append() = curr_field_grp->getName();
  }

  // Create Ascent actions
  conduit::Node  actions;
  conduit::Node& add_extracts = actions.append();
  add_extracts["action"]      = "add_extracts";
  add_extracts["extracts"]    = extracts;

  // Create Conduit node for Ascent to use as a dataset
  conduit::Node domain_grp_node;
  domain_grp->createNativeLayout(domain_grp_node);
  conduit::Node& dataset = domain_grp_node["blueprint"];
  dataset["state/time"]  = time;

  ascent.publish(dataset);
  ascent.execute(actions);
  ascent.close();
}

}  // namespace serac::output
