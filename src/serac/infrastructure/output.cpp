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

#include "mpi.h"

namespace serac::output {

void outputCurves(const axom::sidre::DataStore& datastore, const std::string& data_collection_name,
                  const Language language)
{
  auto [_, rank] = getMPIInfo();
  if (rank != 0) {
     return;
  }

  std::string output_language = "";
  if (language == Language::JSON) {
    output_language = "json";
  } else if (language == Language::HDF5) {
    // FIXME: remove this error when implemented, do we even want hdf5 curves?
    SLIC_ERROR_ROOT("hdf5 has not been implemented in outputCurves");
    output_language = "hdf5";
  } else if (language == Language::YAML) {
    output_language = "yaml";
  }

  const std::string file_name = fmt::format("{0}_curves.{1}", data_collection_name, output_language);
  datastore.getRoot()->getGroup("serac_curves")->save(file_name, output_language);
}

void outputFields(const axom::sidre::DataStore& datastore, const std::string& data_collection_name, double time,
                  const Language language)
{
  SLIC_INFO_ROOT(fmt::format("Outputting field data at time: {}", time));

  // Configure Ascent to extract data
  ascent::Ascent ascent;
  conduit::Node  ascent_opts;
  // Use MPI function that always returns an int type for the communicator
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
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
  // "relay" is the Ascents Extract type for saving data
  extracts["e1/type"]            = "relay";
  extracts["e1/params/path"]     = fmt::format("{}_fields.{}", data_collection_name, output_language);
  extracts["e1/params/protocol"] = output_language;

  // Get domain Sidre group
  const axom::sidre::Group* sidre_root = datastore.getRoot();
  SLIC_ERROR_ROOT_IF(
      !sidre_root->hasGroup(data_collection_name),
      fmt::format("Expected a Sidre Data Collection root at '{0}' but it was not found", data_collection_name));
  const axom::sidre::Group* domain_grp = sidre_root->getGroup(data_collection_name);

  // Add field names to extract field lists
  SLIC_ERROR_ROOT_IF(!domain_grp->hasGroup("blueprint/fields"), "Data Collection did not have `fields`!");
  const axom::sidre::Group* fields_grp = domain_grp->getGroup("blueprint/fields");
  // TODO: get these from input file
  for (axom::sidre::IndexType idx = fields_grp->getFirstValidGroupIndex(); axom::sidre::indexIsValid(idx);
       idx                        = fields_grp->getNextValidGroupIndex(idx)) {
    const axom::sidre::Group* curr_field_grp = fields_grp->getGroup(idx);
    extracts["e1/params/fields"].append()    = curr_field_grp->getName();
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
