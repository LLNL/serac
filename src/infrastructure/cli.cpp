// Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "CLI11/CLI11.hpp"
#include "infrastructure/logger.hpp"
#include "infrastructure/terminator.hpp"

namespace serac {

namespace cli {

//------- Command Line Interface -------

std::shared_ptr<std::unordered_map<std::string, std::string>> defineAndParse(int argc, char* argv[], int rank)
{
  // specify all input arguments
  CLI::App    app{"Serac: a high order nonlinear thermomechanical simulation code"};
  std::string input_file_path;
  app.add_option("-i, --input_file", input_file_path, "Input file to use.")->required()->check(CLI::ExistingFile);

  // Parse the arguments and check if they are good
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    serac::logger::flush();
    auto err_msg = (e.get_name() == "CallForHelp") ? app.help() : CLI::FailureMessage::simple(&app, e);
    SLIC_ERROR_ROOT(rank, err_msg);
    serac::exitGracefully();
  }

  // Store found values
  auto cli_opts = std::make_shared<std::unordered_map<std::string, std::string>>();
  cli_opts->insert({std::string("input_file"), input_file_path});

  return cli_opts;
}

void printGiven(std::shared_ptr<std::unordered_map<std::string, std::string>> cli_opts, int rank)
{
  // Add header
  std::string optsMsg = fmt::format("\n{:*^80}\n", "Command Line Options");

  // Add options
  auto search = cli_opts->find("input_file");
  if (search != cli_opts->end()) optsMsg += fmt::format("Input File: {0}\n", search->second);

  // Add footer
  optsMsg += fmt::format("{:*^80}\n", "*");

  SLIC_INFO_ROOT(rank, optsMsg);
  serac::logger::flush();
}

}  // namespace cli
}  // namespace serac
