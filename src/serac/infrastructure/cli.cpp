// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/cli.hpp"

#include "CLI11/CLI11.hpp"

#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac::cli {

//------- Command Line Interface -------

std::unordered_map<std::string, std::string> defineAndParse(int argc, char* argv[], int rank,
                                                            std::string app_description)
{
  // specify all input arguments
  CLI::App    app{app_description};
  std::string input_file_path;
  app.add_option("-i, --input_file", input_file_path, "Input file to use.")->required()->check(CLI::ExistingFile);
  int  restart_cycle;
  auto restart_opt =
      app.add_option("-c, --restart_cycle", restart_cycle, "Cycle to restart from.")->check(CLI::PositiveNumber);
  bool create_input_file_docs{false};
  app.add_flag("-d, --create-input-file-docs", create_input_file_docs,
               "Writes Sphinx documentation for input file, then exits");

  // Parse the arguments and check if they are good
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    serac::logger::flush();
    if (e.get_name() == "CallForHelp") {
      auto msg = app.help();
      SLIC_INFO_ROOT(rank, msg);
      serac::exitGracefully();
    } else {
      auto err_msg = CLI::FailureMessage::simple(&app, e);
      SLIC_ERROR_ROOT(rank, err_msg);
    }
  }

  // Store found values
  std::unordered_map<std::string, std::string> cli_opts;
  cli_opts.insert({std::string("input_file"), input_file_path});
  if (restart_opt->count()) {
    cli_opts["restart_cycle"] = std::to_string(restart_cycle);
  }
  if (create_input_file_docs) {
    cli_opts.insert({"create_input_file_docs", {}});
  }

  return cli_opts;
}

void printGiven(std::unordered_map<std::string, std::string>& cli_opts, int rank)
{
  // Add header
  std::string optsMsg = fmt::format("\n{:*^80}\n", "Command Line Options");

  // Add options
  auto search = cli_opts.find("input_file");
  if (search != cli_opts.end()) optsMsg += fmt::format("Input File: {0}\n", search->second);

  // Add footer
  optsMsg += fmt::format("{:*^80}\n", "*");

  SLIC_INFO_ROOT(rank, optsMsg);
  serac::logger::flush();
}

}  // namespace serac::cli
