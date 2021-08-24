// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/cli.hpp"

#include "CLI11/CLI11.hpp"

#include "serac/infrastructure/input.hpp"
#include "serac/infrastructure/logger.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac::cli {

//------- Command Line Interface -------

std::unordered_map<std::string, std::string> defineAndParse(int argc, char* argv[], std::string app_description)
{
  // specify all input arguments
  CLI::App    app{app_description};
  std::string input_file_path;
  app.add_option("-i, --input-file", input_file_path, "Input file to use")->required()->check(CLI::ExistingFile);
  int  restart_cycle;
  auto restart_opt =
      app.add_option("-c, --restart-cycle", restart_cycle, "Cycle to restart from")->check(CLI::PositiveNumber);
  bool create_input_file_docs{false};
  app.add_flag("-d, --create-input-file-docs", create_input_file_docs,
               "Writes Sphinx documentation for input file, then exits");
  bool output_fields{false};
  app.add_flag("--output-fields", output_fields, "Writes field data to file system");
  std::string output_directory;
  app.add_option("-o, --output-directory", output_directory, "Directory to put outputted files");

  // Parse the arguments and check if they are good
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    serac::logger::flush();
    if (e.get_name() == "CallForHelp") {
      auto msg = app.help();
      SLIC_INFO_ROOT(msg);
      serac::exitGracefully();
    } else {
      auto err_msg = CLI::FailureMessage::simple(&app, e);
      SLIC_ERROR_ROOT(err_msg);
    }
  }

  // Store found values and set defaults if not set above
  std::unordered_map<std::string, std::string> cli_opts;
  cli_opts.insert({std::string("input-file"), input_file_path});
  // If a restart cycle was specified
  if (restart_opt->count() > 0) {
    cli_opts["restart-cycle"] = std::to_string(restart_cycle);
  }
  if (create_input_file_docs) {
    cli_opts.insert({"create-input-file-docs", {}});
  }
  if (output_fields) {
    cli_opts.insert({"output-fields", {}});
  }
  if (output_directory == "") {
    // if given by user use that otherwise use input file's basename minus extension
    output_directory = serac::input::getInputFileName(input_file_path);
  }
  cli_opts.insert({"output-directory", output_directory});

  return cli_opts;
}

std::string cliValueToString(std::string value) { return value; }

std::string cliValueToString(bool value) { return value ? "true" : "false"; }

std::string cliValueToString(int value) { return std::to_string(value); }

void printGiven(std::unordered_map<std::string, std::string>& cli_opts)
{
  // Add header
  std::string optsMsg = fmt::format("\n{:*^80}\n", "Command Line Options");

  // Create options map
  // clang-format off
  std::vector<std::pair<std::string, std::string>> opts_output_map{
    {"create-input-file-docs", "Create Input File Docs"},
    {"input-file", "Input File"},
    {"output-directory", "Output Directory"},
    {"output-fields", "Output Fields"},
    {"restart-cycle", "Restart Cycle"}};
  // clang-format on

  // Add options to string
  for (auto output_pair : opts_output_map) {
    auto search = cli_opts.find(output_pair.first);
    if (search != cli_opts.end()) {
      optsMsg += fmt::format("{0}: {1}\n", output_pair.second, cliValueToString(search->second));
    }
  }

  // Add footer
  optsMsg += fmt::format("{:*^80}\n", "*");

  SLIC_INFO_ROOT(optsMsg);
  serac::logger::flush();
}

}  // namespace serac::cli
