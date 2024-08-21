// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <thread>
#include <fstream>

#include "mfem.hpp"

#include "axom/CLI11.hpp"
#include "axom/core/utilities/Timer.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/serac_config.hpp"

using timer = axom::utilities::Timer;

//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{

  axom::CLI::App app{"a tool for partitioning large meshes into smaller parts suitable for MPI"};

  std::string input_mesh;
  app.add_option("-i, --input-mesh", input_mesh, "Input file to use")->required()->check(axom::CLI::ExistingFile);

  std::string output_prefix;
  app.add_option("-o, --output-prefix", output_prefix, "Prefix for output meshes")->required();

  int num_parts = 4;
  app.add_option("-n, --num_parts", num_parts, "number of partitions to generate");

  CLI11_PARSE(app, argc, argv);

  timer stopwatch;

  std::cout << "reading in mesh file ... ";
  stopwatch.start();
  std::ifstream infile(input_mesh);
  mfem::Mesh mesh(infile);
  stopwatch.stop();
  std::cout << "completed after " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;

  std::cout << "partitioning mesh into " << num_parts << " pieces ... ";
  stopwatch.start();
  int * partitioning = mesh.GeneratePartitioning(num_parts);
  stopwatch.stop();
  std::cout << "completed after " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;

  mfem::MeshPartitioner partitioner(mesh, num_parts, partitioning);

  mfem::MeshPart mesh_part;

  for (int i = 0; i < num_parts; i++) {
    std::string filename = mfem::MakeParFilename(output_prefix + ".mesh.", i);
    std::cout << "extracting part " << i << " and writing it to " << filename << " ... ";
    stopwatch.start();
    partitioner.ExtractPart(i, mesh_part);
    std::ofstream f(filename);
    f.precision(16);
    mesh_part.Print(f);
    stopwatch.stop();
    std::cout << "completed after " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;
  }

  delete partitioning;

}
