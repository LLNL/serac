#pragma once

#include <fstream>
#include <iostream>
#include <sys/stat.h>

#include "mfem.hpp"


#include "axom/slic/core/SimpleLogger.hpp"
#include "serac/infrastructure/input.hpp"
#include "serac/serac_config.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/mesh_utils_base.hpp"
#include "serac/physics/operators/stdfunction_operator.hpp"
#include "serac/physics/utilities/functional/functional.hpp"
#include "serac/physics/utilities/functional/tensor.hpp"
#include "serac/numerics/assembled_sparse_matrix.hpp"
#include "serac/infrastructure/profiling.hpp"
#include <gtest/gtest.h>
