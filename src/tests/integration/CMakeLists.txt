# Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

# Generate helper script for running ATS
configure_file(
    ${PROJECT_SOURCE_DIR}/tests/integration/ats.sh.in
    ${CMAKE_BINARY_DIR}/ats.sh)
