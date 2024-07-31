// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file common.hpp
 *
 * @brief A file defining some enums and structs that are used by the different physics modules
 */
#pragma once

namespace serac {

/**
 * @brief a struct that is used in the physics modules to clarify which template arguments are
 * user-controlled parameters (e.g. for design optimization)
 */
template <typename... T>
struct Parameters {
  static constexpr int n = sizeof...(T);  ///< how many parameters were specified
};

/**
 * @brief Enum to set the geometric nonlinearity flag
 *
 */
enum class GeometricNonlinearities
{
  On, /**< Include geometric nonlinearities */
  Off /**< Do not include geometric nonlinearities */
};

namespace detail {

/**
 * @brief Prepends a prefix to a target string if @p name is non-empty with an
 * underscore delimiter
 * @param[in] prefix The string to prepend
 * @param[in] target The string to prepend to
 */
inline std::string addPrefix(const std::string& prefix, const std::string& target)
{
  if (prefix.empty()) {
    return target;
  }
  return prefix + "_" + target;
}

/**
 * @brief Removes a prefix and the underscore delimiter from a target string
 * @param[in] prefix The prefix string to remove
 * @param[in] target The larger string to remove the prefix from
 */
inline std::string removePrefix(const std::string& prefix, const std::string& target)
{
  std::string modified_target{target};
  // Ensure the prefix isn't an empty string
  if (!prefix.empty()) {
    // Ensure the prefix is at the beginning of the string
    auto index = modified_target.find(prefix + "_");
    if (index == 0) {
      // Remove the prefix
      modified_target.erase(0, prefix.size() + 1);
    }
  }
  return modified_target;
}

}  // namespace detail

}  // namespace serac
