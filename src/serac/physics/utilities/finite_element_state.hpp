// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file finite_element_state.hpp
 *
 * @brief This file contains the declaration of structure that manages the MFEM objects
 * that make up the state for a given field
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <variant>

#include "mfem.hpp"

#include "axom/sidre/core/MFEMSidreDataCollection.hpp"

#include "serac/infrastructure/logger.hpp"

namespace serac {

namespace detail {

/**
 * @brief A helper type for uniform semantics over owning/non-owning pointers
 *
 * This logic is needed to integrate with the mesh and field reconstruction logic
 * provided by Sidre's MFEMSidreDataCollection.  When a Serac restart occurs, the
 * saved data is used to construct fully functional mfem::(Par)Mesh and
 * mfem::(Par)GridFunction objects.  The FiniteElementCollection and (Par)FiniteElementSpace
 * objects are intermediates in the construction of these objects and are therefore owned
 * by the MFEMSidreDataCollection in the case of a restart/reconstruction.  In a normal run,
 * Serac constructs the mesh and fields, so these FEColl and FESpace objects are owned
 * by Serac.  In both cases, the MFEMSidreDataCollection maintains ownership of the mesh
 * and field objects themselves.
 */
template <typename T>
using MaybeOwningPointer = std::variant<T*, std::unique_ptr<T>>;

/**
 * @brief Retrieves a reference to the underlying object in a MaybeOwningPointer
 * @param[in] obj The object to dereference
 */
template <typename T>
static T& retrieve(MaybeOwningPointer<T>& obj)
{
  return std::visit([](auto&& ptr) -> T& { return *ptr; }, obj);
}
/// @overload
template <typename T>
static const T& retrieve(const MaybeOwningPointer<T>& obj)
{
  return std::visit([](auto&& ptr) -> const T& { return *ptr; }, obj);
}

}  // namespace detail

/**
 * @brief A sum type for encapsulating either a scalar or vector coeffient
 */
using GeneralCoefficient = std::variant<std::shared_ptr<mfem::Coefficient>, std::shared_ptr<mfem::VectorCoefficient>>;

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_scalar_valued(const GeneralCoefficient& coef)
{
  return std::holds_alternative<std::shared_ptr<mfem::Coefficient>>(coef);
}

/**
 * @brief convenience function for querying the type stored in a GeneralCoefficient
 */
inline bool is_vector_valued(const GeneralCoefficient& coef)
{
  return std::holds_alternative<std::shared_ptr<mfem::VectorCoefficient>>(coef);
}

/**
 * @brief Class for encapsulating the critical MFEM components of a solver
 *
 * Namely: Mesh, FiniteElementCollection, FiniteElementState,
 * GridFunction, and a Vector of the solution
 */
class FiniteElementState {
public:
  /**
   * @brief Structure for optionally configuring a FiniteElementState
   */
  // The optionals are explicitly default-constructed to allow the user to partially aggregrate-initialized
  // with only the options they care about
  struct Options {
    /**
     * @brief The polynomial order that should be used for the problem
     */
    int order = 1;
    /**
     * @brief The number of copies of the finite element collections (e.g. vector_dim = 2 or 3 for solid mechanics).
     * Defaults to scalar valued spaces.
     */
    int vector_dim = 1;
    /**
     * @brief The FECollection to use - defaults to an H1_FECollection
     */
    std::unique_ptr<mfem::FiniteElementCollection> coll = {};
    /**
     * The DOF ordering that should be used interally by MFEM
     */
    mfem::Ordering::Type ordering = mfem::Ordering::byVDIM;
    /**
     * @brief The name of the field encapsulated by the state object
     */
    std::string name = "";
    /**
     * @brief Whether the GridFunction should be allocated (and owned by the FEState object)
     */
    bool alloc_gf = true;
  };

  /**
   * Main constructor for building a new state object
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] options The options specified, namely those relating to the order of the problem,
   * the dimension of the FESpace, the type of FEColl, the DOF ordering that should be used,
   * and the name of the field
   */
  FiniteElementState(
      mfem::ParMesh& mesh,
      Options&&      options = {
          .order = 1, .vector_dim = 1, .coll = {}, .ordering = mfem::Ordering::byVDIM, .name = "", .alloc_gf = true});

  /**
   * @brief Minimal constructor for a FiniteElementState given an already-existing field
   * @param[in] mesh The problem mesh (object does not take ownership)
   * @param[in] gf The field for the state to create (object does not take ownership)
   * @param[in] name The name of the field
   */
  FiniteElementState(mfem::ParMesh& mesh, mfem::ParGridFunction& gf, const std::string& name = "");

  /**
   * Returns the MPI communicator for the state
   */
  MPI_Comm comm() const { return detail::retrieve(space_).GetComm(); }

  /**
   * Returns a non-owning reference to the internal grid function
   */
  mfem::ParGridFunction& gridFunc() { return detail::retrieve(gf_); }
  /// \overload
  const mfem::ParGridFunction& gridFunc() const { return detail::retrieve(gf_); }

  /**
   * Returns a GridFunctionCoefficient referencing the internal grid function
   */
  mfem::GridFunctionCoefficient gridFuncCoef() const
  {
    const auto& gf = detail::retrieve(gf_);
    return mfem::GridFunctionCoefficient{&gf, gf.VectorDim()};
  }

  /**
   * Returns a VectorGridFunctionCoefficient referencing the internal grid function
   */
  mfem::VectorGridFunctionCoefficient vectorGridFuncCoef() const
  {
    return mfem::VectorGridFunctionCoefficient{&detail::retrieve(gf_)};
  }

  /**
   * Returns a non-owning reference to the internal mesh object
   */
  mfem::ParMesh& mesh() { return mesh_; }

  /**
   * Returns a non-owning reference to the internal FESpace
   */
  mfem::ParFiniteElementSpace& space() { return detail::retrieve(space_); }
  /// \overload
  const mfem::ParFiniteElementSpace& space() const { return detail::retrieve(space_); }

  /**
   * Returns a non-owning reference to the vector of true DOFs
   */
  mfem::HypreParVector& trueVec() { return true_vec_; }

  /**
   * Returns the name of the FEState (field)
   */
  std::string name() const { return name_; }

  /**
   * Projects a coefficient (vector or scalar) onto the field
   * @param[in] coef The coefficient to project
   */
  void project(const GeneralCoefficient& coef)
  {
    // The generic lambda parameter, auto&&, allows the component type (mfem::Coef or mfem::VecCoef)
    // to be deduced, and the appropriate version of ProjectCoefficient is dispatched.
    std::visit([this](auto&& concrete_coef) { detail::retrieve(gf_).ProjectCoefficient(*concrete_coef); }, coef);
  }
  /// \overload
  void project(mfem::Coefficient& coef) { detail::retrieve(gf_).ProjectCoefficient(coef); }
  /// \overload
  void project(mfem::VectorCoefficient& coef) { detail::retrieve(gf_).ProjectCoefficient(coef); }

  /**
   * Initialize the true DOF vector by extracting true DOFs from the internal
   * grid function into the internal true DOF vector
   */
  void initializeTrueVec() { detail::retrieve(gf_).GetTrueDofs(true_vec_); }

  /**
   * Set the internal grid function using the true DOF values
   */
  void distributeSharedDofs() { detail::retrieve(gf_).SetFromTrueDofs(true_vec_); }

  /**
   * Utility function for creating a tensor, e.g. mfem::HypreParVector,
   * mfem::ParBilinearForm, etc on the FESpace encapsulated by an FEState object
   * @return An owning pointer to a heap-allocated tensor
   * @pre Tensor must have the constructor Tensor::Tensor(ParFiniteElementSpace*)
   */
  template <typename Tensor>
  std::unique_ptr<Tensor> createOnSpace()
  {
    static_assert(std::is_constructible_v<Tensor, mfem::ParFiniteElementSpace*>,
                  "Tensor must be constructible with a ptr to ParFESpace");
    return std::make_unique<Tensor>(&detail::retrieve(space_));
  }

private:
  /**
   * @brief A reference to the mesh object on which the field is defined
   */
  std::reference_wrapper<mfem::ParMesh> mesh_;
  /**
   * @brief Possibly-owning handle to the FiniteElementCollection, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   * @note Must be const as FESpaces store a const reference to their FEColls
   */
  detail::MaybeOwningPointer<const mfem::FiniteElementCollection> coll_;
  /**
   * @brief Possibly-owning handle to the FiniteElementCollection, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   */
  detail::MaybeOwningPointer<mfem::ParFiniteElementSpace> space_;
  /**
   * @brief Possibly-owning handle to the ParGridFunction, as it is owned
   * by the FiniteElementState in a normal run and by the MFEMSidreDataCollection
   * in a restart run
   */
  detail::MaybeOwningPointer<mfem::ParGridFunction> gf_;
  mfem::HypreParVector                              true_vec_;
  std::string                                       name_ = "";
};

/**
 * @brief A shim class for describing the interface of something that can be synced
 */
class SyncableData {
public:
  virtual ~SyncableData() = default;
  virtual void sync()     = 0;
};

/**
 * @brief Stores instances of user-defined type for each quadrature point in a mesh
 * @tparam T The type of the per-qpt data
 * @pre T must be default-constructible (TODO: Do we want to allow non-default constructible types?)
 * @pre T must be trivially copyable (due to the use of memcpy for type punning)
 */
template <typename T>
class QuadratureData : public SyncableData {
public:
  /**
   * @brief Constructs using a mesh and polynomial order
   * @param[in] mesh The mesh for which quadrature-point data should be stored
   * @param[in] p The polynomial order of the associated finite elements
   */
  QuadratureData(mfem::Mesh& mesh, const int p, const bool alloc = true);

  QuadratureData(mfem::QuadratureFunction& qfunc)
      : qspace_(qfunc.GetSpace()), qfunc_(&qfunc), data_(qfunc.Size() / stride_)
  {
    const double* qfunc_ptr = detail::retrieve(qfunc_).GetData();
    int           j         = 0;
    T*            data_ptr  = data_.data();
    for (int i = 0; i < detail::retrieve(qfunc_).Size(); i += stride_) {
      // The only legal (portable, defined) way to do type punning in C++
      std::memcpy(data_ptr + j, qfunc_ptr + i, sizeof(T));
      j++;
    }
  }

  /**
   * @brief Retrieves the data for a given quadrature point
   * @param[in] element_idx The index of the desired element within the mesh
   * @param[in] q_idx The index of the desired quadrature point within the element
   */
  T& operator()(const int element_idx, const int q_idx);

  /**
   * @brief Assigns an item to each quadrature point
   * @param[in] item The item to assign
   */
  QuadratureData& operator=(const T& item);

  /**
   * @brief Iterator to the data for the first quadrature point
   */
  auto begin() { return data_.begin(); }
  /// @overload
  auto begin() const { return data_.begin(); }
  /**
   * @brief Iterator to one element past the data for the last quadrature point
   */
  auto end() { return data_.end(); }
  /// @overload
  auto end() const { return data_.end(); }

  mfem::QuadratureFunction& QFunc() { return detail::retrieve(qfunc_); }

  /**
   * @brief Synchronizes data from the stored vector<T> to the raw double*
   * array used by the underlying mfem::QuadratureFunction
   *
   * Used for saving to a file - MFEMSidreDataCollection
   * (and by extension mfem::DataCollection's interface) only allow for
   * quadrature-point-specific data via mfem::QuadratureFunction, so this logic
   * is needed to glue together a generic array of data with that class
   */
  void sync() override
  {
    double*  qfunc_ptr = detail::retrieve(qfunc_).GetData();
    int      j         = 0;
    const T* data_ptr  = data_.data();
    for (int i = 0; i < detail::retrieve(qfunc_).Size(); i += stride_) {
      // The only legal (portable, defined) way to do type punning in C++
      std::memcpy(qfunc_ptr + i, data_ptr + j, sizeof(T));
      j++;
    }
  }

private:
  // FIXME: These will probably need to be MaybeOwningPointers
  // See https://github.com/LLNL/axom/pull/433
  /**
   * @brief Storage layout of @p qfunc_ containing mesh and polynomial order info
   */
  detail::MaybeOwningPointer<mfem::QuadratureSpace> qspace_;
  /**
   * @brief Per-quadrature point data, stored as array of doubles for compatibility with Sidre
   */
  detail::MaybeOwningPointer<mfem::QuadratureFunction> qfunc_;

  std::vector<T> data_;
  /**
   * @brief The stride of the array
   */
  static constexpr int stride_ = sizeof(T) / sizeof(double);
};

/**
 * @brief "Dummy" specialization, intended to be used as sentinel
 */
template <>
class QuadratureData<void> {
};

// A dummy global so that lvalue references can be bound to something of type QData<void>
// FIXME: There's probably a cleaner way to do this, it's technically a non-const global
// but it's not really mutable because no operations are defined for it
extern QuadratureData<void> dummy_qdata;

// Hijacks the "vdim" parameter (number of doubles per qpt) to allocate the correct amount of storage
template <typename T>
QuadratureData<T>::QuadratureData(mfem::Mesh& mesh, const int p, const bool alloc)
    : qspace_(std::make_unique<mfem::QuadratureSpace>(&mesh, p + 1)),
      // When left unallocated, the allocation can happen inside the datastore
      // Use a raw pointer here when unallocated, lifetime will be managed by the DataCollection
      qfunc_(alloc ? detail::MaybeOwningPointer<mfem::QuadratureFunction>{std::make_unique<mfem::QuadratureFunction>(
                         &detail::retrieve(qspace_), stride_)}
                   : detail::MaybeOwningPointer<mfem::QuadratureFunction>{new mfem::QuadratureFunction(
                         &detail::retrieve(qspace_), nullptr, stride_)}),
      data_(detail::retrieve(qfunc_).Size() / stride_)
{
  // To avoid violating C++'s strict aliasing rule we need to std::memcpy a default-constructed object
  // See e.g. https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8
  // also https://en.cppreference.com/w/cpp/numeric/bit_cast
  // also https://chromium.googlesource.com/chromium/src/base/+/refs/heads/master/bit_cast.h
  static_assert(std::is_default_constructible_v<T>, "Must be able to default-construct the stored type");
  static_assert(std::is_trivially_copyable_v<T>, "Uses memcpy - requires trivial copies");
}

template <typename T>
T& QuadratureData<T>::operator()(const int element_idx, const int q_idx)
{
  // A view into the quadrature point data
  mfem::Vector view;
  detail::retrieve(qfunc_).GetElementValues(element_idx, q_idx, view);
  double*    end_ptr   = view.GetData();
  double*    start_ptr = detail::retrieve(qfunc_).GetData();
  const auto idx       = (end_ptr - start_ptr) / stride_;
  return data_[idx];
}

template <typename T>
QuadratureData<T>& QuadratureData<T>::operator=(const T& item)
{
  data_.assign(data_.size(), item);
  return *this;
}

/**
 * @brief Manages the lifetimes of FEState objects such that restarts are abstracted
 * from physics modules
 */
class StateManager {
public:
  /**
   * @brief Initializes the StateManager with a sidre DataStore (into which state will be written/read)
   * @param[in] ds The DataStore to use
   * @param[in] collection_name_prefix The prefix for the name of the Sidre DataCollection to be created
   * @param[in] cycle_to_load The cycle to load - required for restarts
   */
  static void initialize(axom::sidre::DataStore& ds, const std::string& collection_name_prefix = "serac",
                         const std::optional<int> cycle_to_load = {});

  /**
   * @brief Factory method for creating a new FEState object, signature is identical to FEState constructor
   * @param[in] options Configuration options for the FEState, if a new state is created
   * @see FiniteElementState::FiniteElementState
   * @note If this is a restart then the options (except for the name) will be ignored
   */
  static FiniteElementState newState(FiniteElementState::Options&& options = {});

  template <typename T>
  static QuadratureData<T>& newQuadratureData(const std::string& name, const int p)
  {
    if (is_restart_) {
      auto field = datacoll_->GetQField(name);
      syncable_data_.push_back(std::make_unique<QuadratureData<T>>(*field));
      // return {*field};
      return static_cast<QuadratureData<T>&>(*syncable_data_.back());
    } else {
      SLIC_ERROR_ROOT_IF(datacoll_->HasQField(name),
                         fmt::format("Serac's datacollection was already given a qfield named '{0}'", name));
      syncable_data_.push_back(std::make_unique<QuadratureData<T>>(mesh(), p, false));
      // The static_cast is safe here because we "know" what we just inserted into the vector
      auto& qdata = static_cast<QuadratureData<T>&>(*syncable_data_.back());
      datacoll_->RegisterQField(name, &(qdata.QFunc()));
      return qdata;
    }
  }

  /**
   * @brief Updates the Conduit Blueprint state in the datastore and saves to a file
   * @param[in] t The current sim time
   * @param[in] cycle The current iteration number of the simulation
   */
  static void save(const double t, const int cycle);

  /**
   * @brief Resets the underlying global datacollection object
   */
  static void reset()
  {
    datacoll_.reset();
    is_restart_ = false;
    syncable_data_.clear();
  };

  /**
   * @brief Gives ownership of mesh to StateManager
   */
  static void setMesh(std::unique_ptr<mfem::ParMesh> mesh);

  /**
   * @brief Returns a non-owning reference to mesh held by StateManager
   */
  static mfem::ParMesh& mesh();

  /**
   * @brief Returns the Sidre DataCollection name
   */
  static const std::string collectionName() { return collection_name_; }

private:
  /**
   * @brief The datacollection instance
   *
   * The std::optional is used here to allow for deferred construction on the stack.
   * The object is constructed when the user calls StateManager::initialize.
   */
  static std::optional<axom::sidre::MFEMSidreDataCollection> datacoll_;
  /**
   * @brief Whether this simulation has been restarted from another simulation
   */
  static bool is_restart_;
  /**
   * @brief Name of the Sidre DataCollection
   */
  static std::string                                collection_name_;
  static std::vector<std::unique_ptr<SyncableData>> syncable_data_;
};

}  // namespace serac
