#pragma once

#include "mfem.hpp"

#include "serac/numerics/functional/finite_element.hpp"

namespace serac {

    struct FunctionSpace {

        enum class Type {
            FiniteElement,
            Uniform
        };

        template < typename space >
        FunctionSpace(mfem::ParMesh * mesh, space) {

          if (space::family == Family::UNIFORM) {

            type = Type::Uniform;

          } else {

            type = Type::FiniteElement;

            const int  dim = mesh->Dimension();
            const auto ordering = mfem::Ordering::byNODES;

            switch (space::family) {
              case Family::H1:
                fec = std::make_unique<mfem::H1_FECollection>(space::order, dim);
                break;
              case Family::HCURL:
                fec = std::make_unique<mfem::ND_FECollection>(space::order, dim);
                break;
              case Family::HDIV:
                fec = std::make_unique<mfem::RT_FECollection>(space::order, dim);
                break;
              case Family::L2:
                // We use GaussLobatto basis functions as this is what is used for the serac::Functional FE kernels
                fec = std::make_unique<mfem::L2_FECollection>(space::order, dim, mfem::BasisType::GaussLobatto);
                break;
              default:
                break;
            }

            fes = std::make_unique<mfem::ParFiniteElementSpace>(mesh, fec.get(), space::components, ordering);

          }

        }

        const mfem::Operator* GetProlongationMatrix() const {
            if (type == Type::FiniteElement) {
                return fes->GetProlongationMatrix();
            } else {
                // TODO
                return nullptr;
            }
        }

        int GetTrueVSize() const {
            if (type == Type::FiniteElement) {
                return fes->GetTrueVSize();
            } else {
                // TODO
                return 0;
            }
        }

        mfem::HypreParMatrix *Dof_TrueDof_Matrix() const { 
            if (type == Type::FiniteElement) {
                return fes->Dof_TrueDof_Matrix();
            } else {
                // TODO
                return nullptr;
            }
        }

        auto GetComm() const {
            if (type == Type::FiniteElement) {
                return fes->GetComm();
            } else {
                // TODO
                return MPI_COMM_WORLD;
            }
        }

        int GlobalVSize() const {
            if (type == Type::FiniteElement) {
                return fes->GlobalVSize();
            } else {
                // TODO
                return 0;
            }
        }

        HYPRE_BigInt* GetDofOffsets() const {
            if (type == Type::FiniteElement) {
                return fes->GetDofOffsets();
            } else {
                // TODO
                return nullptr;
            }
        }

        Type type;

        // these are used when type == Type::FiniteElement
        std::unique_ptr < mfem::ParFiniteElementSpace > fes;
        std::unique_ptr < mfem::FiniteElementCollection > fec;

    };

}