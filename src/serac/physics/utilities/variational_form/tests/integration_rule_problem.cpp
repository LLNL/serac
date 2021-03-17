#include "mfem.hpp"

#include "serac/serac_config.hpp"

auto get_quadrature_point_positions(mfem::Mesh & mesh, const mfem::FiniteElementSpace & fes) {

  const mfem::FiniteElement& el = *(fes.GetFE(0));

  const mfem::IntegrationRule ir = mfem::IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);

  std::cout << "integration rule with " << ir.Size() << " points, &ir = " << &ir << std::endl;

  auto geom = mesh.GetGeometricFactors(ir, mfem::GeometricFactors::COORDINATES);

// const GeometricFactors* Mesh::GetGeometricFactors(const IntegrationRule& ir,
//                                                   const int flags)
// {
//    for (int i = 0; i < geom_factors.Size(); i++)
//    {
//       GeometricFactors *gf = geom_factors[i];
//       if (gf->IntRule == &ir && (gf->computed_factors & flags) == flags)
//                       ^^
//                        | 
//                       pointer equality does not imply integration rule equality
//
//       {
//          return gf;
//       }
//    }
// 
//    this->EnsureNodes();
// 
//    GeometricFactors *gf = new GeometricFactors(this, ir, flags);
//    geom_factors.Append(gf);
//    return gf;
// }

  return geom->X;

}

int main() {

  const char * mesh_file = SERAC_REPO_DIR"/data/meshes/star.mesh";

  mfem::Mesh mesh(mesh_file, 1, 1);

  std::cout << "mesh has " << mesh.GetNE() << " elements" << std::endl;

  // expected behavior: 
  // different polynomial orders should change the integration rule 
  // and GetGeometricFactors should return new coordinates for each rule
  // 
  // actual behavior: 
  // different polynomial orders DO change the integration rule 
  // but GetGeometricFactors only computes the coordinates the first time,
  // and just returns those values on subsequent calls (with different integration rules)
  for (int order = 1; order < 5; order++) {
    auto fec = mfem::H1_FECollection(order, mesh.Dimension());
    mfem::FiniteElementSpace fespace(&mesh, &fec);
    mfem::Vector X = get_quadrature_point_positions(mesh, fespace);
    std::cout << "order " << order << ", X.size() = " << X.Size() << std::endl;
  }

}