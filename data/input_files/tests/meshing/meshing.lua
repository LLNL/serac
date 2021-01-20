main_mesh_from_file = {
    -- mesh file
    mesh = "../../../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

main_mesh_cuboid = {
    -- mesh file
    elements = {x = 3, y = 3, z = 3},
    size = {x = 1, y = 2, z = 3},
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

main_mesh_rect = {
    -- mesh file
    elements = {x = 3, y = 3},
    ser_ref_levels = 1,
    par_ref_levels = 0,
}