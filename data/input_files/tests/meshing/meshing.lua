main_mesh_from_file = {
    type = "file",
    -- mesh file
    mesh = "../../../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0,
}

main_mesh_cuboid = {
    type = "box",
    -- mesh file
    elements = {x = 3, y = 3, z = 3},
    size = {x = 1, y = 2, z = 3},
    ser_ref_levels = 0,
    par_ref_levels = 0,
}

main_mesh_rect = {
    type = "box",
    elements = {x = 3, y = 3},
    ser_ref_levels = 0,
    par_ref_levels = 0,
}

main_mesh_fail = {
    type = "invalid", -- this is invalid
    mesh = "../../../meshes/beam-hex.mesh",
    -- serial and parallel refinement levels
    ser_ref_levels = 1,
    par_ref_levels = 0
}
