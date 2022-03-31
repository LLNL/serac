def toss4_spec(gpu_str):
    return " amdgpu_target=" + gpu_str + " ^raja~openmp+rocm ^umpire~openmp+rocm"

def specs():
    conduit = " ^conduit@0.7.2axom~shared~test"
    hdf5 = " ^hdf5~shared~mpi"
    hypre = " ^hypre@2.20.0"
    mfem = " ^mfem@4.2.0"
    raja = " ^raja~shared~examples~exercises"
    umpire = " ^umpire~shared~examples"

    base_tpl = conduit + hdf5 + hypre + mfem + raja + umpire
    compiler_common = "~cpp14+devtools+mfem+c2c"
    scr = "+scr ^scr@3.0rc2~shared~tests~examples"
    cuda = "+cuda cuda_arch=70"
    omp = "~openmp"

    spec_lists = {
        "toss_3_x86_64_ib": [
            "clang@9.0.0" + compiler_common + base_tpl,
            "clang@10.0.0" + compiler_common + scr + base_tpl,
            "gcc@8.3.1" + compiler_common + scr + base_tpl,
            "gcc@8.1.0~cpp14~fortran+devtools~mfem+c2c" + scr + base_tpl,
            "intel@18.0.2" + compiler_common + omp + base_tpl,
            "intel@19.0.4" + compiler_common + omp + base_tpl
        ],

        "toss_4_x86_64_ib": [
            "clang@13.0.0+rocm" + compiler_common + omp + toss4_spec("gfx906") + base_tpl
        ],

        "toss_4_x86_64_ib_cray": [
            "clang@13.0.0+rocm" + compiler_common + omp + toss4_spec("gfx908") + base_tpl
        ],

        "blueos_3_ppc64le_ib_p9": [
            "clang@9.0.0" + compiler_common + omp + base_tpl,
            "clang@8.0.1" + compiler_common + omp + cuda + base_tpl,
            "gcc@7.3.1~cpp14+devtools~mfem+c2c" + base_tpl,
            "xl@16.1.1.1" + compiler_common + omp + cuda + base_tpl,
            "xl@16.1.1.2" + compiler_common + omp + base_tpl
        ],

        "darwin-x86_64": [
            "clang@9.0.0~cpp14+devtools+mfem" + base_tpl
        ]
    }

    return spec_lists

if __name__ == "__main__":
    s = specs()
    print(s)
    print(s["toss_3_x86_64_ib"][0])
