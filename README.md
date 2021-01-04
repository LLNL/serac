Serac
====

[![Build
Status](https://dev.azure.com/llnl-serac/serac/_apis/build/status/LLNL.serac?branchName=develop)](https://dev.azure.com/llnl-serac/serac/_build/latest?definitionId=1&branchName=develop)
[![Documentation Status](https://readthedocs.org/projects/serac/badge/?version=latest)](https://serac.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/LLNL/serac/branch/develop/graph/badge.svg?token=DO4KFMPNM0)](https://codecov.io/gh/LLNL/serac)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](./LICENSE)

Serac is a 3D implicit nonlinear thermal-structural simulation code. Its primary purpose is to investigate multiphysics 
abstraction strategies and implicit finite element-based algorithm development for emerging computing architectures. 
It also serves as a proxy-app for LLNL's DIABLO and ALE3D codes.

Documentation
------

Build, run, and design documentation can be found at [readthedocs](https://serac.readthedocs.io).

Source documentation can be found [here](https://serac.readthedocs.io/en/latest/doxygen/html/index.html).

Contributions
-------------

We welcome all kinds of contributions: new features, bug fixes, and documentation edits.

For more information, see the [contributing guide](./CONTRIBUTING.md).

License
-------

Copyright (c) 2019-2020, Lawrence Livermore National Security, LLC. 
Produced at the Lawrence Livermore National Laboratory.

Copyrights and patents in the Serac project are retained by contributors.
No copyright assignment is required to contribute to Serac.

See [LICENSE](./LICENSE) for details.

Unlimited Open Source - BSD 3-clause Distribution  
`LLNL-CODE-805541`

SPDX usage
-----------

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)

External Packages
-----------------

Serac bundles some of its external dependencies in its repository.  These
packages are covered by various permissive licenses.  A summary listing
follows.  See the license included with each package for full details.


[//]: # (Note: The spaces at the end of each line below add line breaks)

PackageName: BLT  
PackageHomePage: https://github.com/LLNL/blt  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: uberenv  
PackageHomePage: https://github.com/LLNL/uberenv  
PackageLicenseDeclared: BSD-3-Clause  
