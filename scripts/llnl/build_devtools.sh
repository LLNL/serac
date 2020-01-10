#!/bin/bash

# Copyright (c) 2019, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set -e

PROJECTDIR="$(pwd)"
DEVTOOLSDIR=/usr/WS2/smithdev/devtools


# Move previous directory
if [ -d "$DEVTOOLSDIR" ] then
    # Find directory that doesn't exist
    for in {1..100}
    do
        if [ i == 100]
            echo "Too many moved Devtools directories in $DEVTOOLSDIR. Clean up."
            exit 1
        fi
        if [ ! -d "$DEVTOOLSDIR$i" ]
            mv -R $DEVTOOLSDIR $DEVTOOLS$i
            break
        fi
    done
fi



