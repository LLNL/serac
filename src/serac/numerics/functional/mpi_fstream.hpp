#pragma once

#include <fstream>
#include <mpi.h>

namespace mpi {

    struct ofstream : public std::ofstream {
        void initialize() {
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            open("mpi_output_" + std::to_string(rank) + "_" + std::to_string(size) + ".txt");
        }

        // note: don't call this before MPI_Init()
        template<typename T> 
        friend ofstream& operator<<(ofstream&, T);
    };

    template<typename T>
    ofstream& operator<<(ofstream& out, T op) {
        if (!out) out.initialize();
        static_cast<std::ofstream&>(out) << op;
        return out;
    }

    extern ofstream out;

}
