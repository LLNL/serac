// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/infrastructure/input.hpp"

#include <gtest/gtest.h>
#include "mfem.hpp"

#include <unordered_map>
#include <vector>
#include <iostream>
#include <algorithm>

class SlicErrorException : public std::exception {
};

 template < typename S>
 std::vector<S> Union(std::vector<S> v1, std::vector<S> v2) {
    std::vector<S> v(v1.size() + v2.size());
        auto it = std::set_union(v1.begin(),
        v1.end(),
        v2.begin(),
        v2.end(),
        v.begin());
    v.resize(it-v.begin());     
    return v;
}
 
template <typename S>
std::vector<S> Intersection(std::vector<S> & v1, std::vector<S> & v2) {
    std::vector<S> v(v1.size() + v2.size());
        auto it = std::set_intersection(v1.begin(),
        v1.end(),
        v2.begin(),
        v2.end(),
        v.begin());
    v.resize(it-v.begin());     
    return v;
}
 
template <typename S>
std::vector<S> Difference(std::vector<S> & v1, std::vector<S> & v2) {
    std::vector<S> v(v1.size() + v2.size());
        auto it = std::set_difference(v1.begin(),
        v1.end(),
        v2.begin(),
        v2.end(),
        v.begin());
    v.resize(it-v.begin());   
    return v; 
}
 
template <class T>
class Set {
public:
    using index_type = typename std::vector<T>::size_type;
 
    Set() {}
 
    // convert an attribute list into a Set
    Set(std::vector<T> attr_list) {
        for (index_type i = 0; i < attr_list.size(); i++) {
            values_index_list[attr_list[i]].push_back(i);
        }
 
        for (auto kv : values_index_list) {
            keys.push_back(kv.first);
        }
        std::sort(keys.begin(),keys.end());
    }
 
    Set(std::initializer_list<T> l) {
        std::vector<T> attr_list(l);
                for (index_type i = 0; i < attr_list.size(); i++) {
            values_index_list[attr_list[i]].push_back(i);
        }
 
        for (auto kv : values_index_list) {
            keys.push_back(kv.first);
        }
        std::sort(keys.begin(),keys.end());
    }
 
    Set(std::unordered_map<T, std::vector<index_type>>  m) {
        // set the map accordingly
        for (auto [k, v] : m)
        {
            keys.push_back(k);
            values_index_list[k] = v;
        }
    }
 
    // move constructor
    Set(Set<T> && s) {
        std::swap(keys, s.keys);
        std::swap(values_index_list, s.values_index_list);
    }
 
    std::size_t size() { return values_index_list.size(); }
 
    const auto values(std::initializer_list<T> t) { 
        std::vector<T> combine_keys(t);
        std::vector<index_type> combined_values;
        for (auto k : combine_keys) {
            combined_values = Union(values_index_list[k], combined_values);
        }
        return combined_values;
        }
 
    // calls union on all of the std::vectors with the same value
    Set<T> getUnion(Set<T> & s2) {
        // find the union of all the keys in the sets
        Set<T> sunion;
        sunion.keys = Union(keys, s2.keys);
        // go through each of the keys and combine them
        for (auto k : sunion.keys) {
            sunion.values_index_list[k] = Union(values_index_list[k], s2.values_index_list[k]);
        }
        return sunion; 
    }
 
    // Get the intersection of the keys. union on values
    Set<T> getIntersection(Set<T> & s2) {
        // find the union of all the keys in the sets
        Set<T> sall;
        sall.keys = Intersection(keys, s2.keys);
       // go through each of the keys and combine them
        for (auto k : sall.keys) {
            sall.values_index_list[k] = Intersection(values_index_list[k], s2.values_index_list[k]);
        }
        return sall; 
    }
 
    // Get the intersection of the keys. union on values
    Set<T> getDifference(Set<T> & s2) {
        // find the union of all the keys in the sets
        Set<T> sall;
        sall.keys = Difference(keys, s2.keys); 
       // go through each of the keys and combine them
        for (auto k : sall.keys) {
            // the keys are either in one set or the other
            if (values_index_list.find(k) != values_index_list.end()) {
                sall.values_index_list[k] = values_index_list[k];
            } else {
                sall.values_index_list[k] = s2.values_index_list[k];
            }
        }
        return sall; 
    }
 
    // Get the complement of a given subset of keys
    Set<T> getComplement(std::vector<T> subkeys) {
        // find the difference of the keys
        Set<T> sall;
        sall.keys = Difference(keys, subkeys);
        // copy over only the vectors that are part of the commplement
        for (auto k : sall.keys)
        {
            sall.values_index_list[k] = values_index_list[k];
        }
        return sall;
    }
 
    friend std::ostream& operator<<(std::ostream& os, const Set<T> & set)
    {
            for (auto [k,v] : set.values_index_list) {
        os << k << " : ";
        for (auto i : v) {
            os << i << " ";
        }
        os << std::endl;
    }
    return os;
    }
 
protected:
    std::unordered_map<T, std::vector<std::size_t>> values_index_list;
    std::vector<T> keys;
};

class SetTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    axom::slic::setAbortFunction([]() { throw SlicErrorException{}; });
    axom::slic::setAbortOnError(true);
    axom::slic::setAbortOnWarning(false);
  }

  void SetUp() override
  {
  }

};

namespace serac {


TEST_F(SetTest, set)
{
    Set<int> a1(std::vector<int>{1,2,4,2,2,3,4});
    Set<int> a2({1,2,3});
    std::cout << "values size: " << a1.size() << std::endl;
    auto a3 = a1.getUnion(a2);
    std::cout << "Union of a1 and a2" << std::endl;
    std::cout << a3 << std::endl;
 
    auto a4 = a1.getIntersection(a2);
    std::cout << "Intersection of a1 and a2" << std::endl;
    std::cout << a4 << std::endl;
 
    auto a5 = a1.getDifference(a2);
    std::cout << "Difference of a1 and a2" << std::endl;
    std::cout << a5 << std::endl;
 
    // Just calculate the set operations on a set of attributes
    std::vector<Set<int>::index_type> specific_union = Union(a1.values({2}), a2.values({2,3}));
    Set<int> a6({{2, specific_union}});
    std::cout << "Specific union" << std::endl;
    std::cout << a6 << std::endl;
 
    auto a7 = a1.getComplement({1,3});
    std::cout << "Subset of a1 without (1,3)" << std::endl;
    std::cout << a7 << std::endl;

}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}
