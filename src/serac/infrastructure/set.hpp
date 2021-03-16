#pragma once

namespace serac {

template <typename S>
std::vector<S> Union(std::vector<S> v1, std::vector<S> v2)
{
  std::vector<S> v(v1.size() + v2.size());
  auto           it = std::set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
  v.resize(static_cast<typename std::vector<S>::size_type>(it - v.begin()));
  return v;
}

template <typename S>
std::vector<S> Intersection(std::vector<S> v1, std::vector<S> v2)
{
  std::vector<S> v(v1.size() + v2.size());
  auto           it = std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
  v.resize(static_cast<typename std::vector<S>::size_type>(it - v.begin()));
  return v;
}

/**
 * @brief Calculate the difference between v1 and v2
 *
 * @param[in] v1 A vector of indices in v1
 * @param[in] v2 A vector of indices in v2
 */
template <typename S>
std::vector<S> Difference(std::vector<S> v1, std::vector<S> v2)
{
  std::vector<S> v(v1.size() + v2.size());
  auto           it = std::set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
  v.resize(static_cast<typename std::vector<S>::size_type>(it - v.begin()));
  return v;
}

template <class T>
class Set {
public:
  using index_type = typename std::vector<T>::size_type;

  /**
   * @brief Create an empty set
   */
  Set() : total_size(0) {}

  /**
   * @brief Create a set containing all the indices up to size tagged by the attr T
   *
   * @param[in] attr A key to identify this set
   * @param[in] size The number of indices in this set
   */
  Set(T attr, int size = 0) : total_size(size)
  {
    std::vector<T> one_set(size);
    std::iota(std::begin(one_set), std::end(one_set), 0);
    keys.push_back(attr);
    values_index_list[attr] = one_set;
  }

  // convert an attribute list into a Set
  Set(std::vector<T> attr_list) : total_size(attr_list.size())
  {
    for (index_type i = 0; i < attr_list.size(); i++) {
      values_index_list[attr_list[i]].push_back(i);
    }

    for (auto kv : values_index_list) {
      keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
  }

  /// Converts an initializer_list to a set
  ///
  Set(std::initializer_list<T> l) : total_size(l.size())
  {
    std::vector<T> attr_list(l);
    for (index_type i = 0; i < attr_list.size(); i++) {
      values_index_list[attr_list[i]].push_back(i);
    }

    for (auto [kv, _] : values_index_list) {
      keys.push_back(kv);
    }
    std::sort(keys.begin(), keys.end());
  }

  Set(std::unordered_map<T, std::vector<index_type>> m) : total_size(0)
  {
    // set the map accordingly
    for (auto [k, v] : m) {
      keys.push_back(k);
      values_index_list[k] = v;
      total_size += v.size();
    }
  }

  /// move constructor
  Set(Set<T>&& s) : total_size(s.values_size())
  {
    std::swap(keys, s.keys);
    std::swap(values_index_list, s.values_index_list);
  }

  /// the number of keys
  typename std::vector<T>::size_type keys_size() { return values_index_list.size(); }

  /// the total size of all the indices in the set
  typename std::vector<T>::size_type values_size() { return total_size; }

  /**
   * @brief Get all the values that have these keys
   *
   * if the initialize list is empty, just return all values
   *
   * @param[in] t initializer list of keys
   */
  std::vector<index_type> values(std::initializer_list<T> t = {})
  {
    std::vector<T> combine_keys(t);

    // if t is empty, return all keys
    if (combine_keys.size() == 0) {
      combine_keys = keys;
    }
    std::vector<index_type> combined_values;
    for (auto k : combine_keys) {
      combined_values = Union(values_index_list[k], combined_values);
    }
    return combined_values;
  }

  /**
   * @brief Compute the union of this set and s2
   *
   * The keys = the union of the keys
   * The values are the union of the values of the same keys
   * Since it is possible for a value_index to have two corresponding keys
   * and it is preferrable to not have overlapping keys, the value of the overlapping
   * value_index will be the "larger" of the 2 keys.
   *
   * @param[in] s2 the second set
   */
  Set<T> getUnion(Set<T>& s2)
  {
    // find the union of all the keys in the sets
    Set<T> sunion_overlap;
    sunion_overlap.keys = Union(keys, s2.keys);
    // go through each of the keys and combine them
    for (auto k : sunion_overlap.keys) {
      sunion_overlap.values_index_list[k] = Union(values_index_list[k], s2.values_index_list[k]);
      sunion_overlap.total_size += sunion_overlap.values_index_list[k].size();
    }
    // Get rid of overlapping entries
    return Set<T>(sunion_overlap.toList());
  }

  /**
   * @brief Get the intersection of the keys. union on values
   *
   * @param[in] s2 the second set
   */
  Set<T> getIntersection(Set<T>& s2)
  {
    // find the intersection of all the keys in the sets
    Set<T> sall;
    sall.keys = Intersection(keys, s2.keys);
    // go through each of the keys and combine them
    for (auto k : sall.keys) {
      sall.values_index_list[k] = Intersection(values_index_list[k], s2.values_index_list[k]);
      sall.total_size += sall.values_index_list[k].size();
    }
    return sall;
  }

  /// Get the difference of the values while retaining original set values
  Set<T> getDifference(Set<T>& s2)
  {
    // for each key in this set diff with each key in s2
    Set<T> diff;
    for (auto s1_key : keys) {
      //	  auto s1_intersect_s2 = Intersection(values_index_list[s1_key], s2.values_index_list[s2_key]);
      auto s1_diff = Difference(values_index_list[s1_key], s2.values());
      if (diff.values_index_list.find(s1_key) == diff.values_index_list.end()) {
        // key doesn't exist yet
        diff.keys.push_back(s1_key);
        diff.values_index_list[s1_key] = s1_diff;
      } else {
        // append to previous values
        diff.values_index_list[s1_key] = Union(diff.values_index_list[s1_key], s1_diff);
      }
    }

    // Sort the keys and sort each of the values for each key
    std::sort(diff.keys.begin(), diff.keys.end());
    for (auto& [key, list] : diff.values_index_list) {
      std::sort(list.begin(), list.end());
      diff.total_size += list.size();
    }

    return diff;
  }

  /// Get the complement of a given subset of keys
  Set<T> getComplement(std::vector<T> subkeys)
  {
    // find the difference of the keys
    Set<T> sall;
    sall.keys = Difference(keys, subkeys);
    // copy over only the vectors that are part of the commplement
    for (auto k : sall.keys) {
      sall.values_index_list[k] = values_index_list[k];
      sall.total_size += sall.values_index_list[k].size();
    }
    return sall;
  }

  /// Get the subset of a given subset of keys
  Set<T> getSubset(std::vector<T> subkeys)
  {
    // find the difference of the keys
    Set<T> sall;
    sall.keys = subkeys;
    // copy over only the vectors that are part of the subset
    for (auto k : sall.keys) {
      sall.values_index_list[k] = values_index_list[k];
      sall.total_size += sall.values_index_list[k].size();
    }
    return sall;
  }

  /// Allows us to print sets to stream
  friend std::ostream& operator<<(std::ostream& os, const Set<T>& set)
  {
    for (auto [k, v] : set.values_index_list) {
      os << k << " : ";
      for (auto i : v) {
        os << i << " ";
      }
      os << std::endl;
    }
    return os;
  }

  // converts a Set<T> to a std::vector<T>
  std::vector<T> toList()
  {
    std::vector<T> attr_list(values_size());
    for (auto k : keys) {
      for (auto i : values_index_list[k]) {
        attr_list[i] = k;
      }
    }
    return attr_list;
  }

protected:
  std::unordered_map<T, std::vector<std::size_t>> values_index_list;
  std::vector<T>                                  keys;
  typename std::vector<T>::size_type              total_size;
};

}  // namespace serac
