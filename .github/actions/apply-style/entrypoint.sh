#!/bin/bash

# This is a bare minimum of options needed to create the `style` build target
# This does not create a working build.
CMAKE_ARGS=-DCMAKE_CXX_COMPILER=clang++
CMAKE_ARGS="$CMAKE_ARGS -DENABLE_CLANGFORMAT=ON"
CMAKE_ARGS="$CMAKE_ARGS -DCLANGFORMAT_EXECUTABLE=/usr/bin/clang-format"
CMAKE_ARGS="$CMAKE_ARGS -DSERAC_STYLE_CI_ONLY=ON"

# Avoid error "fatal: detected dubious ownership in repository at '/github/workspace'"
REPO_PATH=/github/workspace
git config --global --add safe.directory "$REPO_PATH"
find "$REPO_PATH" -type d | while read -r dir; do
  git config --global --add safe.directory "$dir"
done

git fetch

###
# Attempt to find the branch of the PR from the detached head state
##

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Attempting to find branch that matches commit..."

# Get the current commit SHA
current_commit_sha=$(git rev-parse HEAD)
# List all branches containing the current commit SHA
branches=$(git branch -r --contains $current_commit_sha)

# Loop over the string split by whitespace
branch=""
num_branches_found=0
for _possible_branch in $branches; do
  # Skip items that start with "pull/"
  if [[ $_possible_branch == pull/* ]]; then
    continue
  fi
  if [[ $_possible_branch == origin/* ]]; then
    _possible_branch=$(echo "$_possible_branch" | sed 's/origin\///')
  fi
  echo "Possible Branch: $_possible_branch"
  branch=$_possible_branch
  num_branches_found=$((num_branches_found+1))
done

if [ "$num_branches_found" -ne 1 ]; then
  echo "Error: Unable to find a single branch that matched git sha $current_commit_sha"
  exit 1
fi

echo "Found branch: $branch"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

git checkout $branch

git submodule update --init --recursive

mkdir build && cd build 
cmake $CMAKE_ARGS ..
make style
cd ..

git config user.name "Agent Style"
git config user.email "no-reply@llnl.gov"
if [ -n "$(git status --porcelain)" ]; then
  git commit -am 'Apply style updates'
  git push
fi
