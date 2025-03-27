#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <vector>

#include <detail/dtree.hpp>

namespace symprop {

DTree::~DTree() {
  if (root) {
    std::queue <DTreeNode *> q;
    q.push(root);

    while (!q.empty()) {
      DTreeNode *t = q.front();
      q.pop();
      for (auto it = t->children.begin(); it != t->children.end(); it++) {
        if (it->second) {
          q.push(it->second);
        }
      }
      delete t;
    }
  }
}

/**
 * height = nmodes - 1
 * vec = nz.indices
 * node = root
 **/
void banerjee_insert_helper(DTreeNode *node, std::vector<dim_t> &vec,
                            size_t position, size_t depth, Nonzero nz) {
  if (position == vec.size() - 1 && depth == vec.size() - 1) {
    nz.rem_index = vec[position];
    node->owners.push_back(nz);
    return;
  }

  for (size_t i = position; i < vec.size() - 1; i++) {
    DTreeNode *child = node->find_child(vec[position]);
    if (!child) {
      child = new DTreeNode(vec[position]);
      node->insert_child(vec[position], child);
    }
    banerjee_insert_helper(child, vec, i + 1, depth + 1, nz);
  }
}

void DTree::banerjee_insert(std::span<const dim_t> indices, real_t value) {
  static std::vector<dim_t> vec;
  vec.assign(indices.begin(), indices.end());
  DTree::banerjee_insert(vec, value);
}

void DTree::banerjee_insert(std::vector<dim_t> &indices, real_t value) {
  size_t l = indices.size();
  for (int j_ = l - 1; j_ >= 0; j_--) {
    std::swap(indices[j_], indices[l - 1]);
    std::sort(indices.begin(), indices.begin() + indices.size() - 1);

    // last one is the remaining index in the leaf node
    for (size_t i = 0; i < indices.size() - 1; i++) {
      DTreeNode *child = root->find_child(indices[i]);
      if (!child) {
        child = new DTreeNode(indices[i]);
        root->insert_child(indices[i], child);
      }
      banerjee_insert_helper(root, indices, i, 0, Nonzero(value));
    }

    std::swap(indices[j_], indices[l - 1]);
  }
}

void DTree::kronecker_product(const std::vector<real_t> &U, size_t R) {
  std::vector<kpnode> parents;
  std::vector<kpnode> children;

  auto Urow = [&](size_t i) { return std::span(U).subspan((i - 1) * R, R); };
  DTreeNode *parent;

  parents.push_back(kpnode(root, std::vector<dim_t>()));
  while (!parents.empty()) {
    size_t parents_size = parents.size();
    children.reserve(parents_size);
    size_t level = parents[0].subindex.size();
    for (size_t i = 0; i < parents_size; i++) {
      parent = parents[i].n;

      parent->for_each_child([&](size_t child_ind, DTreeNode *child) {
        if (level == 0) {
          child->kptensor = std::make_unique<Symtensor<1>>(Urow(child_ind));
          children.push_back(kpnode(child, std::vector<dim_t>{dim_t(child_ind)}));
        } else {
          auto res = create_symtensor(R, level + 1);
          std::vector<dim_t> indices = parents[i].subindex;
          indices.push_back(child_ind);
          for (size_t i = 0; i < indices.size(); ++i) {
            auto *node = exclude_search(std::span(indices), i);
            // std::cout << "K(";
            // for (auto i : excluded_view)
            //   std::cout << i << ",";
            // std::cout << ") X U(" << indices[i] << ")" << std::endl;

            res->sym_outer_prod(*node->kptensor, Urow(indices[i]));
          }

          // std::cout << "K(";
          // for (auto i : indices)
          //   std::cout << i << ",";
          // std::cout << ") computed"
          //           << "\n";

          child->kptensor = std::move(res);
          children.push_back(kpnode(child, indices));
        }
      });
    }
    parents.swap(children);
    children.clear();
  }
}

// in c++23 we have ranges::to
template <std::ranges::range R> auto to_vector(R &&r) {
  std::vector<std::ranges::range_value_t<R>> v;
  if constexpr (requires { std::ranges::size(r); }) {
    v.reserve(std::ranges::size(r));
  }
  for (auto &&e : r) {
    v.push_back(static_cast<decltype(e) &&>(e));
  }
  return v;
}

std::vector<kpnode> DTree::leaves() {
  std::vector<kpnode> leaves;
  // for (auto &node : iou) {
  //   for (size_t i = 0; i < node.size(); i++) {
  //     auto excluded_view =
  //         std::views::iota(0u, node.size()) |
  //         std::views::filter([i](size_t idx) { return idx != i; }) |
  //         std::views::transform([&node](size_t idx) { return node[idx]; });
      
  //     // for (auto i : excluded_view)
  //     //   std::cout << i << ",";
  //     // std::cout << std::endl;

  //     auto *leaf = search(excluded_view);
  //     leaves.push_back(kpnode(leaf, to_vector(excluded_view)));
  //   }
  // }

  return leaves;
}

void DTree::print(std::ostream &f) { root->print(f, 0); }

} // namespace symprop
