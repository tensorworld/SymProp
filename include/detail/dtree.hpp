#ifndef SYMPROP_DTREE_HPP
#define SYMPROP_DTREE_HPP

#include <detail/dense/symtensor.hpp>
#include <memory>
#include <unordered_map>
#include <utils/types.hpp>

#include <algorithm>
#include <map>
#include <ranges>
#include <span>
#include <vector>

namespace symprop {

struct Nonzero {
  size_t rem_index;
  real_t value;
  bool valid_branch;
  Nonzero() : Nonzero(0, -1.0) {}
  Nonzero(real_t d) : Nonzero(0, d) {}
  Nonzero(size_t ri, real_t d) : rem_index{ri}, value{d}, valid_branch{false} {}
  void print() {
    std::cout << "Nonzero at leaf node with remaining index " << rem_index
              << " and value " << value << std::endl;
  }
};

struct DTreeNode {
  constexpr static dim_t n_inline = 8;
  dim_t ichildren[n_inline];
  DTreeNode *ichildren_ptr[n_inline];
  dim_t nchildren = 0;
  dim_t value = 0;

  std::unordered_map<dim_t, DTreeNode *> children;
  std::vector<Nonzero> owners;
  std::unique_ptr<SymtensorBase> kptensor = nullptr;

  DTreeNode(dim_t index) : value{index} {}

  DTreeNode *find_child(dim_t index) {
    size_t n_i = std::min(nchildren, n_inline);
    for (size_t i = 0; i < n_i; i++) {
      if (ichildren[i] == index) {
        return ichildren_ptr[i];
      }
    }
    auto it = children.find(index);
    if (it != children.end()) {
      return it->second;
    }
    return nullptr;
  }

  void insert_child(dim_t index, DTreeNode *child) {
    if (nchildren < n_inline) {
      ichildren[nchildren] = index;
      ichildren_ptr[nchildren] = child;
      nchildren++;
    } else {
      children[index] = child;
      nchildren++;
    }
  }

  void print(std::ostream &f, size_t depth) {
    for (size_t i = 0; i < depth; i++) {
      f << "  ";
    }
    if (owners.size() == 0) {
      f << "Node " << value << " with " << nchildren << " children\n";
    } else {
      f << "Leaf " << value;
      for (auto &owner : owners) {
        f << " (" << owner.rem_index << ", " << owner.value << ")";
      }
      f << "\n";
    }

    for_each_child([&](dim_t _ [[maybe_unused]], DTreeNode *child) {
      child->print(f, depth + 1);
    });
  }

  template <typename Func> void for_each_child(Func &&f) {
    size_t n_i = std::min(nchildren, n_inline);
    for (size_t i = 0; i < n_i; i++) {
      f(ichildren[i], ichildren_ptr[i]);
    }
    for (auto &[index, child] : children) {
      f(index, child);
    }
  }
};

struct kpnode {
  DTreeNode *n;
  std::vector<dim_t> subindex;
  kpnode(DTreeNode *n_, std::vector<dim_t> s_) : n(n_), subindex(s_) {}
  kpnode(kpnode &&) = default;
  kpnode &operator=(kpnode &&) = default;
  kpnode(const kpnode &) = delete;
  kpnode &operator=(const kpnode &) = delete;
  ~kpnode() = default;
};

class DTree {
  DTreeNode *root;
  // std::vector<std::vector<size_t>> iou;

public:
  DTree() : root(new DTreeNode(0)) {}
  DTree(const DTree &) = delete;
  DTree &operator=(const DTree &) = delete;
  DTree(DTree &&) = delete;
  DTree &operator=(DTree &&) = delete;
  ~DTree();

  void banerjee_insert(std::vector<dim_t> &indices, real_t value);
  void banerjee_insert(std::span<const dim_t> indices, real_t value);
  void kronecker_product(const std::vector<real_t> &U, size_t R);

  DTreeNode *exclude_search(std::span<const dim_t> indices,
                            order_t exclude_index) {
    auto excluded_view =
        std::views::iota(0u, indices.size()) |
        std::views::filter(
            [exclude_index](size_t idx) { return idx != exclude_index; }) |
        std::views::transform([&indices](size_t idx) { return indices[idx]; });

    // for (auto i : excluded_view) {
    //   std::cout << i << " ";
    // }
    // std::cout << std::endl;

    return search(excluded_view);
  }

  template <std::ranges::view View> DTreeNode *search(View view) {
    auto *node = root;
    for (auto i : view) {
      node = node->find_child(i);
      assert(node);
    }
    assert(node);
    return node;
  }

  std::vector<kpnode> leaves();
  void print(std::ostream &f);
  DTreeNode *root_node() const { return root; }
};

}; // namespace symprop

#endif // SYMPROP_DTREE_HPP