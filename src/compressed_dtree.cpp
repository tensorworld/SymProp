#include <detail/compressed_dtree.hpp>
#include <detail/dense/symtensor_array.hpp>
#include <detail/dense/symtensor_base.hpp>
#include <omp.h>
#include <utils/function_timer.hpp>
#include <utils/threading.hpp>

#include <queue>
#include <stack>

namespace symprop {

CompressedDTree::CompressedDTree(const DTree &dt, const order_t depth)
    : depth(depth), level_size(depth), level_idx(depth), level_ptr(depth),
      dep_list(depth) {
  START_FUNCTION_TIMER();
  std::queue<DTreeNode *> q;
  DTreeNode *t = dt.root_node();
  q.push(t);

  std::vector<index_t> ptr;
  std::vector<dim_t> idx;
  std::vector<DTreeNode *> children;

  for (size_t level = 0; level < depth; level++) {
    size_t k = q.size();
    if (level)
      ptr.push_back(0);
    for (size_t i = 0; i < k; i++) {
      t = q.front();
      q.pop();

      t->for_each_child([&]([[maybe_unused]] dim_t _, DTreeNode *child) {
        children.push_back(child);
        // q.push(child);
        // idx.push_back(child_idx);
      });

      std::sort(children.begin(), children.end(),
                [](DTreeNode *a, DTreeNode *b) { return a->value < b->value; });

      for (auto child : children) {
        q.push(child);
        idx.push_back(child->value);
      }
      children.clear();

      if (level) {
        ptr.push_back(ptr.back() + t->nchildren);
      }
    }

    level_size[level] = idx.size();
    level_idx[level] = idx;

    if (level) {
      assert(idx.size() == ptr.back());
      level_ptr[level - 1] = ptr;
      ptr.clear();
    }
    idx.clear();
  }

  // remaining elements in q are the leaf nodes
  level_ptr[depth - 1].resize(level_size[depth - 1] + 1, 0);
  assert(level_size[depth - 1] == q.size());
  for (size_t i = 0; i < level_size[depth - 1]; i++) {
    t = q.front();
    q.pop();
    level_ptr[depth - 1][i + 1] = level_ptr[depth - 1][i] + t->owners.size();
    for (auto &nz : t->owners) {
      nonzero.push_back({nz.rem_index, nz.value});
    }
  }
  init_dep_list();

  order_t ms_level = 0;
  index_t max_size = 0;
  for (order_t level = 0; level < depth; level++) {
    if (level_size[level] * (level + 1) > max_size) {
      max_size = level_size[level] * (level + 1);
      ms_level = level;
    }
  }

  p_parent_indices =
      std::make_unique<Buffer2D<dim_t>>(level_size[ms_level], ms_level + 1);
  p_child_indices =
      std::make_unique<Buffer2D<dim_t>>(level_size[ms_level], ms_level + 1);
};

void CompressedDTree::init_dep_list() {
  START_FUNCTION_TIMER();
  std::vector<dim_t> parent_indices;
  std::vector<dim_t> child_indices;

  parent_indices.reserve(level_size[0]);
  for (index_t i = 0; i < level_size[0]; i++) {
    parent_indices.push_back(level_idx[0][i]);
  }

  for (size_t level = 1; level < depth; level++) {
    child_indices.reserve(level_size[level] * (level + 1));
    dep_list[level].reserve(level_size[level] * (level + 1));

    for (index_t i = 0; i < level_size[level - 1]; i++) {
      auto parent_index =
          std::span<const dim_t>(parent_indices).subspan(i * level, level);
      index_t child_start = level_ptr[level - 1][i];
      index_t child_end = level_ptr[level - 1][i + 1];

      for (index_t j = child_start; j < child_end; j++) {
        dim_t idx_val = level_idx[level][j];

        child_indices.insert(child_indices.end(), parent_index.begin(),
                             parent_index.end());
        child_indices.push_back(idx_val);
        auto child_index = std::span<const dim_t>(child_indices)
                               .subspan(j * (level + 1), level + 1);

        for (index_t k = 0; k < child_index.size(); k++) {
          index_t parent_idx = exclude_search(child_index, k);
          dep_list[level].push_back(parent_idx);
        }
      }
    }

    parent_indices.swap(child_indices);
    child_indices.clear();
  }
}

struct compnode {
  std::vector<dim_t> subindex;
  std::unique_ptr<SymtensorBase> kptensor;

  compnode(std::vector<dim_t> subindex,
           std::unique_ptr<SymtensorBase> &&kptensor)
      : subindex(std::move(subindex)), kptensor(std::move(kptensor)) {}
};

void CompressedDTree::kronecker_product(const std::vector<real_t> &U,
                                        size_t R) {
  START_FUNCTION_TIMER();
  std::vector<compnode> parents;
  std::vector<compnode> children;

  auto Urow = [&](size_t i) { return std::span(U).subspan((i - 1) * R, R); };

  parents.push_back(compnode(std::vector<dim_t>(), nullptr));
  size_t level = 0;
  while (level < depth) {
    children.reserve(level_size[level]);

    if (level == 0) {
      for (size_t i = 0; i < level_size[level]; i++) {
        dim_t idx_val = level_idx[level][i];
        children.push_back(
            compnode(std::vector<dim_t>{idx_val},
                     std::make_unique<Symtensor<1>>(Urow(idx_val))));
      }
    } else {
      for (index_t i = 0; i < level_size[level - 1]; i++) {
        auto &parent = parents[i];
        std::vector<dim_t> &indices = parent.subindex;
        index_t child_start = level_ptr[level - 1][i];
        index_t child_end = level_ptr[level - 1][i + 1];

        for (size_t j = child_start; j < child_end; j++) {
          dim_t idx_val = level_idx[level][j];
          indices.push_back(idx_val);
          auto res = create_symtensor(R, level + 1);

          for (size_t k = 0; k < indices.size(); k++) {
            index_t parent_idx = exclude_search(indices, k);
            res->sym_outer_prod(*parents[parent_idx].kptensor,
                                Urow(indices[k]));
          }
          children.push_back(compnode(indices, std::move(res)));
          indices.pop_back();
        }
      }
    }
    parents.swap(children);
    children.clear();
    level += 1;
  }

  leaf_kptensors.clear();
  leaf_kptensors.reserve(parents.size());
  for (auto &parent : parents) {
    leaf_kptensors.push_back(std::move(parent.kptensor));
  }
}

void CompressedDTree::kronecker_product_opt1(const std::vector<real_t> &U,
                                             size_t R) {
  START_FUNCTION_TIMER();
  std::vector<compnode> parents;
  std::vector<compnode> children;

  auto Urow = [&](size_t i) { return std::span(U).subspan((i - 1) * R, R); };

  parents.push_back(compnode(std::vector<dim_t>(), nullptr));
  size_t level = 0;
  while (level < depth) {
    children.reserve(level_size[level]);
    if (level == 0) {
      for (size_t i = 0; i < level_size[level]; i++) {
        dim_t idx_val = level_idx[level][i];
        children.push_back(
            compnode(std::vector<dim_t>{idx_val},
                     std::make_unique<Symtensor<1>>(Urow(idx_val))));
      }
    } else {
      for (index_t i = 0; i < level_size[level - 1]; i++) {
        auto &parent = parents[i];
        std::vector<dim_t> &indices = parent.subindex;
        index_t child_start = level_ptr[level - 1][i];
        index_t child_end = level_ptr[level - 1][i + 1];

        for (size_t j = child_start; j < child_end; j++) {
          dim_t idx_val = level_idx[level][j];
          indices.push_back(idx_val);
          auto res = create_symtensor(R, level + 1);

          for (size_t k = 0; k < indices.size(); k++) {
            index_t parent_idx = dep_list[level][j * (level + 1) + k];
            res->sym_outer_prod(*parents[parent_idx].kptensor,
                                Urow(indices[k]));
          }
          children.push_back(compnode(indices, std::move(res)));
          indices.pop_back();
        }
      }
    }
    parents.swap(children);
    children.clear();
    level += 1;
  }

  leaf_kptensors.clear();
  leaf_kptensors.reserve(parents.size());
  for (auto &parent : parents) {
    leaf_kptensors.push_back(std::move(parent.kptensor));
  }
}

void CompressedDTree::kronecker_product_opt2(const std::vector<real_t> &U,
                                             size_t R) {
  START_FUNCTION_TIMER();
  std::vector<std::unique_ptr<SymtensorBase>> parents;
  std::vector<std::unique_ptr<SymtensorBase>> children;

  auto Urow = [&](size_t i) { return std::span(U).subspan((i - 1) * R, R); };
  std::vector<dim_t> parent_indices;
  std::vector<dim_t> child_indices;

  parents.push_back(nullptr);
  size_t level = 0;
  while (level < depth) {
    children.reserve(level_size[level]);
    child_indices.reserve(level_size[level] * (level + 1));

    if (level == 0) {
      for (size_t i = 0; i < level_size[level]; i++) {
        dim_t idx_val = level_idx[level][i];
        children.push_back(std::make_unique<Symtensor<1>>(Urow(idx_val)));
        child_indices.push_back(idx_val);
      }
    } else {
      for (index_t i = 0; i < level_size[level - 1]; i++) {
        auto parent_index =
            std::span<const dim_t>(parent_indices).subspan(i * level, level);
        index_t child_start = level_ptr[level - 1][i];
        index_t child_end = level_ptr[level - 1][i + 1];

        for (size_t j = child_start; j < child_end; j++) {
          dim_t idx_val = level_idx[level][j];
          child_indices.insert(child_indices.end(), parent_index.begin(),
                               parent_index.end());
          child_indices.push_back(idx_val);
          auto child_index = std::span<const dim_t>(child_indices)
                                 .subspan(j * (level + 1), level + 1);

          auto res = create_symtensor(R, level + 1);
          for (size_t k = 0; k < child_index.size(); k++) {
            index_t parent_idx = dep_list[level][j * (level + 1) + k];
            res->sym_outer_prod(*parents[parent_idx], Urow(child_index[k]));
          }

          children.push_back(std::move(res));
        }
      }
    }

    parents.swap(children);
    parent_indices.swap(child_indices);
    children.clear();
    child_indices.clear();
    level += 1;
  }

  leaf_kptensors.clear();
  leaf_kptensors.reserve(parents.size());
  for (auto &parent : parents) {
    leaf_kptensors.push_back(std::move(parent));
  }
}

void CompressedDTree::kronecker_product_opt3(const std::vector<real_t> &U,
                                             size_t R) {
  START_FUNCTION_TIMER();
  SymtensorArray parents(R, 1, level_size[0]);
  auto Urow = [&](size_t i) { return std::span(U).subspan((i - 1) * R, R); };

  std::vector<dim_t> parent_indices;
  std::vector<dim_t> child_indices;

  parent_indices.reserve(level_size[0]);
  for (size_t i = 0; i < level_size[0]; i++) {
    dim_t idx_val = level_idx[0][i];
    parents.set(i, Urow(idx_val));
    parent_indices.push_back(idx_val);
  }

  for (order_t level = 1; level < depth; level++) {
    FunctionTimer timer("level " + std::to_string(level));

    SymtensorArray children(R, level + 1, level_size[level]);
    child_indices.reserve(level_size[level] * (level + 1));

    for (index_t i = 0; i < level_size[level - 1]; i++) {
      auto parent_index =
          std::span<const dim_t>(parent_indices).subspan(i * level, level);
      index_t child_start = level_ptr[level - 1][i];
      index_t child_end = level_ptr[level - 1][i + 1];

      for (size_t j = child_start; j < child_end; j++) {
        dim_t idx_val = level_idx[level][j];
        child_indices.insert(child_indices.end(), parent_index.begin(),
                             parent_index.end());
        child_indices.push_back(idx_val);
        auto child_index = std::span<const dim_t>(child_indices)
                               .subspan(j * (level + 1), level + 1);

        for (size_t k = 0; k < child_index.size(); k++) {
          index_t parent_idx = dep_list[level][j * (level + 1) + k];
          children.sym_outer_prod(j, parents, parent_idx, Urow(child_index[k]));
        }
      }
    }

    parents = std::move(children);
    parent_indices.swap(child_indices);
    child_indices.clear();
  }

  leaf_kptensors_array = std::make_unique<SymtensorArray>(std::move(parents));
}

void CompressedDTree::kronecker_product_opt4(const std::vector<real_t> &U,
                                             size_t R) {
  START_FUNCTION_TIMER();
  SymtensorArray parents(R, 1, level_size[0]);
  auto Urow = [&](size_t i) { return std::span(U).subspan((i - 1) * R, R); };

  order_t ms_level = 0;
  index_t max_size = 0;
  for (order_t level = 0; level < depth; level++) {
    if (level_size[level] * (level + 1) > max_size) {
      max_size = level_size[level] * (level + 1);
      ms_level = level;
    }
  }

  Buffer2D<dim_t> parent_indices(level_size[ms_level], ms_level + 1);
  Buffer2D<dim_t> child_indices(level_size[ms_level], ms_level + 1);

  parent_indices.reshape(level_size[0], 1);
  for (size_t i = 0; i < level_size[0]; i++) {
    dim_t idx_val = level_idx[0][i];
    parents.set(i, Urow(idx_val));
    parent_indices(i, 0) = idx_val;
  }

  for (order_t level = 1; level < depth; level++) {
    FunctionTimer timer("level " + std::to_string(level));
    SymtensorArray children(R, level + 1, level_size[level]);
    child_indices.reshape(level_size[level], level + 1);

#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < level_size[level - 1]; i++) {
      auto parent_index = parent_indices[i];

      index_t child_start = level_ptr[level - 1][i];
      index_t child_end = level_ptr[level - 1][i + 1];

      for (size_t j = child_start; j < child_end; j++) {
        dim_t idx_val = level_idx[level][j];
        std::span<dim_t> child_index = child_indices[j];

        std::copy(parent_index.begin(), parent_index.end(),
                  child_index.begin());
        child_index.back() = idx_val;

        for (size_t k = 0; k < child_index.size(); k++) {
          index_t parent_idx = dep_list[level][j * (level + 1) + k];
          children.sym_outer_prod(j, parents, parent_idx, Urow(child_index[k]));
        }
      }
    }

    parents = std::move(children);
    parent_indices.swap(child_indices);
  }

  leaf_kptensors_array = std::make_unique<SymtensorArray>(std::move(parents));
}

void CompressedDTree::kronecker_product_opt5(const std::vector<real_t> &U,
                                             size_t R, std::vector<real_t> &Y,
                                             size_t ldy) {
  START_FUNCTION_TIMER();
  SymtensorArray parents(R, 1, level_size[0], false);
  auto Urow = [&](size_t i) { return std::span(U).subspan((i - 1) * R, R); };
  auto Yrow = [&](size_t i) {
    return std::span(Y).subspan((i - 1) * ldy, ldy);
  };

  auto &parent_indices = *p_parent_indices;
  auto &child_indices = *p_child_indices;

  parent_indices.reshape(level_size[0], 1);
  for (size_t i = 0; i < level_size[0]; i++) {
    dim_t idx_val = level_idx[0][i];
    parents.set(i, Urow(idx_val));
    parent_indices(i, 0) = idx_val;
  }

  std::vector<std::unique_ptr<SymtensorBase>> t_row(threading::nthreads());
  std::vector<omp_lock_t> locks(Y.size() / ldy);

  for (size_t i = 0; i < locks.size(); i++) {
    omp_init_lock(&locks[i]);
  }

#pragma omp parallel
  { t_row[omp_get_thread_num()] = create_symtensor(R, depth); }

  for (order_t level = 1; level < depth; level++) {
    FunctionTimer timer("level " + std::to_string(level));
    SymtensorArray children(R, level + 1, level_size[level], false);
    child_indices.reshape(level_size[level], level + 1);
    std::vector<index_t> freqs(depth + 1);
    std::vector<dim_t> unnz_indices(depth + 1);

#pragma omp parallel for schedule(dynamic, 32) firstprivate(freqs)                \
    firstprivate(unnz_indices)
    for (index_t i = 0; i < level_size[level - 1]; i++) {
      auto parent_index = parent_indices[i];

      index_t child_start = level_ptr[level - 1][i];
      index_t child_end = level_ptr[level - 1][i + 1];

      for (size_t j = child_start; j < child_end; j++) {
        dim_t idx_val = level_idx[level][j];
        std::span<dim_t> child_index = child_indices[j];

        std::copy(parent_index.begin(), parent_index.end(),
                  child_index.begin());
        child_index.back() = idx_val;

        if (level != depth - 1) {
          children.zero_init(j);

          for (size_t k = 0; k < child_index.size(); k++) {
            index_t parent_idx = dep_list[level][j * (level + 1) + k];
            children.sym_outer_prod(j, parents, parent_idx,
                                    Urow(child_index[k]));
          }
        } else {
          auto &t_row_ = *t_row[omp_get_thread_num()];
          t_row_.set_zero();
          auto t_row_span = std::span<real_t>(t_row_);

          for (size_t k = 0; k < child_index.size(); k++) {
            index_t parent_idx = dep_list[level][j * (level + 1) + k];
            sym_outer_prod(t_row_span, parents.symtensor_data(parent_idx),
                           Urow(child_index[k]), level + 1);
          }

          for (size_t inz = level_ptr[level][j]; inz < level_ptr[level][j + 1];
               inz++) {
            dim_t rem_index = nonzero[inz].first;
            real_t nnz_val = nonzero[inz].second;
            std::copy(child_index.begin(), child_index.end(),
                      unnz_indices.begin());
            unnz_indices.back() = rem_index;
            std::sort(unnz_indices.begin(), unnz_indices.end());
            count_freq(std::span(unnz_indices), std::span(freqs));
            real_t coeff = nnz_val * multinomial_coeff(std::span(freqs)) /
                           static_cast<real_t>(factorial[depth + 1]);

            omp_set_lock(&locks[rem_index - 1]);
            for (size_t l = 0; l < t_row_.packed_size(); l++) {
              Yrow(rem_index)[l] += t_row_span[l] * coeff;
            }
            omp_unset_lock(&locks[rem_index - 1]);
          }
        }
      }
    }

    parents = std::move(children);
    parent_indices.swap(child_indices);
  }
}

index_t CompressedDTree::exclude_search(std::span<const dim_t> indices,
                                        order_t exclude_index) {
  auto excluded_view =
      std::views::iota(0u, indices.size()) |
      std::views::filter(
          [exclude_index](size_t idx) { return idx != exclude_index; }) |
      std::views::transform([&indices](size_t idx) { return indices[idx]; });
  return search(excluded_view);
}

void CompressedDTree::print() const {
  std::cout << "compressed dtree depth " << depth << "\n";
  // size_t indent = 0;
  // std::stack<std::pair<dim_t, index_t>> st;
  for (size_t i = 0; i < depth; i++) {
    if (i != depth - 1) {
      std::cout << "level " << i << " size " << level_size[i] << "\n";
      for (size_t j = 0; j < level_size[i]; j++) {
        std::cout << "node " << level_idx[i][j] << " ptr " << level_ptr[i][j]
                  << " | ";
      }
      std::cout << " ptr " << level_ptr[i][level_size[i]] << "\n";
    }
    if (i == depth - 1) {
      for (size_t j = 0; j < level_size[i]; j++) {
        std::cout << "node " << level_idx[i][j] << " ptr " << level_ptr[i][j]
                  << " | ";
        for (size_t k = level_ptr[i][j]; k < level_ptr[i][j + 1]; k++) {
          std::cout << nonzero[k].first << " " << nonzero[k].second << " ";
        }
        std::cout << "\n";
      }
    }
  }

  std::cout << "\n";
}

} // namespace symprop
