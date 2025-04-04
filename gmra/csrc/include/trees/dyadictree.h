#pragma once
#ifndef _GMRA_DYADICTREE_H_
#define _GMRA_DYADICTREE_H_


// SYSTEM INCLUDES
#include <map>
#include <memory>
#include <torch/extension.h>
#include <tuple>
#include <unordered_set>
#include <vector>


// C++ PROJECT INCLUDES
#include "trees/covertree.h"


class DyadicCell;
class DyadicTree;


using DyadicCellPtr = std::shared_ptr<DyadicCell>;


class DyadicCell
{
    friend class DyadicTree;

public:
    DyadicCell(torch::Tensor idxs): _idxs(idxs) {}

    virtual
    ~DyadicCell() = default;

    torch::Tensor
    get_idxs() { return this->_idxs; }

    std::vector<DyadicCellPtr>
    get_children() { return std::vector<DyadicCellPtr>(this->_children.begin(),
                                                       this->_children.end()); }

private:
    torch::Tensor               _idxs;
    std::list<DyadicCellPtr>    _children;
};


class DyadicTree
{
public:
    DyadicTree(std::shared_ptr<CoverTree> tree);

    virtual
    ~DyadicTree() = default;

    bool
    validate();

    std::vector<torch::Tensor>
    get_idxs_at_level(int64_t level);

    // void
    // save(std::string path);

    DyadicCellPtr
    get_root() { return this->_root; }

    int64_t
    get_num_nodes() { return this->_num_nodes; }

    int64_t
    get_num_levels() { return this->_num_levels; }

private:
    DyadicCellPtr   _root;
    int64_t         _max_scale;
    int64_t         _min_scale;
    int64_t         _num_nodes;
    int64_t         _num_levels;
};



#endif // end of _GMRA_DYADICTREE_H_

