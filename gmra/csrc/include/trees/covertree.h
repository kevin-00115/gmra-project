#pragma once
#ifndef _GMRA_COVERTREE_H_
#define _GMRA_COVERTREE_H_


// SYSTEM INCLUDES
#include <map>
#include <memory>
#include <torch/extension.h>
#include <tuple>
#include <unordered_set>
#include <vector>


// C++ PROJECT INCLUDES


class CoverTree;
class CoverNode;

using CoverNodePtr = std::shared_ptr<CoverNode>;
using Q_TYPE = std::tuple<std::list<CoverNodePtr>, torch::Tensor>;

class CoverNode : public std::enable_shared_from_this<CoverNode>
{
    friend class CoverTree;

public:
    CoverNode(int64_t pt_idx):
        _pt_idx(pt_idx),
        _children(),
        _is_root(false),
        _parent()
    {}

    virtual
    ~CoverNode();

    void
    add_child(CoverNodePtr child,
              int64_t i);

    std::list<CoverNodePtr>
    get_children(int64_t level,
                 bool only_children = false);

    torch::Tensor
    get_subtree_idxs(int64_t max_scale,
                     int64_t min_scale);

    int64_t
    pt_idx() { return this->_pt_idx; }

    bool
    equals(CoverNodePtr other);

private:
    int64_t                                                 _pt_idx;
    std::map<int64_t, std::unordered_set<CoverNodePtr> >    _children;
    bool                                                    _is_root;
    std::weak_ptr<CoverNode>                                _parent;
};


class CoverTree
{
public:

    CoverTree(int64_t max_scale = 10,
              float base = 2.0):
        _root(nullptr),
        _max_scale(max_scale),
        _min_scale(max_scale),
        _num_nodes(0),
        _base(base),
        _nodes()
    {}

    CoverTree(std::string path):
        _root(nullptr),
        _max_scale(-1),
        _min_scale(-1),
        _num_nodes(-1),
        _base(-1.0),
        _nodes()
    {
        this->load_covertree(path);
    }

    virtual
    ~CoverTree();

    void
    insert(torch::Tensor X);

    void
    insert_pt(int64_t pt_idx,
              torch::Tensor X);

    bool
    validate(torch::Tensor X);

    torch::Tensor
    parent_vector();

    CoverNodePtr
    get_root() { return this->_root; }

    void
    save(std::string path);

    bool
    equals(std::shared_ptr<CoverTree> other);

    int64_t
    get_max_scale() { return this->_max_scale; }

    int64_t
    get_min_scale() { return this->_min_scale; }

    int64_t
    get_num_nodes() { return this->_num_nodes; }

private:

    void
    load_covertree(const std::string& path);

    Q_TYPE
    get_children_and_distances(torch::Tensor pt,
                               torch::Tensor X,
                               const Q_TYPE& Qi_p_ds,
                               int64_t level);

    torch::Tensor
    compute_distances(torch::Tensor pt,
                      torch::Tensor X);

    float
    get_min_dist(const Q_TYPE& Qi_p_ds);

    CoverNodePtr                    _root;
    int64_t                         _max_scale;
    int64_t                         _min_scale;
    int64_t                         _num_nodes;
    float                           _base;
    std::list<CoverNodePtr>         _nodes;
};


// std::shared_ptr<CoverTree>
// load_covertree(std::string path);


#endif // end of _GMRA_COVERTREE_H_

