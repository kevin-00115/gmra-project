// SYSTEM INCLUDES
#include <iostream>
#include <math.h>
#include <list>


// C++ PROJECT INCLUDES
#include "trees/dyadictree.h"





DyadicTree::DyadicTree(std::shared_ptr<CoverTree> tree):
    _root(nullptr),
    _max_scale(tree->get_max_scale()),
    _min_scale(tree->get_min_scale()),
    _num_nodes(0),
    _num_levels(0)
{
    this->_root = std::shared_ptr<DyadicCell>(new DyadicCell(tree->get_root()->get_subtree_idxs(this->_max_scale, this->_min_scale)));
    this->_num_nodes++;
    this->_num_levels++;

    std::list<CoverNodePtr> nodes = {tree->get_root()};
    std::list<DyadicCellPtr> cells = {this->_root};

    for(int64_t scale = this->_max_scale; scale > this->_min_scale; --scale)
    {
        std::list<CoverNodePtr> new_nodes;
        std::list<DyadicCellPtr> new_cells;

        auto node_it = nodes.begin();
        auto cell_it = cells.begin();

        while(node_it != nodes.end())
        {
            CoverNodePtr node = *node_it;
            DyadicCellPtr cell = *cell_it;

            for(auto child_node: node->get_children(scale, false))
            {
                new_nodes.push_back(child_node);
                DyadicCellPtr child_cell(new DyadicCell(child_node->get_subtree_idxs(scale-1, this->_min_scale)));

                cell->_children.push_back(child_cell);
                new_cells.push_back(child_cell);
                this->_num_nodes++;
            }

            node_it++;
            cell_it++;
        }

        nodes = new_nodes;
        cells = new_cells;
        this->_num_levels++;
    }
    // std::cout << this->_num_levels << " " << this->_max_scale << " "
    //           << this->_min_scale << std::endl;
    TORCH_CHECK(this->_num_levels == (this->_max_scale - this->_min_scale + 1),
                "ERROR num_levels != max_scale - min_scale");

    /**
    auto root_children = tree->get_root().get_children(tree->get_max_scale(), false);
    std::vector<int64_t> root_idxs(root_children.size(), -1);
    for(size_t i = 0; i < root_children.size(); ++i)
    {
        root_idxs[i] = root_children[i]->pt_idx();
    }

    std::list<CoverNodePtr> nodes_at_level = {tree->get_root()};
    std::list<DyadicCellPtr> cells_at_level = {this->_root};

    for(int64_t scale = this->_max_scale-1; scale >= this->_min_scale; --scale)
    {
        
    }
    **/
}


bool
DyadicTree::validate()
{
    std::list<DyadicCellPtr> cells = {this->_root};
    torch::Tensor root_idxs = this->_root->_idxs;

    for(int64_t level = 0; level < this->_num_levels - 1; ++level)
    {
        std::list<DyadicCellPtr> next_cells;
        std::list<torch::Tensor> all_idxs;

        for(auto cell: cells)
        {
            for(auto child_cell: cell->_children)
            {
                next_cells.push_back(child_cell);
                all_idxs.push_back(child_cell->_idxs);
            }
        }

        torch::Tensor idx_union = torch::cat(std::vector<torch::Tensor>(all_idxs.begin(),
                                                                        all_idxs.end()),
                                               0).view({-1});

        torch::Tensor idx_unique = std::get<0>(torch::_unique2(idx_union));

        // idxs should all be disjoint
        if(idx_unique.size(0) != idx_union.size(0))
        {
            std::cout << "WARNING: dyadic cells are not disjoint at level "
                      << level << std::endl;
            return false;
        }

        // idxs should union to root_idxs
        if(idx_unique.size(0) != root_idxs.size(0))
        {
            std::cout << "WARNING: dyadic cells do not cover root at level "
                      << level << std::endl;
            return false;
        }

        cells = next_cells;
    }

    /**
    int64_t level = 1;
    while(cells.size() > 0)
    {
        std::list<DyadicCellPtr> next_cells;
        std::list<torch::Tensor> all_idxs;

        for(auto cell: cells)
        {
            for(auto child_cell: cell->_children)
            {
                next_cells.push_back(child_cell);
                all_idxs.push_back(child_cell->_idxs);
            }
        }

        torch::Tensor idx_union = torch::cat(std::vector<torch::Tensor>(all_idxs.begin(),
                                                                        all_idxs.end()),
                                               0).view({-1});

        torch::Tensor idx_unique = std::get<0>(torch::_unique2(idx_union));

        // idxs should all be disjoint
        if(idx_unique.size(0) != idx_union.size(0))
        {
            std::cout << "WARNING: dyadic cells are not disjoint at level "
                      << level << std::endl;
            return false;
        }

        // idxs should union to root_idxs
        if(idx_unique.size(0) != root_idxs.size(0))
        {
            std::cout << "WARNING: dyadic cells do not cover root at level "
                      << level << std::endl;
            return false;
        }

        cells = next_cells;
        level++;
    }
    **/

    return true;
}

std::vector<torch::Tensor>
DyadicTree::get_idxs_at_level(int64_t level)
{

    std::list<DyadicCellPtr> cells = {this->_root};
    while(level > 0)
    {
        std::list<DyadicCellPtr> child_cells;
        for(auto cell: cells)
        {
            for(auto child_cell: cell->_children)
            {
                child_cells.push_back(child_cell);
            }
        }
        cells = child_cells;
        level--;
    }

    std::vector<torch::Tensor> idxs;
    for(auto cell: cells)
    {
        idxs.push_back(cell->_idxs);
    }
    return idxs;
}

