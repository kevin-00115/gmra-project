// SYSTEM INCLUDES
#include <fstream>
#include <sstream>
#include <ostream>
#include <iostream>
#include <math.h>
#include <list>
#include <tuple>


// C++ PROJECT INCLUDES
#include "trees/covertree.h"


// using ALL = torch::indexing::Slice;


//////////////////////////////////////////
// taken from https://stackoverflow.com/questions/9358718/similar-function-in-c-to-pythons-strip
#include <string>
#include <cctype>

std::string strip(const std::string &inpt)
{
    auto start_it = inpt.begin();
    auto end_it = inpt.rbegin();
    while (std::isspace(*start_it))
        ++start_it;
    while (std::isspace(*end_it))
        ++end_it;
    return std::string(start_it, end_it.base());
}
//////////////////////////////////////////


CoverNodePtr
lookup(std::list<CoverNodePtr> l, int64_t idx)
{
    for(auto it = l.begin(); it != l.end(); ++it)
    {
        if(idx == 0)
        {
            return *it;
        }
        idx--;
    }
    return nullptr;
}


CoverNode::~CoverNode()
{
    this->_children.clear();
}


void
CoverNode::add_child(CoverNodePtr child,
                     int64_t scale)
{
    auto it = this->_children.find(scale);
    if(it != this->_children.end())
    {
       it->second.insert(child);
    } else
    {
        std::unordered_set<CoverNodePtr> new_children = {child};
        this->_children.insert({scale, new_children});
    }
    child->_parent = this->shared_from_this();
}

std::list<CoverNodePtr>
CoverNode::get_children(int64_t scale,
                        bool only_children)
{
    //std::cout << "CoverNode::get_children: enter" << std::endl;
    std::list<CoverNodePtr> children;
    if(!only_children)
    {
        children.push_back(this->shared_from_this());
    }
    auto it = this->_children.find(scale);
    if(it != this->_children.end())
    {
        //std::cout << "CoverNode::get_children: node: " << this->_pt_idx << "has "
        //      << it->second.size() << "children!" << std::endl;
        for(auto child: it->second)
        {
            children.push_back(child);
        }
    }

    //std::cout << "CoverNode::get_children: only_children: " << only_children
    //          << " len(children): " << children.size() << std::endl;
    //std::cout << "CoverNode::get_children: exit" << std::endl;
    return children;
}

torch::Tensor
CoverNode::get_subtree_idxs(int64_t max_scale,
                            int64_t min_scale)
{
    std::list<CoverNodePtr> nodes_at_current_scale = {this->shared_from_this()};
    std::unordered_set<int64_t> pt_idxs;

    for(int64_t scale = max_scale; scale >= min_scale; --scale)
    {
        std::list<CoverNodePtr> children;
        for(auto node: nodes_at_current_scale)
        {
            for(auto c: node->get_children(scale, false))
            {
                children.push_back(c);
                pt_idxs.insert(node->_pt_idx);
            }
        }

        nodes_at_current_scale = children;
    }

    return torch::tensor(std::vector<int64_t>(pt_idxs.begin(), pt_idxs.end()));
}

bool
CoverNode::equals(CoverNodePtr other)
{
    bool is_same = this->_pt_idx == other->_pt_idx && this->_is_root == other->_is_root;

    if(!this->_is_root && !other->_is_root)
    {
        is_same = is_same && this->_parent.lock()->_pt_idx ==
                             other->_parent.lock()->_pt_idx;
    }

    return is_same;
}


CoverTree::~CoverTree()
{
}


void
CoverTree::insert(torch::Tensor X)
{
    for(int64_t pt_idx = 0; pt_idx < X.size(0); ++pt_idx)
    {
        this->insert_pt(pt_idx, X);
    }
}

void
CoverTree::insert_pt(int64_t pt_idx,
                     torch::Tensor X)
{
    //std::cout << "CoverTree::insert_pt: inserting pt_idx: " << pt_idx << std::endl;
    if(!this->_root)
    {
        this->_root = std::make_shared<CoverNode>(pt_idx);
        this->_nodes.push_back(this->_root);
        this->_root->_is_root = true;
        this->_num_nodes++;
        //std::cout << "CoverTree::insert_pt: done" << std::endl;
        return;
    }

    const torch::Tensor& pt = X.index({pt_idx}); //, ALL()});
    // const torch::Tensor& pt = X[torch::tensor({pt_idx})];

    Q_TYPE Qi_p_ds = {{this->_root},
                      this->compute_distances(pt,
                        X.index({this->_root->_pt_idx}))}; //, ALL()}))};

    int64_t scale = this->_max_scale;
    CoverNodePtr parent = nullptr;
    int64_t pt_scale = -1;

    bool stop = false;
    while(!stop)
    {
        float radius = std::pow(this->_base, scale);

        Q_TYPE Q_p_ds = this->get_children_and_distances(pt, X, Qi_p_ds, scale);
        float min_dist = this->get_min_dist(Q_p_ds);

        if(min_dist == 0)
        {
            return;
        } else if(min_dist > radius)
        {
            stop = true;
        } else
        {
            if(this->get_min_dist(Qi_p_ds) <= radius)
            {
                torch::Tensor parent_indices = (std::get<1>(Qi_p_ds) <= radius).nonzero();
                int64_t choice_idx = torch::randint(parent_indices.size(0), {1})
                    .item<int64_t>();
                parent = lookup(std::get<0>(Qi_p_ds),
                                parent_indices[choice_idx].item<int64_t>());
                pt_scale = scale;
            }

            {
                torch::Tensor new_Qi_p_ds_mask = std::get<1>(Q_p_ds) <= radius;
                std::list<CoverNodePtr> new_Qi_p_ds;
                int64_t idx = 0;
                for(auto it = std::get<0>(Q_p_ds).begin(); it != std::get<0>(Q_p_ds).end();
                    ++it)
                {
                    if(new_Qi_p_ds_mask[idx].item<bool>())
                    {
                        new_Qi_p_ds.push_back(*it);
                    }
                    idx++;
                }
                Qi_p_ds = std::make_tuple(new_Qi_p_ds,
                                         std::get<1>(Q_p_ds).index({new_Qi_p_ds_mask}));
                // Qi_p_ds = std::make_tuple(new_Qi_p_ds,
                //                           std::get<1>(Q_p_ds)[new_Qi_p_ds_mask]);
            }
            scale -= 1;
        }
    }

    auto new_node = std::make_shared<CoverNode>(pt_idx);
    this->_nodes.push_back(new_node);
    parent->add_child(new_node, pt_scale);
    this->_num_nodes++;
    this->_min_scale = std::min(this->_min_scale, pt_scale-1);

    //std::cout << "CoverTree::insert_pt: done" << std::endl;
}


std::string
get_idxs(const std::unordered_set<CoverNodePtr>& Q_i)
{
    std::ostringstream out;
    out << "{";
    for(auto node : Q_i)
    {
        out << node->pt_idx() << ", ";
    }
    out << "}";
    return out.str();
}


bool
CoverTree::validate(torch::Tensor X)
{
    bool success = true;
    std::unordered_set<CoverNodePtr> Q_i = {this->_root};
    for(int64_t scale = this->_max_scale; scale >= this->_min_scale; --scale)
    {
        float radius = std::pow(this->_base, scale);

        std::unordered_set<CoverNodePtr> Q_i_minus_1;
        for(auto node : Q_i)
        {
            for(auto child : node->get_children(scale, false))
            {
                Q_i_minus_1.insert(child);
            }
        }

        // std::cout << "DEBUG CoverTree::validate scale: " << scale << std::endl
        //           << "Q_i: " << get_idxs(Q_i) << std::endl
        //           << "Q_{i-1}" << get_idxs(Q_i_minus_1) << std::endl;

        // check nesting
        bool nesting = true;
        for(auto node : Q_i)
        {
            bool found_node = Q_i_minus_1.find(node) != Q_i_minus_1.end();
            if(!found_node)
            {
                std::cout << "WARNING CoverTree::validate scale: " << scale
                          << " fails nesting for node: " << node->_pt_idx << std::endl;
            }
            nesting = nesting && found_node;
        }

        // check covering
        bool covering = true;
        for(auto child : Q_i_minus_1)
        {
            bool found_parent = false;
            for(auto parent : Q_i)
            {
                bool dist_requirement =
                    this->compute_distances(X.index({parent->_pt_idx}).view({1,-1}),
                                            X.index({child->_pt_idx}).view({1,-1}))
                    .item<float>() < radius;

                auto children_list = parent->get_children(scale, false);
                std::unordered_set<CoverNodePtr> children(children_list.begin(),
                                                          children_list.end());
                bool is_parent = children.find(child) != children.end();
                found_parent = found_parent || (dist_requirement && is_parent);
            }

            if(!found_parent)
            {
                std::cout << "WARNING CoverTree::validate scale: " << scale
                          << " could not find a parent for node: " << child->_pt_idx
                          << std::endl;
            }
            covering = covering && found_parent;
        }

        // check separation
        bool separation = true;
        for(auto node1 : Q_i)
        {
            for(auto node2 : Q_i)
            {
                if(node1 != node2)
                {
                    bool dist_requirement =
                        (this->compute_distances(X.index({node1->_pt_idx}).view({1,-1}),
                                                 X.index({node2->_pt_idx}).view({1,-1}))
                         .item<float>() > radius);
                    if(!dist_requirement)
                    {
                        std::cout << "WARNING CoverTree::validate scale: " << scale
                                  << " nodes [" << node1->_pt_idx << ", "
                                  << node2->_pt_idx << "] are not close enough"
                                  << std::endl;
                    }
                    separation = separation && dist_requirement;
                }
            }
        }


        success = success && nesting && covering && separation;
        Q_i = Q_i_minus_1;
    }

    return success;
}

torch::Tensor
CoverTree::parent_vector()
{
    std::vector<int64_t> parent_vec(this->_num_nodes, -1);

    for(auto node : this->_nodes)
    {
        if(!node->_is_root)
        {
            parent_vec[node->_pt_idx] = node->_parent.lock()->_pt_idx;
        }
    }
    return torch::tensor(parent_vec);
}

void
CoverTree::save(std::string path)
{
    std::ofstream file;
    file.open(path);

    file << "{" << std::endl;

    file << "\tmin_scale:\t"  << this->_min_scale     << "," << std::endl;
    file << "\tmax_scale:\t"  << this->_max_scale     << "," << std::endl;
    file << "\tbase:\t"       << this->_base          << "," << std::endl;
    file << "\tnum_nodes:\t"  << this->_num_nodes     << "," << std::endl;
    file << "\troot:\t"       << this->_root->_pt_idx << "," << std::endl;
    file << "\tnodes:\t [";

    int64_t i = 0;
    for(auto it = this->_nodes.begin(); it != this->_nodes.end(); ++it)
    {
        file << (*it)->_pt_idx;
        if(i < this->_num_nodes - 1)
        {
            file << ", ";
        }
        i++;
    }
    file << "]," << std::endl;

    // now write adjacency lists
    file << "\tadjacency:\t{" << std::endl;
    i = 0;
    for(auto it = this->_nodes.begin(); it != this->_nodes.end(); ++it)
    {
        auto nodeptr = (*it);
        file << "\t\t" << nodeptr->_pt_idx << ":\t{" << std::endl;
        file << "\t\t\tis_root:\t" << nodeptr->_is_root << "," << std::endl;
        if(!nodeptr->_is_root)
        {
            file << "\t\t\tparent:\t" << nodeptr->_parent.lock()->_pt_idx
                 << "," <<  std::endl;
        } else
        {
            file << "\t\t\tparent:\t" << -1 << "," << std::endl;
        }

        file << "\t\t\tscale_map: {" << std::endl;
        int64_t j = 0;
        for(auto map_it = nodeptr->_children.begin();
            map_it != nodeptr->_children.end();
            ++map_it)
        {
            auto scale = map_it->first;
            auto children = map_it->second;

            file << "\t\t\t\t" << scale << ":\t[";
            int64_t k = 0;
            for(auto child_it = children.begin(); child_it != children.end(); ++child_it)
            {
                file << (*child_it)->_pt_idx;
                if(k < (int64_t)(children.size()) - 1)
                {
                    file << ", ";
                }
                k++;
            }

            file << "]";
            if(j < (int64_t)(nodeptr->_children.size()) - 1)
            {
                file << ",";
            }
            file << std::endl;
            j++;
        }
        file << "\t\t\t}" << std::endl;

        file << "\t\t}";
        if(i < this->_num_nodes - 1)
        {
            file << ",";
        }
        i++;
        file << std::endl;
    }
    file << "\t}" << std::endl;

    file << "}" << std::endl;

    file.close();
}

bool
CoverTree::equals(std::shared_ptr<CoverTree> other)
{
    bool is_same = (this->_max_scale == other->_max_scale)
                   && (this->_min_scale == other->_min_scale)
                   && (this->_num_nodes == other->_num_nodes)
                   && (this->_base == other->_base)
                   && this->_root->equals(other->_root);

    std::map<int64_t, CoverNodePtr> this_node_map;
    for(auto it = this->_nodes.begin(); it != this->_nodes.end(); ++it)
    {
        this_node_map.insert({(*it)->_pt_idx, *it});
    }

    for(auto it = other->_nodes.begin(); it != other->_nodes.end(); ++it)
    {
        if(this_node_map.find((*it)->_pt_idx) != this_node_map.end())
        {
            auto other_node = (*it);
            auto this_node = this_node_map[other_node->_pt_idx];

            is_same = is_same && this_node->equals(other_node);
        } else
        {
            is_same = false;
        }
    }

    return is_same;
}

Q_TYPE
CoverTree::get_children_and_distances(torch::Tensor pt,
                                      torch::Tensor X,
                                      const Q_TYPE& Qi_p_ds,
                                      int64_t scale)
{
    //std::cout << "CoverTree::get_children_and_distances: entering" << std::endl;

    std::list<CoverNodePtr> Q;
    std::list<int64_t> Q_idxs;
    for(auto node: std::get<0>(Qi_p_ds))
    {
        //std::cout << "CoverTree::get_children_and_distances: node->pt_idx: "
        //          << node->_pt_idx << std::endl;
        for(auto child : node->get_children(scale, true))
        {
            //std::cout << "CoverTree::get_children_and_distances: \tchild->pt_idx: "
            //          << child->_pt_idx << std::endl;
            Q.push_back(child);
            Q_idxs.push_back(child->_pt_idx);
        }
    }

    //std::cout << "CoverTree::get_children_and_distances: Q_idxs: {";
    //for(auto i : Q_idxs)
    //{
    //    std::cout << i << ",";
    //}
    //std::cout << "}" << std::endl;

    if(Q_idxs.size() > 0)
    {
        torch::Tensor X_pts = X.index({torch::tensor(std::vector<int64_t>(Q_idxs.begin(),
                                                                          Q_idxs.end()))});
                                             //,
                                             //ALL()});
        // torch::Tensor X_pts = X[torch::tensor(std::vector<int64_t>(Q_idxs.begin(),
        //                                                            Q_idxs.end()))];

        // now join them together
        // std::list<CoverNodePtr> joined;
        for(auto x: std::get<0>(Qi_p_ds))
        {
            Q.push_back(x);
        }

        torch::Tensor Q_dists = this->compute_distances(pt, X_pts).view({-1});
        torch::Tensor Qi_dists = std::get<1>(Qi_p_ds);

        //std::cout << "CoverTree::get_children_and_distances: Q_dists.sizes(): "
        //          << Q_dists.sizes() << std::endl;
        //std::cout << "CoverTree::get_children_and_distances: Qi_dists.sizes(): "
        //          << Qi_dists.sizes() << std::endl;
        torch::Tensor dists = torch::cat({Q_dists, Qi_dists}, 0).view({-1});
        //std::cout << "CoverTree::get_children_and_distances: exit" << std::endl;
        return std::make_tuple(Q, dists);
    }

    return Qi_p_ds;
}


torch::Tensor
CoverTree::compute_distances(torch::Tensor pt,
                             torch::Tensor X)
{
    //std::cout << "CoverTree::compute_distances: enter" << std::endl;
    //std::cout << "CoverTree::compute_distances: pt.sizes(): " << pt.sizes() << std::endl;
    //std::cout << "CoverTree::compute_distances: X.sizes(): " << X.sizes() << std::endl;
    torch::Tensor ds = torch::sqrt(torch::pow(torch::abs(X - pt.view({1,-1})), 2).sum(1));

    //std::cout << "CoverTree::compute_distances: dists: " << ds.view({1,-1}) << std::endl;
    //std::cout << "CoverTree::compute_distances: exit" << std::endl;
    return ds;
}

float
CoverTree::get_min_dist(const Q_TYPE& Qi_p_ds)
{
    const auto distances = std::get<1>(Qi_p_ds);
    torch::Tensor min_dist = torch::min(distances);
    return min_dist.cpu().item<float>();
}


std::tuple<std::string, std::string>
split_pair(const std::string& line)
{
    auto colon_position = line.find(":");
    TORCH_CHECK(colon_position != std::string::npos,
                "ERROR ':' not in string");

    auto key = strip(line.substr(0, colon_position));
    auto value = strip(line.substr(colon_position+1, std::string::npos));
    if(value.at(value.size()-1) == ',')
    {
        value = value.substr(0, value.size()-1);
    }

    return std::make_tuple(key, value);
}


std::list<int64_t>
parse_idx_list(const std::string& list_str)
{
    std::list<int64_t> list;

    TORCH_CHECK(list_str.at(0) == '[' &&
                list_str.at(list_str.size()-1) == ']',
                "ERROR list_str not parsed correctly");

    // remove '[' and ']'
    std::istringstream node_stream(list_str.substr(1, list_str.size()-1));
    std::string element;
    while(std::getline(node_stream, element, ','))
    {
        int64_t node = std::stoll(strip(element));
        list.push_back(node);
    }

    return list;
}


std::tuple<int64_t, int64_t, float, int64_t, int64_t, std::list<int64_t> >
parse_metadata(const std::string& path)
{
    int64_t max_scale = -1;
    int64_t min_scale = -1;
    float base = -1.0;
    int64_t num_nodes = -1;
    int64_t root = -1;
    std::list<int64_t> nodes;

    std::ifstream file(path);
    TORCH_CHECK(file.is_open(), "ERROR file could not be opened");
    std::string line;
    while(std::getline(file, line))
    {
        line = strip(line);
        // std::cout << "line [" << line << "]" << std::endl;
        if(line.rfind("min_scale", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "min_scale",
                        "ERROR min_scale found but not parsed correctly");
            min_scale = std::stoll(std::get<1>(pair));
        } else if(line.rfind("max_scale", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "max_scale",
                        "ERROR max_scale found but not parsed correctly");
            max_scale = std::stoll(std::get<1>(pair));
        } else if(line.rfind("base", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "base",
                        "ERROR base found but not parsed correctly");
            base = std::stof(std::get<1>(pair));
        } else if(line.rfind("num_nodes", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "num_nodes",
                        "ERROR num_nodes found but not parsed correctly");
            num_nodes = std::stoll(std::get<1>(pair));
        } else if(line.rfind("root", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "root",
                        "ERROR root found but not parsed correctly");
            root = std::stoll(std::get<1>(pair));
        } else if(line.rfind("nodes", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "nodes",
                        "ERROR nodes found but not parsed correctly");
            nodes = parse_idx_list(std::get<1>(pair));
        }
    }

    /**
    std::cout << "PARSE_METADATA: max_scale: " << max_scale << std::endl;
    std::cout << "PARSE_METADATA: min_scale: " << min_scale << std::endl;
    std::cout << "PARSE_METADATA: base: " << base << std::endl;
    std::cout << "PARSE_METADATA: num_nodes: " << num_nodes << std::endl;
    std::cout << "PARSE_METADATA: root: " << root << std::endl;
    std::cout << "PARSE_METADATA: nodes: {";
    for(auto it = nodes.begin(); it != nodes.end(); ++it)
    {
        std::cout << (*it) << ", ";
    }
    std::cout << "}" << std::endl;
    */

    return std::make_tuple(min_scale, max_scale, base, num_nodes, root, nodes);
}


std::tuple<bool, int64_t, std::map<int64_t, std::list<int64_t> > >
parse_node(std::ifstream& file)
{
    std::string line;
    bool is_root = false;
    int64_t parent = -1;
    std::map<int64_t, std::list<int64_t> > children;

    bool found_root_attr = false;
    bool found_parent_attr = false;
    bool found_children_attr = false;

    bool stop = false;
    while(std::getline(file, line) && !stop)
    {
        line = strip(line);
        // std::cout << "line [" << line << "]" << std::endl;
        if(line.rfind("is_root", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "is_root",
                        "ERROR is_root found but not parsed correctly");


            /**
            std::cout << "is_root str val: [" << std::get<1>(pair) << "]" << std::endl;
            std::cout << (std::get<1>(pair).size() != 1)
                      << (std::get<1>(pair)[0] < '0')
                      << (std::get<1>(pair)[0] > '1') << std::endl;
            */
            TORCH_CHECK(std::get<1>(pair).size() == 1
                        && std::get<1>(pair)[0] >= '0'
                        && std::get<1>(pair)[0] <= '1',
                        "ERROR invalid boolean input for is_root");
            is_root = ( std::get<1>(pair)[0] == '1' );
            found_root_attr = true;
        } else if(line.rfind("parent", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "parent",
                        "ERROR parent found but not parsed correctly");
            parent = std::stoll(std::get<1>(pair));
            found_parent_attr = true;
        } else if(line.rfind("scale_map", 0) != std::string::npos)
        {
            auto pair = split_pair(line);
            TORCH_CHECK(std::get<0>(pair) == "scale_map",
                        "ERROR scale_map found but not parsed correctly");

            // need to parse a list of elements, each on a new line
            std::string child_line;
            while(std::getline(file, child_line) &&
                  strip(child_line).find(":") != std::string::npos)
            {
                auto scale_pair = split_pair(child_line);
                int64_t scale = std::stoll(std::get<0>(scale_pair));
                std::list<int64_t> nodes = parse_idx_list(std::get<1>(scale_pair));

                children.insert({scale, nodes});
            }
            found_children_attr = true;
        }

        stop = found_root_attr && found_parent_attr && found_children_attr;
    }

    return std::make_tuple(is_root, parent, children);
}

void
CoverTree::load_covertree(const std::string& path)
{
    auto metadata = parse_metadata(path);
    this->_max_scale = std::get<1>(metadata);
    this->_base = std::get<2>(metadata);

    // create tree object and assign basic metadata
    this->_min_scale = std::get<0>(metadata);
    int64_t root_pt_idx = std::get<4>(metadata);
    auto node_pt_idxs = std::get<5>(metadata);

    TORCH_CHECK((int64_t)node_pt_idxs.size() == std::get<3>(metadata),
                "ERROR parsed node list != reported number of nodes");
    this->_num_nodes = std::get<3>(metadata);

    // need to recreate nodes, build a map as well as tree->_nodes
    // so we can quickly assign parents (and root) in the future
    std::map<int64_t, std::shared_ptr<CoverNode> > node_map;
    for(auto it = node_pt_idxs.begin(); it != node_pt_idxs.end(); ++it)
    {
        auto new_node = std::make_shared<CoverNode>(*it);
        node_map.insert({*it, new_node});
        this->_nodes.push_back(new_node);
    }

    // check that root exists in map and assign
    TORCH_CHECK(node_map.find(root_pt_idx) != node_map.end(),
                "ERROR reported root node not found in reported nodes");
    this->_root = node_map[root_pt_idx];


    // need to search for the adjacency entry
    std::ifstream file(path);
    TORCH_CHECK(file.is_open(), "ERROR file could not be opened");

    std::string line;
    while(std::getline(file, line) &&
          strip(line).rfind("adjacency", 0) == std::string::npos)
    {
        // std::cout << "line: [" << line << "]" << std::endl;
    }

    // now we're at the adjacency line
    // std::cout << "found adjacency line!" << std::endl;
    // std::cout << "line [" << line << "]" << std::endl;

    while(std::getline(file, line))
    {
        line = strip(line);
        // std::cout << "line [" << line << "]" << std::endl;

        if(line.find(":") != std::string::npos)
        {
            auto node_pair = split_pair(line);
            int64_t node_idx = std::stoll(std::get<0>(node_pair));
            auto node_attrs = parse_node(file);

            bool is_root = std::get<0>(node_attrs);
            int64_t parent = std::get<1>(node_attrs);
            auto node_children = std::get<2>(node_attrs);
            TORCH_CHECK((parent == -1 && node_idx == root_pt_idx) || (parent != -1),
                        "ERROR node has no parent but is not root");

            if(parent != -1)
            {
                node_map[node_idx]->_parent = node_map[parent];
            }
            node_map[node_idx]->_is_root = is_root;
            for(auto map_it = node_children.begin();
                map_it != node_children.end();
                ++map_it)
            {
                auto scale = map_it->first;
                auto scale_children = map_it->second;

                std::unordered_set<CoverNodePtr> children_nodes;
                for(auto child_it = scale_children.begin();
                    child_it != scale_children.end();
                    ++child_it)
                {
                    children_nodes.insert(node_map[*child_it]);
                }

                node_map[node_idx]->_children.insert({scale, children_nodes});
            }
        }
    }
    file.close();
}

