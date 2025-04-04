// SYSTEM INCLUDES
// #include <pybind11/pybind11.h>
#include <torch/extension.h>


// C++ PROJECT INCLUDES
#include "trees/covertree.h"
#include "trees/dyadictree.h"


namespace py = pybind11;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

    py::class_<CoverNode, std::shared_ptr<CoverNode> >(m, "CoverNode")
        .def(py::init([](int64_t pt_idx) { return std::make_shared<CoverNode>(pt_idx); }))
        .def("add_child", &CoverNode::add_child)
        .def("get_children", &CoverNode::get_children)
        .def("get_subtree_idxs", &CoverNode::get_subtree_idxs)
        .def_property_readonly("pt_idx", &CoverNode::pt_idx);

    py::class_<CoverTree, std::shared_ptr<CoverTree> >(m, "CoverTree")
        .def(py::init<int64_t, float>(),
             "constructor",
             py::arg("max_scale") = 10,
             py::arg("base") = 2.0)
        .def(py::init<std::string>(),
             "constructor",
             py::arg("path"))
        .def("insert", &CoverTree::insert)
        .def("insert_pt", &CoverTree::insert_pt)
        .def("validate", &CoverTree::validate)
        .def("parent_vector", &CoverTree::parent_vector)
        .def("save", &CoverTree::save)
        .def("__eq__", &CoverTree::equals)
        .def_property_readonly("root", &CoverTree::get_root)
        .def_property_readonly("num_nodes", &CoverTree::get_num_nodes)
        .def_property_readonly("min_scale", &CoverTree::get_min_scale)
        .def_property_readonly("max_scale", &CoverTree::get_max_scale);


    py::class_<DyadicCell, std::shared_ptr<DyadicCell> >(m, "DyadicCell")
        .def(py::init<torch::Tensor>())
        .def_property_readonly("idxs", &DyadicCell::get_idxs)
        .def_property_readonly("children", &DyadicCell::get_children);

    py::class_<DyadicTree, std::shared_ptr<DyadicTree> >(m, "DyadicTree")
        .def(py::init<std::shared_ptr<CoverTree> >())
        .def("validate", &DyadicTree::validate)
        .def("get_idxs_at_level", &DyadicTree::get_idxs_at_level)
        .def_property_readonly("root", &DyadicTree::get_root)
        .def_property_readonly("num_nodes", &DyadicTree::get_num_nodes)
        .def_property_readonly("num_levels", &DyadicTree::get_num_levels);
}
