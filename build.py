#   -*- coding: utf-8 -*-
from pybuilder.core import use_plugin, init

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
# use_plugin("python.coverage")
use_plugin("python.distutils")
use_plugin("python.sphinx")
use_plugin('python.install_dependencies')
use_plugin("python.pycharm")
use_plugin('pypi:pybuilder_bandit')

name = "lava-dl"
default_task = ["analyze", "publish"]
version = "0.1.1"
summary = "A library of deep learning tools, which consists of lava.lib.dl.slayer and lava.lib.dl.netx for training and deployment of event-based deep neural networks on traditional as well as neuromorphic backends. Lava-DL is part of Lava Framework "
url = "https://lava-nc.org"
license = ["BSD-3-Clause"]


@init
def set_properties(project):
    project.set_property("dir_source_main_python", "src")
    project.set_property("dir_source_unittest_python", "tests/lava")
    project.set_property("dir_source_main_scripts", "scripts")
    project.set_property("dir_docs", "docs")

    project.set_property("sphinx_config_path", "docs")
    project.set_property("sphinx_source_dir", "docs")
    project.set_property("sphinx_output_dir", "docs/_build")
    project.set_property("sphinx_doc_author", "Lava Project")
    project.set_property("sphinx_doc_builder", "html")
    project.set_property("sphinx_project_name", project.name)
    project.set_property("sphinx_project_version", project.version)

    project.depends_on_requirements("requirements.txt")
    project.build_depends_on("sphinx")
    project.plugin_depends_on("sphinx_rtd_theme")
    project.plugin_depends_on("sphinx_tabs")

    project.set_property("verbose", True)

    project.set_property("coverage_break_build", False)
    project.set_property("coverage_exceptions", [""])

    project.set_property("flake8_include_test_sources", True)
    project.set_property("flake8_break_build", True)
    project.set_property("flake8_ignore", "W503,E203")
    project.set_property("flake8_max_line_length", "80")
    project.set_property(
        "flake8_exclude_patterns",
        "target/*,.svn,CVS,.bzr,.hg,.git,__pycache__,.pybuilder/*",
    )

    project.get_property('distutils_commands').append('build')
    project.get_property('distutils_commands').append('sdist')
    project.get_property('distutils_commands').append('bdist_dumb')

    project.set_property('bandit_break_build', True)
    project.set_property('bandit_include_testsources', False)


@init(environments="unit")
def set_properties_unit(project):
    project.set_property("dir_source_main_python", "src")
    project.set_property("dir_source_unittest_python", "tests/lava")
    project.set_property("dir_source_main_scripts", "scripts")
    project.set_property("dir_docs", "docs")

    project.set_property("sphinx_config_path", "docs")
    project.set_property("sphinx_source_dir", "docs")
    project.set_property("sphinx_output_dir", "docs/_build")
    project.set_property("sphinx_doc_author", "Lava Project")
    project.set_property("sphinx_doc_builder", "html")
    project.set_property("sphinx_project_name", project.name)
    project.set_property("sphinx_project_version", project.version)

    project.depends_on_requirements("requirements.txt")
    project.build_depends_on("sphinx")
    project.plugin_depends_on("sphinx_rtd_theme")
    project.plugin_depends_on("sphinx_tabs")

    project.set_property("verbose", True)
    
    project.set_property("coverage_threshold_warn", 0)
    project.set_property("coverage_break_build", False)

    project.set_property("flake8_include_test_sources", True)
    project.set_property("flake8_break_build", True)
    project.set_property("flake8_ignore", "W503,E203")
    project.set_property(
        "flake8_exclude_patterns",
        "target/*,.svn,CVS,.bzr,.hg,.git,__pycache__,.pybuilder/*",
    )

    project.set_property("unittest_module_glob", "test_*")

    project.set_property('bandit_break_build', True)
    project.set_property('bandit_include_testsources', False)
