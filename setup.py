#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This script is required for a pip installation of the PyRat software.
    Please refer to the "README.md" file for more info.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Imports
import setuptools

#####################################################################################################################################################
####################################################################### SCRIPT ######################################################################
#####################################################################################################################################################

# Set package info
setuptools.setup(
    name = "PyRat",
    version = "4.0.0",
    author = "Bastien Pasdeloup",
    author_email = "bastien.pasdeloup@imt-atlantique.fr",
    description = "PyRat softare used in the PyRat course at IMT Atlantique",
    long_description = open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/BastienPasdeloup/PyRat",
    project_urls = {"Bug Tracker" : "https://github.com/BastienPasdeloup/PyRat/issues",
                    "Course" : "https://formations.imt-atlantique.fr/pyrat"}
    license = "MIT",
    packages = ["PyRat"],
    install_requires = ["pygame"])

#####################################################################################################################################################
#####################################################################################################################################################
