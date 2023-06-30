#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program is needed for an automatic installation of the PyRat software.
    Please refer to the README.md file for more information.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Standard imports
import setuptools

#####################################################################################################################################################
####################################################################### SCRIPT ######################################################################
#####################################################################################################################################################

# Installation details
setuptools.setup \
(
    name =                          "PyRat",
    version =                       "4.0.0",
    author =                        "Bastien Pasdeloup",
    author_email =                  "bastien.pasdeloup@imt-atlantique.fr",
    description =                   "PyRat softare used in the PyRat course at IMT Atlantique",
    long_description =              open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    url =                           "https://github.com/BastienPasdeloup/PyRat",
    project_urls =                  {"Course" : "https://formations.imt-atlantique.fr/pyrat"},
    license =                       "MIT",
    packages =                      setuptools.find_packages(),
    include_package_data =          True,
    install_requires =              ["pygame", "gym", "colored", "distinctipy", "playsound==1.2.2", "numpy", "scipy"]
)

#####################################################################################################################################################
#####################################################################################################################################################
