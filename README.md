<!-- ##################################################################################################################################################### -->
<!-- ######################################################################## INFO ####################################################################### -->
<!-- ##################################################################################################################################################### -->

<!--
    This file contains the public text that appears on the PyRat GitHub repository.
    It contains a short description and installation details.
-->

<!-- ##################################################################################################################################################### -->
<!-- ###################################################################### CONTENTS ##################################################################### -->
<!-- ##################################################################################################################################################### -->

<img align="left" width="120px" src="pyrat/gui/drawings/pyrat.png" />

# PyRat

This repository contains the software used in the PyRat course at IMT Atlantique.

The course contents is available at this address:<br />https://formations.imt-atlantique.fr/pyrat/.

# Install

Installation of the PyRat software can be done directly using `pip`. Do not clone or download the repository. Instead please follow the following instructions:

1) Open a terminal and navigate (using the `cd` command if on Linux/OSX or `dir` command if on Windows) to the directory where you want to create your PyRat workspace.

2) Install the PyRat software using the following command:<br />`python -m pip install --user git+https://github.com/BastienPasdeloup/PyRat.git`

3) Then, run the following command to create a PyRat workspace:<br />`python -c "import pyrat; pyrat.PyRat.setup_workspace()"`

4) Check your installation by navigating to `pyrat_workspace/programs` and running the following command:<br />`python random_1.py`

<!-- ##################################################################################################################################################### -->
<!-- ##################################################################################################################################################### -->