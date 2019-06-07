LITIV Framework external test app
=================================

[![Language](https://img.shields.io/badge/lang-C%2B%2B14-f34b7d.svg)](http://en.cppreference.com/w/cpp/compiler_support)

This standalone app is used to show how the [LITIV framework](http://github.com/plstcharles/litiv) can be added as a dependency to a CMake project. The [CMakeLists.txt](./CMakeLists.txt) herein contains all the code that should be added to the pre-existing solution, and the [cmake](./cmake/) directory contains all the scripts required to find and verify the framework's subdependencies.

If you have already compiled and installed the LITIV framework on your computer, running CMake (or CMake-GUI) with this baseline project should allow you to start using it externally. You might need to specify paths to your install directories as the project is being configured if you did not use the default ones (e.g. `/usr/local`), and if `USER_DEVELOP` is not a predefined environment variable on your machine.

For more information on requirements, see the [framework's page](http://github.com/plstcharles/litiv).


Building
--------

First, you must create a 'build' directory somewhere (ideally, in this directory), start cmake from there (via command-line or gui), and point to the source directory (where this README is located). For example, in your terminal, type:
```
git clone https://github.com/plstcharles/litiv-ext-test
cd litiv-ext-test
mkdir build
cd build
cmake ../
```
Using the CMake GUI is highly encouraged, as you will have direct access to more options when configuring the framework. It is also easier to set paths to missing/optional 3rd party libraries using the interface if those are not found automatically by CMake.


Citation
--------
If you use a module from the LITIV framework in your own work, please cite its related LITIV publication(s) as acknowledgment. See [this page](http://www.polymtl.ca/litiv/pub/index.php) for a full list.

