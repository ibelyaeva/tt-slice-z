__docformat__ = 'restructuredtext'

hard_dependencies = ("numpy", "pandas")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies

# class
import pyten.tenclass
import pyten.method
import pyten.tools
