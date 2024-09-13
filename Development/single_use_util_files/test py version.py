import platform
import sys

python_version = platform.python_version()
print("Python Version:", python_version)
print("Location of Python executable:",  sys.executable)
print("list of PYTHONPATHS")
print(sys.path)



