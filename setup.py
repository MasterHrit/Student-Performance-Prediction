from setuptools import find_packages,setup

HYPHEN_E_DOT="-e ."
def get_requirements(filename):
    """get_requirements This function will return the list of packages in filename <requirements.txt>

    Args:
        filename (string): filename <requirements.txt> which contains the packages

    Returns:
        list: Returns the list of packages in filename <requirements.txt>
    """
    with open(filename,"r") as fileobject:
        list_packages=fileobject.readlines()
    stripped_package_list=[package.strip("\n") for package in list_packages]
    # stripped_package_list=[package.replace("\n","") for package in list_packages]
    if (HYPHEN_E_DOT in stripped_package_list):
        stripped_package_list.remove(HYPHEN_E_DOT)
    return stripped_package_list

setup(
    name="mlproject",
    version="0.0.1",
    author="Hritik",
    author_email="agarwal.hritik96@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)