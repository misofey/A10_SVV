import unittest
import numpy

from objects.assembly import Assembly
from objects.element import Element
from objects.mesh import Mesh
from telescope import telescope_geometry

class MyTestCase(unittest.TestCase):
    def test_element(self):

        inp = telescope_geometry('simple_beam.toml')

        assembly = Assembly(inp)
        assembly.mesh = Mesh(assembly)
        assembly.mesh.assign_element_properties()

        for ix in range(assembly.mesh['nElements']):
            # for each element, generate local properties
            element = Element(assembly, ix)
            # rewrite the local mesh with the mesh that now has the element properties included
            assembly.mesh = element.assembly.mesh

    def test_element_properties(self):
        inp = telescope_geometry('simple_beam')
        assembly = Assembly(inp)
        assembly.mesh = Mesh(assembly)
        self.run()

if __name__ == '__main__':
    unittest.main()
