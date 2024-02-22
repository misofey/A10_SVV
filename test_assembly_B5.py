import unittest
#import pytest          #also possible to use
import numpy as np
from objects.assembly import Assembly           #import the code you want to test
from telescope import telescope_geometry
from main import MechRes2D as sim

# ------- link: https://docs.python.org/3/library/unittest.html#assert-methods -------

class TestBlock5(unittest.TestCase):
    def setUp(self):
        # Initialize necessary objects and variables
        self.assembly = Assembly(telescope_geometry('telescope.toml'))
        self.inp = telescope_geometry('telescope.toml')
        self.obj = TestBlock5()

# integration test: tests if values of T are the same in two different modules
    def test_test(self):
        # Set up test inputs
        a = self.assembly.input['T']
        b = self.inp['T']

        # Assert expected outcomes or relations
        for i in range(min(len(a),len(b))):
            self.assertEqual(a[i],b[i], "functions input files not the same")
            # more assertions possible
    #def test_matrix_U(self):
        #Set up test inputs
         #u_calc = np.shape(self.assembly.output['U'])

         #if np.all(self.inp['type'] == 0):
            #u_true = (self.inp['points']*3,1)
         #else:
            #u_true = ((self.inp['points']+np.sum(self.inp['seeds']),1)
         #self.assertEqual(u_calc,u_true,"matrix U not the same")

    def test_matrix_P(self):

        a = self.assembly.input['P']
        a = len(a)
        #for n in range(10):
        self.assertEqual(a, 27, "functions input files not the same")

if __name__ == '__main__':
    unittest.main()
