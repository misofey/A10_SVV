import math
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class Assembly:
    def __init__(self, input) -> None:
        
        # Read user input dictionary
        self.input = input

        # Transform Young's modulus input in GPa to MPa
        for material in self.input['mat']: material['E'] *= 10**3

        # Initialise output
        self.output = {'name': self.input['name']}

    def plot_input(self, show=True, save=False) -> None:

        # Check the provided input
        # --------------------------------------------------------------------------
        if 'parts' in self.input.keys() is False:
            raise ('Error: self.input.parts has not been specified by user')

        # Display the structure for the user to check it visually
        # --------------------------------------------------------------------------
        # Plot the parts
        for ix in range(len(self.input['parts'][0, :])):
            plt.plot(self.input['points'][0, self.input['parts'][:, ix]-1],
                     self.input['points'][1, self.input['parts'][:, ix]-1], 'k-', marker='o', linewidth=2, markersize=12)

        # Plot the points
        plt.plot(self.input['points'][0, :], self.input['points']
                    [1, :], 'or', linewidth=4, markersize=10)
        
        # Format the axes of the plot
        plt.xlim([min(self.input['points'][0, :])-0.5, max(self.input['points'][0, :])+0.5])
        plt.ylim([min(self.input['points'][1, :])-0.5, max(self.input['points'][1, :])+0.5])

        # Optionally, save and show figure
        if save:
            plt.savefig('mesh_1.png')
        if show:
            plt.show()

    def global_stiffness_matrix(self):

        # Initialise the global stiffness matrix
        # Each node has 3 degrees of freedom: u,v, and theta
        # --------------------------------------------------------------------------
        
        self.mesh['K'] = np.zeros((3 * self.mesh['nNodes'], 3 * self.mesh['nNodes']))
        for ix in range(len(self.input['parts'][0, :])):
            # Assemlbe the global stiffness matrix
            for jx in self.mesh['part'][ix]['elementNumbers']:
                # Bounds
                lb1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber1'] - 1) + 1
                lb2 = 3 * self.mesh['element'][jx - 1]['nodeNumber1']
                ub1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber2'] - 1) + 1
                ub2 = 3 * self.mesh['element'][jx - 1]['nodeNumber2']
                bounds = np.concatenate(
                    (np.arange(lb1, lb2 + 1), np.arange(ub1, ub2 + 1))) - 1
                # Assemble
                self.mesh['K'][np.ix_(bounds, bounds)] = self.mesh['K'][np.ix_(
                    bounds, bounds)] + self.mesh['element'][jx - 1]['K']

    def global_thermal_load_vector(self):

        # Create global thermal load vector
        # --------------------------------------------------------------------------

        self.output['Q'] = np.zeros((np.shape(self.mesh['K'])[0], 1))
        for ix in range(len(self.input['parts'][0, :])):
            # Assemlbe the thermal load vector
            for jx in self.mesh['part'][ix]['elementNumbers']:
                # Bounds
                lb1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber1'] - 1) + 1
                lb2 = 3 * self.mesh['element'][jx - 1]['nodeNumber1']
                ub1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber2'] - 1) + 1
                ub2 = 3 * self.mesh['element'][jx - 1]['nodeNumber2']
                bounds = np.concatenate(
                    (np.arange(lb1, lb2 + 1), np.arange(ub1, ub2 + 1))) - 1
                # Assemble
                self.output['Q'][bounds] = self.output['Q'][bounds] + self.mesh['element'][jx - 1]['Q']

    def global_mass_matrix(self):

        # Initialise the global mass matrix
        # --------------------------------------------------------------------------
        
        self.mesh['m'] = np.zeros((3 * self.mesh['nNodes'], 3 * self.mesh['nNodes']))
        for ix in range(len(self.input['parts'][0, :])):
            # Assemble the global stiffness matrix
            for jx in self.mesh['part'][ix]['elementNumbers']:
                # Bounds
                lb1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber1'] - 1) + 1
                lb2 = 3 * self.mesh['element'][jx - 1]['nodeNumber1']
                ub1 = 3 * (self.mesh['element'][jx - 1]['nodeNumber2'] - 1) + 1
                ub2 = 3 * self.mesh['element'][jx - 1]['nodeNumber2']
                bounds = np.concatenate((np.arange(lb1, lb2 + 1), np.arange(ub1, ub2 + 1))) - 1
                # Assemble
                self.mesh['m'][np.ix_(bounds, bounds)] = \
                    self.mesh['m'][np.ix_(bounds, bounds)] + self.mesh['element'][jx - 1]['m']

    def loads_bc(self):

        # Applying Loads and Boundary Conditions
        # --------------------------------------------------------------------------
        # Assemble the loading vector
        # Applied loads
        self.output['P'] = np.zeros((np.shape(self.mesh['K'])[0], 1))
        self.output['P'][:max((self.input['points']).shape) * 3] = self.input['P']

        # Displacements of entire system
        self.output['U'] = np.zeros((np.shape(self.mesh['K'])[0], 1))

        # Reaction forces of entire system
        self.output['R'] = np.zeros((np.shape(self.mesh['K'])[0], 1))

        # Reduce the amount of degrees of freedom if rod element is selected
        # --------------------------------------------------------------------------
        if np.all(self.input['type'] == 1):
            # Nothing happens
            pass
        else:
            # Find the indices of the remaining DOFs
            remainDF = np.nonzero(matlib.repmat(
                [1, 1, 0], 1, int(max(self.mesh['K'].shape) / 3)))[1]
            remainBC = np.nonzero(matlib.repmat(
                [1, 1, 0], 1, int(max(self.input['bc'].shape) / 3)))[1]
            # Reduce vector with boundary conditions
            self.input['bc'] = self.input['bc'][remainBC]
            # Reduce stiffness and mass matrices
            self.mesh['K'] = self.mesh['K'][np.ix_(remainDF, remainDF)]
            self.mesh['m'] = self.mesh['m'][np.ix_(remainDF, remainDF)]
            # Reduce displacement, load, thermal load and reaction force vectors
            self.output['U'] = self.output['U'][remainDF]
            self.output['P'] = self.output['P'][remainDF]
            self.output['Q'] = self.output['Q'][remainDF]
            self.output['R'] = self.output['R'][remainDF]

        # Remove blocked degrees of freedom
        # --------------------------------------------------------------------------
        # Active degrees of freedom
        # 1 means d.o.f. is restrained, 0 means d.o.f. is free
        activeDF = np.ones((np.shape(self.mesh['K'])[0]))
        # Find the clamped nodes
        inactiveDF = np.nonzero(self.input['bc'])[0]
        # Inactive degrees of freedom
        activeDF[inactiveDF] = 0
        inactiveDF = np.ones((np.shape(self.mesh['K'])[0])) - activeDF
        # Convert to zeros and ones to indices
        activeDF = np.nonzero(activeDF)[0]
        inactiveDF = np.nonzero(inactiveDF)[0]

        # Reduced stiffness matrices
        Kr = self.mesh['K'][np.ix_(activeDF, activeDF)]
        Ksr = self.mesh['K'][np.ix_(inactiveDF, activeDF)]
        # Reduced load vectors
        Pr = self.output['P'][activeDF]
        Ps = self.output['P'][inactiveDF]
        Qr = self.output['Q'][activeDF]
        Qs = self.output['Q'][inactiveDF]
        # Reduced mass matrix
        mr = self.mesh['m'][np.ix_(activeDF, activeDF)]

        # Return mesh as output
        self.output['Kr'] = Kr
        self.output['Ksr'] = Ksr
        self.output['Pr'] = Pr
        self.output['Ps'] = Ps
        self.output['Qr'] = Qr
        self.output['Qs'] = Qs
        self.output['mr'] = mr

        # Active and inactive degrees of freedom
        self.output['activeDF'] = activeDF
        self.output['inactiveDF'] = inactiveDF

        # Store a reference to the mesh in the output dictionary for ease of access
        self.output['mesh'] = self.mesh


    def stress_strain(self):
        pass


    def plot_output(self, color='red', show=True):

        # Plot output
        # TODO: improve plots to ease comparison, etc.
        # --------------------------------------------------------------------------

        if np.all(self.input['type'] == 1):
            # Reshape 3 displacements per node
            locDisp3D = self.output['U'].reshape(3, self.mesh['nNodes'], order='F')
        else:
            # Reshape 2 displacements per node
            locDisp3D = self.output['U'].reshape(2, self.mesh['nNodes'], order='F')

        # 2D locations of the displaced nodes
        locDisp = self.mesh['nodes'] + self.input['plot_scale']*locDisp3D[0:2, :]
        
        # Plot elements
        for ix in range(max(self.input['parts'][0, :].shape)):
            plt.plot(self.input['points'][0, self.input['parts'][:, ix]-1], self.input['points'][1, self.input['parts'][:, ix]-1],
                    '0.8', linewidth=2)

        # Plot original nodes
        plt.plot(self.mesh['nodes'][0, :], self.mesh['nodes'][1, :], 'o', '0.8',
                linewidth=2, markerfacecolor='None', markersize=5)
        
        # Plot displaced nodes
        plt.plot(locDisp[0, :], locDisp[1, :], linewidth=2,
                markerfacecolor='None', markersize=5, color=color)
        
        # Set horizontal axis limits
        x_min = min(min(self.input['points'][0, :]), min(locDisp[0, :]))
        x_max = max(max(self.input['points'][0, :]), max(locDisp[0, :]))
        horizontal_span = x_max - x_min
        plt.xlim([
            x_min - horizontal_span*0.05,
            x_max + horizontal_span*0.05
        ])

        # Set vertical axis limits
        y_min = min(min(self.input['points'][1, :]), min(locDisp[1, :]))
        y_max = max(max(self.input['points'][1, :]), max(locDisp[1, :]))
        vertical_span = y_max - y_min
        plt.ylim([
            y_min - vertical_span*0.05,
            y_max + vertical_span*0.05,
        ])

        # Set axis labels
        plt.xticks(np.linspace(x_min, x_max, 5))
        plt.yticks(np.linspace(y_min, y_max, 5))
        
        # Vertical axis label decimal places
        scaled_displacements_x, scaled_displacements_y = abs(self.input['plot_scale']*locDisp3D[0:2, :])
        defmin_x = min(scaled_displacements_x[scaled_displacements_x > 0]) if not all(np.isclose(scaled_displacements_x, 0)) else 1
        defmin_y = min(scaled_displacements_y[scaled_displacements_y > 0]) if not all(np.isclose(scaled_displacements_y, 0)) else 1
        order_of_magnitude = lambda n: math.floor(math.log(n, 10))
        decimals_required = lambda n: abs(order_of_magnitude(n)) + 2 if order_of_magnitude(n) < 0 else 0
        plt.gca().xaxis.set_major_formatter(
            FormatStrFormatter(f'%.{decimals_required(defmin_x)}f')
        )
        plt.gca().yaxis.set_major_formatter(
            FormatStrFormatter(f'%.{decimals_required(defmin_y)}f')
        )
        
        # Grid
        plt.grid(True)

        # Resize
        plt.tight_layout()


        if show:
            plt.show()

    def save_output(self):
        # TODO: export output for analysis
        pass