import math
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from objects.mesh import Mesh


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

        txt_pos_y=20
        txt_pos_x = 50
        # Plot the points
        plt.plot(self.input['points'][0, :], self.input['points']
                    [1, :], 'or', linewidth=4, markersize=10)
        for node in range(0,int(np.ceil(len(self.input['points'][0,:])/2))):
            plt.annotate(f"{node+1}", xy=(self.input['points'][0,node], self.input['points']
                    [1,node]), xytext=(self.input['points'][0,node]+txt_pos_x,self.input['points']
                    [1,node]+txt_pos_y),size=15,ha='center', va='bottom')

        for node in range(int(np.ceil(len(self.input['points'][0,:])/2)),len(self.input['points'][0,:])):
            plt.annotate(f"{node+1}", xy=(self.input['points'][0,node], self.input['points']
                    [1,node]), xytext=(self.input['points'][0,node]-txt_pos_x,self.input['points']
                    [1,node]+txt_pos_y),size=15,ha='center', va='bottom')

        range_x = max(self.input['points'][0, :])-min(self.input['points'][0, :])
        range_y = max(self.input['points'][1, :]) - min(self.input['points'][1, :])

        # Format the axes of the plot
        plt.xlim([min(self.input['points'][0, :])-0.1*range_x, max(self.input['points'][0, :])+0.1*range_x])
        plt.ylim([min(self.input['points'][1, :])-0.1*range_y, max(self.input['points'][1, :])+0.1*range_y])

        plt.minorticks_on()
        plt.grid(True, which="major", linestyle="-",alpha=0.2)
        plt.grid(True, which="minor",linestyle=":",alpha=0.7)

        plt.hlines(0, min(self.input['points'][0, :]) - 0.1 * range_x,
                   max(self.input['points'][0, :]) + 0.1 * range_x, "k", zorder=-100)
        plt.fill_between([min(self.input['points'][0, :]) - 0.1 * range_x,
                          max(self.input['points'][0, :]) + 0.1 * range_x],
                         min(self.input['points'][1, :]) - 0.1 * range_y, 0, color='gray', alpha=0.5)

        # Optionally, save and show figure.
        if save:
            plt.savefig('mesh_1.png')
        if show:
            plt.show()

    def initialize_mesh(self):
        self.mesh = Mesh(self)

    def global_stiffness_matrix(self):

        # Initialise the global stiffness matrix
        # Each node has 3 degrees of freedom: u,v, and theta
        # --------------------------------------------------------------------------
        
        self.mesh['K'] = np.zeros((3 * self.mesh['nNodes'], 3 * self.mesh['nNodes']))
        for ix in range(len(self.input['parts'][0, :])):
            # Assemble the global stiffness matrix
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

    def analysis(self):

        # Displacements and reaction forces
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        KrInv = np.linalg.inv(self.output['Kr'])
        Ur = KrInv @ (self.output['Pr'] - self.output['Qr'])
        # Displacements of entire system
        self.output['U'][self.output['activeDF']] = Ur
        # Reaction forces
        Rs = self.output['Ksr'] @ Ur - (self.output['Ps'] - self.output['Qs'])
        # Reaction forces of entire system
        self.output['R'][self.output['inactiveDF']] = Rs

        # Eigenfrequency analysis
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # Compute eigenvalues
        eigenfreqs = np.linalg.eig(-KrInv @ self.output['mr'])[0]
        # Proces first userdefined number of eigenvalues
        eigenfreqs = np.sqrt(np.abs(eigenfreqs[:self.input['nFreq']] ** (-1)))
        # Return eigenfrequencies to assembly.output.
        self.output['eigenfrequencies'] = eigenfreqs

        # get the length of each element
        old_lengths = np.array([element['length'] for element in self.mesh['element']])

        # get the new location of each of the nodes
        assert all(type == self.input['type'][0] for type in self.input['type']) is True
        if self.input['type'][0]:
            dof = 3
        else:
            dof = 2

        x_y_displacements = self.output['U'].reshape(self.output['U'].shape[0]//dof, dof)[:, :2]
        new_node_locations = self.mesh['nodes'] + x_y_displacements.T

        # get the new lengths of each element
        new_lengths = []
        for element in self.mesh['element']:
            node1 = new_node_locations[:, element['nodeNumber1']-1]
            node2 = new_node_locations[:, element['nodeNumber2']-1]
            new_lengths.append(np.linalg.norm(node2-node1))
        new_lengths = np.array(new_lengths)
        self.output['strain'] = (new_lengths - old_lengths)/old_lengths

        self.output['stress'] = np.zeros_like(self.output['strain'])

        E_array = np.array([element['E'] for element in self.mesh['element']])
        self.output['stress'] = E_array * self.output['strain']

    def plot_structure_stresses(self, show=True):
        # Calculate stress at each element
        element_stresses = self.output['stress']

        # Plot original elements
        plt.plot(self.mesh['nodes'][0, :], self.mesh['nodes'][1, :], 'o', '0.8',
                 linewidth=2, markerfacecolor='None', markersize=5)

        # Prepare stress values for display
        stress_min = float(min(element_stresses))
        stress_max = float(max(element_stresses))
        normalized_stresses = (element_stresses - stress_min) / (stress_max - stress_min)

        # Create a colormap
        cmap = plt.cm.Spectral

        # Plot displaced elements with stress-based color
        for i in range(len(element_stresses)):
            node1 = self.mesh['nodes'][:, self.mesh['element'][i]['nodeNumber1'] - 1]
            node2 = self.mesh['nodes'][:, self.mesh['element'][i]['nodeNumber2'] - 1]
            line_coord = np.vstack((node1, node2))
            color = cmap(normalized_stresses[i])
            plt.plot(line_coord[:, 0], line_coord[:, 1], color=color, label=str(i))

        # Add color bar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(element_stresses)
        plt.colorbar(sm, label='Stress [MPa]')

        # Set horizontal and vertical axis limits
        range_x = max(self.mesh['nodes'][0, :]) - min(self.mesh['nodes'][0, :])
        range_y = max(self.mesh['nodes'][1, :]) - min(self.mesh['nodes'][1, :])
        plt.xlim([min(self.mesh['nodes'][0, :]) - 0.1 * range_x, max(self.mesh['nodes'][0, :]) + 0.1 * range_x])
        if range_y != 0:
            plt.ylim([min(self.mesh['nodes'][1, :]) - 0.1 * range_y, max(self.mesh['nodes'][1, :]) + 0.1 * range_y])
        else:
            plt.ylim([min(self.mesh['nodes'][1, :]) - 0.5 * range_x, max(self.mesh['nodes'][1, :]) + 0.5 * range_x])

        # Adds gridline
        plt.minorticks_on()
        plt.grid(True, which="major", linestyle="-",alpha=0.2)
        plt.grid(True, which="minor",linestyle=":",alpha=0.7)

        # Adds floor
        plt.hlines(0, min(self.mesh['nodes'][0, :]) - 0.1 * range_x,
                   max(self.mesh['nodes'][0, :]) + 0.1 * range_x, "k", zorder=-100)
        plt.fill_between([min(self.mesh['nodes'][0, :]) - 0.1 * range_x,
                          max(self.mesh['nodes'][0, :]) + 0.1 * range_x],
                         min(self.mesh['nodes'][1, :]) - 0.1 * range_y, 0, color='gray', alpha=0.5)

        # Add title and labels
        plt.xlabel('x-displacement [mm]')
        plt.ylabel('y-displacement [mm]')

        # Show plot
        if show:
            plt.show()

    def plot_output(self, color='red', show=True):
        # Reshape displacement vector based on element type
        locDisp3D = self.output['U'].reshape(3, self.mesh['nNodes'], order='F') if np.all(self.input['type'] == 1) else \
        self.output['U'].reshape(2, self.mesh['nNodes'], order='F')

        # Calculate displaced node positions
        locDisp = self.mesh['nodes'] + self.input['plot_scale'] * locDisp3D[:2]

        # Plot elements
        for ix in range(max(self.input['parts'][0].shape)):
            plt.plot(self.input['points'][0, self.input['parts'][:, ix] - 1],
                     self.input['points'][1, self.input['parts'][:, ix] - 1],
                     '0.8', linewidth=2)

        # Plot original nodes
        plt.plot(self.mesh['nodes'][0], self.mesh['nodes'][1], 'o', '0.8', linewidth=2, markerfacecolor='None',
                 markersize=5)

        # Plot displaced nodes
        plt.plot(locDisp[0], locDisp[1], 'o', linewidth=2, markerfacecolor='None', markersize=5, color=color)

        # Calculate axis limits
        x_min, x_max = min(self.input['points'][0].min(), locDisp[0].min()), max(self.input['points'][0].max(),
                                                                                 locDisp[0].max())
        y_min, y_max = min(self.input['points'][1].min(), locDisp[1].min()), max(self.input['points'][1].max(),
                                                                                 locDisp[1].max())

        # Set axis limits
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        # Calculate tick locations
        x_ticks = np.linspace(x_min, x_max, 5)
        y_ticks = np.linspace(y_min, y_max, 5)

        # Set ticks
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)

        # Calculate decimal places for axis labels
        scaled_displacements_x, scaled_displacements_y = abs(self.input['plot_scale'] * locDisp3D[0:2, :])
        defmin_x = min(scaled_displacements_x[scaled_displacements_x > 0]) if not all(
            np.isclose(scaled_displacements_x, 0)) else 1
        defmin_y = min(scaled_displacements_y[scaled_displacements_y > 0]) if not all(
            np.isclose(scaled_displacements_y, 0)) else 1

        order_of_magnitude = lambda n: math.floor(math.log(n, 10))
        decimals_required = lambda n: abs(order_of_magnitude(n)) + 2 if order_of_magnitude(n) < 0 else 0

        # Format axis labels
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals_required(defmin_x)}f'))
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter(f'%.{decimals_required(defmin_y)}f'))

        # Configure minor ticks and grid
        plt.minorticks_on()
        plt.grid(True, which="major", linestyle="-",alpha=0.2)
        plt.grid(True, which="minor",linestyle=":",alpha=0.7)

        # Resize
        plt.tight_layout()

        # Show plot if specified
        if show:
            plt.show()

    def save_output(self):
        # TODO: export output for analysis
        pass
