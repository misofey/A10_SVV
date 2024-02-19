# -*- coding: utf-8 -*-


import numpy as np

from objects.assembly            import Assembly
from objects.mesh                import Mesh
from objects.element             import Element

from telescope import telescope_geometry


def MechRes2D(inp):
    # =============================================================================
    #  AE3212-II Structures assignment 2021-2023
    # =============================================================================
    # ============================Start of preamble================================
    # This Python-function belongs to the AE3212-II Simulation, Verification and
    # Validation Structures assignment. The file contains a function that
    # calculates the mechanical response of a 2D structure that has been
    # specified by the user.
    #
    # The function uses a structure as input and returns a structure as output.
    #
    # Proper functioning of the code can be checked with the following input:
    #
    # Written by Antonio López Rivera and Václav Marek,
    # based on the original by Julien van Campen
    # Aerospace Structures and Computational Mechanics
    # TU Delft
    # October 2021 â€“ January 2022 & January 2023
    # ==============================End of Preamble================================
    
    # --------------------------------------------------------------------------
    # 0. User Input
    # --------------------------------------------------------------------------
    assembly = Assembly(inp)

    # --------------------------------------------------------------------------
    # 1. Input Check
    # --------------------------------------------------------------------------
    if PLOT:
        assembly.plot_input()
    
    # --------------------------------------------------------------------------
    # 2. Creation of Parts - Plot for check by user
    # --------------------------------------------------------------------------    
    # Create mesh
    assembly.mesh = Mesh(assembly)
        
    if PLOT:
        # Plot mesh
        assembly.mesh.plot_mesh()
        # Show the user the mesh overlaid over the geometry
        assembly.mesh.plot_mesh(False)
        assembly.plot_input()

        # Pause for user to do visual inspection of structure and mesh
        input('Please inspect the mesh shown in figures 1 and 2 and 3. \
               They may also be saved in the working path as "mesh_1.png", "mesh_2.png", and "mesh_3.png", if so configured. \
               Press enter to continue.')
    
    # --------------------------------------------------------------------------
    # 3. Creation of Elements
    # --------------------------------------------------------------------------    
    # Assign properties
    assembly.mesh.assign_element_properties()

    # Creation of Elements
    for ix in range(assembly.mesh.mesh['nElements']):
        # for each element, generate local properties
        element = Element(assembly, ix)
        # rewrite the local mesh with the mesh that now has the element properties included
        assembly.mesh = element.assembly.mesh

    # --------------------------------------------------------------------------
    # 4. Assembling the structure
    # --------------------------------------------------------------------------
    assembly.global_mass_matrix()
    assembly.global_stiffness_matrix()
    assembly.global_thermal_load_vector()
    
    # --------------------------------------------------------------------------
    # 5. Application of Loads and BCs
    # --------------------------------------------------------------------------
    assembly.loads_bc()
    
    # --------------------------------------------------------------------------
    # 6. Performing the analysis
    # TODO: Calculate stresses and strains
    # --------------------------------------------------------------------------
    o = assembly.output
    
    # Displacements and reaction forces
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    KrInv = np.linalg.inv(o['Kr'])
    Ur = KrInv@(o['Pr']-o['Qr'])
    # Displacements of entire system
    o['U'][o['activeDF']] = Ur
    # Reaction forces
    Rs = o['Ksr']@Ur - (o['Ps']-o['Qs'])
    # Reaction forces of entire system
    o['R'][o['inactiveDF']] = Rs

    # Eigenfrequency analysis
    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Compute eigenvalues
    eigenfreqs = np.linalg.eig(-KrInv@o['mr'])[0]
    # Proces first userdefined number of eigenvalues
    eigenfreqs = np.sqrt(np.abs(eigenfreqs[:inp['nFreq']]**(-1)))
    # Return eigenfrequencies to assembly.output.
    o['eigenfrequencies'] = eigenfreqs

    # Write any changes in the output alias back to the assembly's output
    assembly.output = o
    
    # --------------------------------------------------------------------------
    # 7. Plot results
    # TODO: improve plots to ease comparison, etc.
    # --------------------------------------------------------------------------
    assembly.plot_output()

    # --------------------------------------------------------------------------
    # 8. Save simulation results
    # TODO: fill the block (consider CSV, pickling)
    # --------------------------------------------------------------------------
    assembly.save_output()
    
    return assembly.output


def ConstGeom(inp):
    # This script constructs the geometry of the parts, as input for MechRes

    # angles dish
    phi1 = 0
    phi2 = inp['phi']/3/180*np.pi
    phi3 = 2*phi2
    phi4 = inp['phi']/180*np.pi
    # shared x-coordinates
    x1 = -inp['R']*np.sin(phi1)
    x2 = -inp['R']*np.sin(phi2)
    x3 = -inp['R']*np.sin(phi3)
    x4 = -inp['R']*np.sin(phi4)
    # shared y-coordinates
    y1 = inp['cl']+inp['R']*(1-np.cos(phi1))
    y2 = inp['cl']+inp['R']*(1-np.cos(phi2))
    y3 = inp['cl']+inp['R']*(1-np.cos(phi3))
    y4 = inp['cl']+inp['R']*(1-np.cos(phi4))
    # angle legs
    theta = np.arctan((inp['b']+x3)/y3)
    # create structure for selected type
    out = {}  # initialize output
    if inp['type']:
        # ---- beam structure ----
        # x-coordinates
        x5 = -inp['b']
        # y-coordinates
        y5 = 0
        # assemble points
        out['points'] = np.array([[x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5]])
        out['parts'] = np.array([[1, 2, 3, 3], [2, 3, 4, 5]])
        # element type
        out['type'] = np.ones((max(out['parts'].shape)), dtype=int)
    else:
        # ---- rod structure ----
        # x-coordinates
        x5 = x3 + inp['cl']/2*np.cos(np.pi/3+theta)
        x6 = x3 - inp['cl']/2*np.cos(np.pi/3-theta)
        x7 = -inp['b']
        # y-coordinates
        y5 = y3 - inp['cl']/2*np.sin(np.pi/3+theta)
        y6 = y3 - inp['cl']/2*np.sin(np.pi/3-theta)
        y7 = 0
        # assemble points
        out['points'] = np.array(
            [[x1, x2, x3, x4, x5, x6, x7], [y1, y2, y3, y4, y5, y6, y7]])

        out['parts'] = np.array([[1, 2, 3, 1, 2, 3, 3, 4, 5, 5, 6], [
                                2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7]])
        # element type
        out['type'] = np.zeros((max(out['parts'].shape)), dtype=int)

    # initialise seed for every part to 0
    out['seed_array'] = np.zeros((max(out['parts'].shape)))
    
    # initialise material for every part to 0
    out['material_indeces'] = np.zeros((max(out['parts'].shape)), dtype=int)
    # initialise section for every part to 0
    out['section_indeces'] = np.zeros((max(out['parts'].shape)), dtype=int)

    # create boundary conditions
    # 1 means d.o.f. is restrained, 0 means d.o.f. is free
    out['bc'] = np.zeros((3*max(out['points'].shape)))
    # final point is clamped
    out['bc'][-3:] = 1

    # create loading vector
    out['P'] = np.zeros((3*max(out['points'].shape), 1))

    # create thermal vector
    out['T'] = np.zeros((max(out['points'].shape), 1))
    if inp['type']:
        # first 4 nodes are exposed to deltaT
        out['T'][0:4] = inp['deltaT']*np.ones((4, 1))
    else:
        # all nodes are exposed to deltaT
        out['T'] = inp['deltaT']*np.ones((max(out['points'].shape)))

    # create mirror side of structure, if so selected
    if inp['sym']:
        # connectivity
        a = np.nonzero(out['parts'][0, :] == 1)[0]
        l = max(out['points'].shape)
        # points (first point should not be duplicated)
        out['points'] = np.vstack((np.concatenate((out['points'][0, :], -out['points'][0, 1:]), axis=0).reshape(
            1, 2*l-1), np.concatenate((out['points'][1, :], out['points'][1, 1:]), axis=0).reshape(1, 2*l-1)))
        # boundary conditions (first point should not be duplicated)
        out['bc'] = np.concatenate((out['bc'], out['bc'][3:]))
        # loading vector
        out['P'] = np.concatenate((out['P'], out['P'][3:]))
        # thermal vector
        out['T'] = np.concatenate((out['T'], out['T'][1:]))
        # advancing the node numbers to be connected apart for node 1
        c = (l-1)*np.ones((np.shape(out['parts'])), dtype=int)
        c[0, a] = np.zeros((1, len(a)), dtype=int)
        out['parts'] = np.concatenate((out['parts'], c+out['parts']), axis=1)
        # double the initialized vectors
        out['type'] = np.concatenate((out['type'], out['type']))
        out['seed_array'] = np.concatenate((out['seed_array'], out['seed_array']))
        out['material_indeces'] = np.concatenate((out['material_indeces'], out['material_indeces']))
        out['section_indeces'] = np.concatenate((out['section_indeces'], out['section_indeces']))
    else:
        # apply symmetry boundary conditions at point 1
        out['bc'][0] = 1 # 1 means d.o.f. is restrained, 0 means d.o.f. is free
        out['bc'][2] = 1

    # convert theta to degrees
    out['theta'] = theta*180/np.pi
    # store input data to output
    out['inp'] = inp
    out['mat'] = inp['mat']
    out['sec'] = inp['sec']
    out['seeds'] = inp['seeds']
    out['lump_mass'] = inp['lump_mass']
    out['name'] = inp['name']
    out['plot_scale'] = inp['plot_scale']
    return out


if __name__ == '__main__':

    # Set to True to show input validation plots
    PLOT = True
    
    # Create input geometry from TOML file
    inp = telescope_geometry('telescope.toml')

    # Run simulation
    output = MechRes2D(inp)
