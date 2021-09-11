#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

file_name = input("Input file name (without apostrophes): ")  # fetching filename with inputs from user
file_name2 = input("Output file name (without apostrophes): ")  # fetching output filename from user

'''
 input file has a following structure:
 line 1: number of 'commands' in a file
 lines 2 - 2+commands: commands with one of the two following forms:
    1) filename - bond - a - b
    2) filename - angle - a - b - c 
'''

# at_num_symbol - dic mapping atomic number to element symbol (up to 86 - Rn)
at_num_symbol = \
    {1: 'H', 2: 'He',
     3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
     11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
     19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
     30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
     37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag',
     48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
     55: 'Cs', 56: 'Ba', 57: 'La', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au',
     80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn',
     58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er',
     69: 'Tm', 70: 'Yb', 71: 'Lu'}

# 1 Bohr = 0.52917721 A
# 1 Hartree = 627.509608 kcal/mol
conversion = 627.509608 / 0.52917721 / 0.52917721  # Hartree / Bohr^2


class Atom(object):
    atomic_number = 0
    symbol = ""
    coordinates = np.empty((1, 3), dtype=np.float64)  # coordinates in Bohrs
    angs_coordinates = np.empty((1, 3), dtype=np.float64)  # coordinates in Angstroms

    # the class "constructor"
    def __init__(self, atomic_number, coordinates):
        self.atomic_number = int(float(atomic_number))
        self.symbol = at_num_symbol[int(float(atomic_number))]
        self.coordinates = coordinates
        self.angs_coordinates = self.coordinates * 0.52917721  # 1 Bohr = 0.52917721 A


def make_atom(atomic_number, coordinates):  # makes atom object and returns its reference
    atom = Atom(atomic_number, coordinates)
    return atom


class Bond(object):
    atom_one = None
    atom_two = None
    hessian = None  # hessian in Hartrees/Bohr^2
    kcm_hessian = None  # hessian in kilocalories/mol * A

    # the class "constructor"
    def __init__(self, atom_one, atom_two, hessian):
        self.atom_one = atom_one
        self.atom_two = atom_two
        self.hessian = hessian
        self.kcm_hessian = self.hessian * conversion
        self.kcm_hessian2 = np.transpose(self.kcm_hessian)


def make_bond(atom_one, atom_two, hessian):  # makes bond object and returns its reference
    bond = Bond(atom_one, atom_two, hessian)
    return bond


# hessian is a 3x3 matrix, because of that inserting values in right order requires translating
def insert_hessian(position, value, hessian):  # inserts read value to the matrix of bond hessian values
    if position == 0:
        hessian[0][0] = value
    elif position == 1:
        hessian[0][1] = value
    elif position == 2:
        hessian[0][2] = value
    elif position == 3:
        hessian[1][0] = value
    elif position == 4:
        hessian[1][1] = value
    elif position == 5:
        hessian[1][2] = value
    elif position == 6:
        hessian[2][0] = value
    elif position == 7:
        hessian[2][1] = value
    elif position == 8:
        hessian[2][2] = value


# FLAGS = headers that read_file is looking for in order to acquire information from given file
FLAGS = ["Nuclear charges", "Current cartesian coordinates", "Cartesian Force Constants"]


def read_file(file, atom_one, atom_two):
    bool_FLAGS = [True, True, True]
    with open(file, 'r') as file:
        file.seek(0)
        atom_one_charge = atom_two_charge = 0
        atom_one_coordinates = np.empty((1, 3), dtype=np.float64)
        atom_two_coordinates = np.empty((1, 3), dtype=np.float64)
        maximum, minimum = max(atom_one, atom_two), min(atom_one, atom_two)  # gets a value of higher, lower atom number
        for line in file:
            if line[0] != ' ':
                if bool_FLAGS[0]:  # "Nuclear charges"
                    words = set(line.split())  # header words
                    flag = set(FLAGS[0].split())  # flag words
                    if flag == flag & words:  # check if common words are the flag - Nuclear charges
                        bool_FLAGS[0] = False  # "Nuclear charges" flag found
                        value = 0  # value of current position- current considered atom charge
                        '''since there are 5 values in a row indexed starting from 1, given atom number we can 
                        determine amount of rows to skip using modulo5 and by 5 division 
                        [ 1 2 3 4 5  ] 
                        [ 6 7 8 9 10 ] 
                        [ ... ] 
                        skip1 -> minimum // 5 is an amount of rows preceding first atom charge 
                        skip2 is much harder to determine, because there are many possible edge cases, therefore 
                        its easier to underestimate the amount of rows skipped by subtracting 3 from most possible rows 
                        skipped and work the way up from there reading at most 3 additional rows with 5 values 
                        // is used instead of / since positions are int values '''
                        skip1 = minimum // 5
                        if minimum % 5 == 0: skip1 -= 1
                        skip2 = (maximum - minimum) // 5 - 3
                        if skip1 > 0:
                            for _ in range(skip1):
                                next(file)  # skipping a row
                            value += 5 * skip1  # correcting the current position value
                        newline = file.readline()
                        for j in newline.split():  # assessing atom_one_charge and possibly atom_two charge
                            value += 1
                            if value == minimum:
                                atom_one_charge = j
                            elif value == maximum:
                                atom_two_charge = j
                        if skip2 > 0:
                            for _ in range(skip2):
                                next(file)  # skipping a row
                            value += 5 * skip2  # correcting the current position value
                        while atom_two_charge == 0:  # if atom_two_charge wasn't in the same row
                            # while statement used as skip2 value is ambiguous
                            newline = file.readline()
                            for j in newline.split():
                                value += 1
                                if value == maximum:
                                    atom_two_charge = j  # assessing the atom_two charge

                elif bool_FLAGS[1]:  # "Current cartesian coordinates"
                    words = set(line.split())  # header words
                    flag = set(FLAGS[1].split())  # flag words
                    if flag == flag & words:  # check if common words are the flag - Current cartesian coordinates
                        bool_FLAGS[1] = False  # "Nuclear charges" flag found
                        value = 0  # value of current position- current considered atom coordinate
                        '''
                        since there are three coordinates for each atom and positions are indexed starting from 1,
                        the formula 3n-2 gives a position of the x coordinate of corresponding n-th atom 
                        skip values are ascertained similiary to "Nuclear charge" flag section, the only difference
                        being 2 more values per atom in this section 
                        '''
                        atom_one_position = 3 * minimum - 2  # position of the x coordinate of the first atom
                        atom_two_position = 3 * maximum - 2  # position of the x coordinate of the second atom
                        skip1 = atom_one_position // 5  # lines preceding lower atom x coordinate
                        if atom_one_position % 5 == 0: skip1 -= 1
                        skip2 = (atom_two_position - atom_one_position) // 5 - 3  # lines between lower and higher
                        # atom x coordinate
                        check1 = True  # bool to make sure all three coordinates of atom one were found
                        check2 = True  # bool to make sure all three coordinates of atom two were found
                        if skip1 > 0:
                            for _ in range(skip1):
                                next(file)  # skipping a row
                            value += 5 * skip1  # correcting the current position value
                        '''
                         all three coordinates are not necessarily in the same row, one of the edge cases being:
                        [ aa aa aa aa x1 ]
                        [ y1 z1 x2 y2 z2 ]
                        [ aa aa aa aa ...]
                        its vital to consider each value for each position in not skipped rows 
                        '''
                        while check1:
                            newline = file.readline()
                            for j in newline.split():
                                value += 1
                                if value == atom_one_position:  # x1
                                    atom_one_coordinates[0][0] = j  # inserts found coordinate to coordinates vector
                                elif value == atom_one_position + 1:  # y1
                                    atom_one_coordinates[0][1] = j
                                elif value == atom_one_position + 2:  # z1
                                    atom_one_coordinates[0][2] = j
                                    check1 = False  # all coordinates of first atom found
                                elif value == atom_two_position:  # x2
                                    atom_two_coordinates[0][0] = j
                                elif value == atom_two_position + 1:  # y2
                                    atom_two_coordinates[0][1] = j
                                elif value == atom_two_position + 2:  # z2
                                    atom_two_coordinates[0][2] = j
                                    check2 = False  # for some cases when the numbers are one after the other, and
                                    # coordinates of the first one are in two rows, its possible to find coordinates
                                    # of both atoms at the same time
                        if skip2 > 0:
                            for _ in range(skip2):
                                next(file)  # skipping a row
                            value += 5 * skip2  # correcting the current position value
                        while check2:
                            newline = file.readline()
                            for j in newline.split():
                                value += 1
                                if value == atom_two_position:  # x2
                                    atom_two_coordinates[0][0] = j
                                elif value == atom_two_position + 1:  # y2
                                    atom_two_coordinates[0][1] = j
                                elif value == atom_two_position + 2:  # z2
                                    atom_two_coordinates[0][2] = j
                                    check2 = False  # all three coordinates of the second atom were found

                elif bool_FLAGS[2]:  # "Cartesian Force Constants
                    words = set(line.split())  # header words
                    flag = set(FLAGS[2].split())  # flag words
                    if flag == flag & words:  # check if common words are the flag - Cartesian Force Constants
                        bool_FLAGS[2] = False  # "Cartesian Force Constants" flag found
                        value = 0  # value of current position- current considered hessian value
                        '''
                             x1  y1  z1  x2  y2  z2  x3 y3
                             _   _   _   _   _   _   _  _ 
                        x1 [ 1 
                        y1 [ 2   3 
                        z1 [ 4   5   6 
                        x2 [ 7   8   9   10
                        y2 [ 11  12  13  14  15
                        z2 [ 16  17  18  19  20  21
                        x3 [ 22  23  24  25  26  27  28
                        y3 [ ...
                        The key to get right values in hessian is establishing the position of a first value in said
                        hessian. Dividing 3 from the bigger atom number multiplied by 3 provides a shift value - 
                        number of rows preceding the row with the first hessian value. Then, 3n - 2 formula used
                        smaller atom number gives us the right position within the row. To jump from the first position
                        to fourth is a matter of adding a shift value plus one- since shift is a number of elements in
                        last skipped row. Similarly jump from fourth to seventh position can be made.  
                        '''
                        shift = 3 * maximum - 3  # number of rows skipped
                        # positions of atoms in file corresponding to hessian matrix:
                        first_position = (3 * minimum - 2) + sum(k for k in range(1, shift + 1))
                        fourth_position = first_position + shift + 1
                        seventh_position = fourth_position + shift + 2
                        hessian = [first_position, first_position + 1, first_position + 2,
                                   fourth_position, fourth_position + 1, fourth_position + 2,
                                   seventh_position, seventh_position + 1, seventh_position + 2]
                        hessian_values = np.empty((3, 3),
                                                  dtype=np.float64)  # empty float 3x3 matrix to collect actual hessian
                        position = 0  # itterates through positions in matrix
                        hessian_value = first_position  # first position that will be looked for
                        skip1 = first_position // 5  # amount of rows before first position
                        if first_position % 5 == 0: skip1 -= 1
                        if skip1 > 0:
                            for _ in range(skip1):
                                next(file)  # skipping a row
                            value += 5 * skip1  # correcting the current position value
                        while position < 9:  # as long as all 9 values are not found
                            newline = file.readline()
                            for j in newline.split():
                                value += 1
                                if value == hessian_value:
                                    insert_hessian(position, j, hessian_values)  # insert value to hessian_value
                                    position += 1  # gets new position
                                    if position < 9: hessian_value = hessian[position]  # gets new position

                        a1 = make_atom(atom_one_charge, atom_one_coordinates)  # create atom one
                        a2 = make_atom(atom_two_charge, atom_two_coordinates)  # create atom two
                        if atom_one > atom_two:  # if atoms in a given file are in an order: higher value, lower value-
                            # its necessary to switch found values afterwards
                            temp = a1
                            a1 = a2
                            a2 = temp
                        bond = make_bond(a1, a2, hessian_values)  # create bond
                        file.close()
                        return a1, a2, bond  # return [object(a1), object(a2), object(bond)]
    print("file error")  # the function executed on a well-structured file should never reach this point
    file.close()


# read_file collects information about charges, cartesian coordinates and hessian between two given atoms

'''
The norm of a vector is zero if and only if the vector is a zero vector. All possible atom positions require the atoms
to be in different positions so vector between them will never be zero- there is no need for checking whether the norm
differs from 0 when dividing.  
'''

# bond_info calls read_file and determines bond force constant among other figures
def bond_info(file, atom_one, atom_two):
    # result  =  [ atom one, atom two,   bond one-two   ]
    result = read_file(file, atom_one, atom_two)

    w, v = np.linalg.eig(-1. * result[2].kcm_hessian)  # eigenvalues and eigenvectors
    diff_u_ab = result[1].angs_coordinates - result[0].angs_coordinates  # difference in B, A coordinates
    u_ab = diff_u_ab / np.linalg.norm(diff_u_ab)  # normalized vector
    k = sum(w[i] * np.abs(np.dot(u_ab, v[:, i])) for i in range(3))  # k bond force constant for first hessian

    w2, v2 = np.linalg.eig(-1. * result[2].kcm_hessian2)  # second eigenvalues and eigenvectors
    k2 = sum(w2[i] * np.abs(np.dot(u_ab, v2[:, i])) for i in range(3))  # k bond force constant for second hessian
    '''
    print(w[0])
    print(np.abs(np.dot(u_ab,v[:,0])))
    print(w[0] * np.abs(np.dot(u_ab,v[:,0])))
    print(w[1])
    print(np.abs(np.dot(u_ab,v[:,1])))
    print(w[0] * np.abs(np.dot(u_ab,v[:,1])))
    print(w[2])
    print(np.abs(np.dot(u_ab,v[:,2])))
    print(w[2] * np.abs(np.dot(u_ab,v[:,2])))
    '''

    K = (k + k2) / 2  # mean k force constant for two calculated values
    distance = np.linalg.norm(diff_u_ab)  # distance AB
    return [result[0],result[1],distance,K]
    #return [atom A, atom B, distance AB, k_force_constant]

# angle_info calls read_file for atoms (1,2), (2,3) and determines bond angle force constant among other figures
def angle_info(file, atom_one, atom_two, atom_three):
    # result  =  [ atom one, atom two,   bond one-two   ]
    # result2 =  [ atom two, atom three, bond two-three ]
    result = read_file(file, atom_one, atom_two)
    result2 = read_file(file, atom_two, atom_three)

    # eigenvalues and eigenvectors for A-B, B-C hessians
    w, v = np.linalg.eig(-1. * result[2].kcm_hessian)
    w2, v2 = np.linalg.eig(-1. * result2[2].kcm_hessian)

    # eigenvalues and eigenvectors for transposed A-B, B-C hessians
    w3, v3 = np.linalg.eig(-1. * result[2].kcm_hessian2)
    w4, v4 = np.linalg.eig(-1. * result2[2].kcm_hessian2)


    diff_u_ab = result[1].angs_coordinates - result[0].angs_coordinates  # difference in coordinates B - A
    u_ab = diff_u_ab / np.linalg.norm(diff_u_ab)

    #diff_u_ba = result[0].angs_coordinates - result[1].angs_coordinates  # A - B
    #u_ba = diff_u_ba / np.linalg.norm(diff_u_ba)


    diff_u_cb = result2[0].angs_coordinates - result2[1].angs_coordinates  # difference in coordinates B - C
    u_cb = diff_u_cb / np.linalg.norm(diff_u_cb)

    #diff_u_bc = result2[1].angs_coordinates - result2[0].angs_coordinates  # C - B
    #u_bc = diff_u_bc / np.linalg.norm(diff_u_bc)


    cross = np.cross(u_cb, u_ab)  # cross product of u_cb and u_ab vectors
    cross_norm = cross / np.linalg.norm(cross)  # normalised u_n
    u_pa = np.cross(cross_norm, u_ab)  # u_pa = u_n x u_ab
    u_pc = np.cross(u_cb, cross_norm)  # u_pc = u_cb x u_n
    R_ab_sq = np.linalg.norm(diff_u_ab) ** 2  # length of bond ab squared
    R_cb_sq = np.linalg.norm(diff_u_cb) ** 2  # length of bond cb squared

    # one over k formula
    one_over_kt =\
    1 / (2 * (R_ab_sq * sum(w[i] *  np.abs(np.dot(u_pa, v[:, i])) for i in range(3)))) +\
    1 / (2 * (R_ab_sq * sum(w3[i] * np.abs(np.dot(u_pa, v3[:, i])) for i in range(3)))) +\
    1 / (2 * (R_cb_sq * sum(w2[i] * np.abs(np.dot(u_pc, v2[:, i])) for i in range(3)))) +\
    1 / (2 * (R_cb_sq * sum(w4[i] * np.abs(np.dot(u_pc, v4[:, i])) for i in range(3))))
    k_theta = 1 / one_over_kt  # angle force constant   

    # angle ABC in degrees
    angle = np.degrees(np.arccos(u_ab[0][0] * u_cb[0][0] + u_ab[0][1] * u_cb[0][1] + u_ab[0][2] * u_cb[0][2]))
    '''
    one_over_kt = 1 / (R_ab_sq * sum(
    (w[i] / 2) * np.abs(np.dot(u_pa, v[:, i])) + (w3[i] / 2) * np.abs(np.dot(u_pa, v3[:, i])) for i in range(3))) +\
                1 / (R_cb_sq * sum(
    (w2[i] / 2) * np.abs(np.dot(u_pc, v2[:, i])) + (w4[i] / 2) * np.abs(np.dot(u_pc, v4[:, i])) for i in range(3)))
    R_ab_sq * sum((w[i]/2) * np.abs(np.dot(u_pa, v[:, i])) + (w3[i]/2) * np.abs(np.dot(u_pa, v3[:, i])))
    R_cb_sq * sum((w2[i]/2) * np.abs(np.dot(u_pa, v2[:, i])) + (w4[i]/2) * np.abs(np.dot(u_pa, v4[:, i])))
    '''

    return (result[0],result[1],result2[1],angle,k_theta)
    #return [atom1, atom2, atom3, angle_in_degrees, angle_force_constant]

    # driver code
with open(file_name, 'r') as input_file:
    output_file = open(file_name2, "w+")
    commands = input_file.readline()
    for c in range(int(commands)):  # for each command in input file
        command = input_file.readline().split()  # ascertain between bond/angle
        if command[1] == "bond":  # bond_info (file_name, atom_one, atom_two)
            bond = bond_info(command[0], int(command[2]), int(command[3]))
            # write into an output: no. file_name bond atom1.symbol-atom2.symbol distance12 bond_force_constant
            output_file.write(str(c+1) + " " + command[0] + " " + command[1] + " " + str(bond[0].symbol) +
                              "-" + str(bond[1].symbol) + " " + str(bond[2]) + " " + str(bond[3][0]) + "\n")
        elif command[1] == "angle":  # angle_info (file_name, atom_one, atom_two (the atom in the middle), atom_three)
            angle = angle_info(command[0], int(command[2]), int(command[3]), int(command[4]))
            # write into an output: no. file_name angle atom1.symbol-atom2.symbol-atom3symbol
            # angle123_in_degrees angle_force_co nstant
            output_file.write(str(c+1) + " " + command[0] + " " + command[1] + " " + str(angle[0].symbol) +
                              "-" + str(angle[1].symbol) + "-" + str(angle[2].symbol) + " "
                              + str(angle[3]) + " " + str(angle[4][0]) + "\n")
        else:
            print("line ", c, ": incorrect command name")
    input_file.close()
    output_file.close()

