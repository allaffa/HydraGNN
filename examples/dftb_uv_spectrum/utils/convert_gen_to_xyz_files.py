#----------------------------------------------------------------------------#
def generate_xyz_files(molecule_directory):
    input_dftb_file = '%s/geo_end.gen' % (molecule_directory)
    try:
        num_atoms = None
        atom_dict = {}
        atom_xyz = []
        atom_counter = 1
        with open(input_dftb_file, 'r') as genfile:
            for row in genfile:
                if num_atoms is None:
                    num_atoms = int(row.split()[0].strip())
                    continue
                if len(atom_dict) == 0:
                    for atom_type in [x.strip() for x in row.split()]:
                        atom_dict[atom_counter] = atom_type
                        atom_counter += 1
                    continue

                row_split = [x.strip() for x in row.split()]
                atom_xyz.append('%s\t%s' % (atom_dict[int(row_split[1])], '\t'.join(row_split[2:])))

        with open('%s/geo_end.xyz' % (molecule_directory), 'w') as hlfile:
            hlfile.write('%d\n' % num_atoms)
            atom_types = []
            for index in range(1, atom_counter):
                atom_types.append(atom_dict[index])
            hlfile.write('%s\n' % ' '.join(list(atom_types)))
            hlfile.write('%s\n' % '\n'.join(atom_xyz))
    except Exception as e:
        print(f"Error Generating XYZ File for Molecule in directory {molecule_directory}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise e
