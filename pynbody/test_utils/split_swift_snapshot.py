import re

import h5py
import numpy as np


def _update_array_attribute(obj, attr_name, index, value):
    """Update an element in a HDF5 object array attribute"""
    data = obj.attrs[attr_name]
    data[index] = value
    obj.attrs.modify(attr_name, data)


def split_swift_snapshot(template_file, nr_files, cell_file_index,
                         output_name, cell_mask=None):
    """
    Given a single file SWIFT snapshot at path template_file, write a new,
    multi-file snapshot by assigning cells to files using cell_file_index.
    """
    input_snap = h5py.File(template_file, "r")

    # Remove any .X.hdf5 suffix from the output name
    output_name = str(output_name)
    m = re.match(r"^(.*)\.[0-9]+\.hdf5", output_name)
    if m is not None:
        output_name = m.group(1)

    # Get list of particle types and cell metadata
    ptypes = list(input_snap["Cells/Counts"])
    cell_counts = {}
    cell_input_offsets = {}
    for ptype in ptypes:
        cell_counts[ptype] = input_snap["Cells/Counts"][ptype][...]
        cell_input_offsets[ptype] = input_snap["Cells/OffsetsInFile"][ptype][...]
        # Here we assume that the cells are sorted by offset in the input snapshot
        assert np.all(cell_input_offsets[ptype][1:] >= cell_input_offsets[ptype][:-1])

    # If a cell mask is specified, discard cells which are not selected
    if cell_mask is not None:
        for ptype in cell_mask:
            cell_counts[ptype][cell_mask[ptype]==False] = 0

    # Check total number of cells
    nr_cells = len(cell_file_index)
    assert nr_cells == input_snap["Cells/Centres"].shape[0]

    # Compute the offset to each cell in the output files
    cell_output_offsets = {}
    for ptype in ptypes:
        cell_output_offsets[ptype] = np.zeros(nr_cells, dtype=np.int64)
        particles_in_file = np.zeros(nr_files, dtype=int)
        for cell_nr in range(nr_cells):
            file_nr = cell_file_index[cell_nr]
            cell_output_offsets[ptype][cell_nr] = particles_in_file[file_nr]
            particles_in_file[file_nr] += cell_counts[ptype][cell_nr]

    # Loop over output files to create
    for file_nr in range(nr_files):
        output_snap = h5py.File(f"{output_name}.{file_nr}.hdf5", "w")

        # Copy everything except the particle data arrays to the output
        for name, h5obj in input_snap.items():
            if not name.startswith("PartType"):
                input_snap.copy(name, output_snap)
        output_snap["Header"].attrs.modify("NumFilesPerSnapshot", nr_files)

        # Loop over particle types
        for ptype in ptypes:

            # Count particles of this type in this file
            nr_particles = sum(cell_counts[ptype][cell_file_index==file_nr])
            if nr_particles == 0:
                continue
            total_nr_particles = sum(cell_counts[ptype])

            # Create the PartType group
            group = output_snap.create_group(ptype)
            for attr_name, attr_val in input_snap[ptype].attrs.items():
                group.attrs[attr_name] = attr_val
            group.attrs.modify("NumberOfParticles", nr_particles)
            group.attrs.modify("TotalNumberOfParticles", total_nr_particles)

            # Update header attributes for this type
            index = int(ptype[-1])
            assert f"PartType{index}" == ptype
            _update_array_attribute(output_snap["Header"], "NumPart_ThisFile", index, nr_particles)
            _update_array_attribute(output_snap["Header"], "NumPart_Total", index, total_nr_particles)
            _update_array_attribute(output_snap["Header"], "NumPart_Total_HighWord", index, 0)

            # Update cell metadata
            output_snap["Cells/OffsetsInFile"][ptype][...] = cell_output_offsets[ptype]
            output_snap["Cells/Files"][ptype][...] = cell_file_index

            # Copy dataset contents and attributes
            for name in input_snap[ptype]:
                input_dset = input_snap[ptype][name]
                assert isinstance(input_dset, h5py.Dataset)
                shape = list(input_dset.shape)
                shape[0] = nr_particles
                output_dset = output_snap[ptype].create_dataset_like(name, input_dset, shape=shape, chunks=None)
                for attr_name, attr_val in input_dset.attrs.items():
                    output_dset.attrs[attr_name] = attr_val
                for input_offset, output_offset, count in zip(cell_input_offsets[ptype][cell_file_index==file_nr],
                                                              cell_output_offsets[ptype][cell_file_index==file_nr],
                                                              cell_counts[ptype][cell_file_index==file_nr]):
                    output_dset[output_offset:output_offset+count,...] = input_dset[input_offset:input_offset+count,...]

        output_snap.close()

    input_snap.close()


def hash_swift_cell_coordinates(filename):
    """Hash the coordinates of particles in each cell in sequence. This hash
    should not be affected by splitting a snapshot into multiple files,
    assuming the input did not use lossy compression.
    """
    with h5py.File(filename, "r") as f0:
        nr_files = f0["Header"].attrs["NumFilesPerSnapshot"][0]
        nr_cells = f0["Cells/Centres"].shape[0]

    # Get the full list of filenames
    filename = str(filename)
    if nr_files == 1:
        filenames = [filename,]
    else:
        m = re.match(r"^(.*)\.[0-9]+\.hdf5", filename)
        assert m is not None
        filenames = [f"{m.group(1)}.{file_nr}.hdf5" for file_nr in range(nr_files)]

    # Open all of the files
    snap_file = [h5py.File(filename, "r") for filename in filenames]

    # For each cell, read the particle coordinates and add them to the hash
    import hashlib
    m = hashlib.sha256()
    ptypes = sorted(list(snap_file[0]["Cells/Counts"]))
    for ptype in ptypes:
        cell_file_index = snap_file[0]["Cells/Files"][ptype][...]
        cell_offsets = snap_file[0]["Cells/OffsetsInFile"][ptype][...]
        cell_counts = snap_file[0]["Cells/Counts"][ptype][...]
        all_coords = [snap_file[file_nr][ptype]["Coordinates"][...] for file_nr in range(nr_files)]
        for cell_nr in range(nr_cells):
            file_nr = cell_file_index[cell_nr]
            offset = cell_offsets[cell_nr]
            count = cell_counts[cell_nr]
            pos = all_coords[file_nr][offset:offset+count,...]
            m.update(pos)

    for f in snap_file:
        f.close()

    return m.hexdigest()
