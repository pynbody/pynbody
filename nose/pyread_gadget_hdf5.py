#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routine for reading Gadget3 HDF5 output files."""

import pandas as pd
import h5py
import numpy
import fnmatch
import os


__author__ = 'Alan Duffy'
__email__ = 'mail@alanrduffy.com'
__version__ = '0.1.0'


def pyread_gadget_hdf5(filename, ptype, var_name, sub_dir=None,\
    smooth=None,cgsunits=None,physunits=None,floatunits=None,\
    silent=None,nopanda=None,noloop=None,leaveh=None):

    """
    +
     NAME:
           PYREAD_GADGET_HDF5
    
     PURPOSE:
           This function reads in HDF5 outputs (snapshots, FOF and SubFind files) of the OWLS Gadget3 code
    
     CATEGORY:
           I/O, HDF5, OWLS / PAISTE :)
    
     REQUIREMENTS:
             import pandas as pd
             import h5py
             import numpy
             import fnmatch
             import os
    
     CALLING SEQUENCE:
             from pyread_gadget_hdf5 import *
           Result = pyread_gadget_hdf5(filename, ptype, var_name
           [, sub_dir=sub_dir, smooth=True, cgsunits=True, physunits=True,
                floatunits=True, silent=True, nopanda=True, noloop=True,
                leaveh=True] )
    
     INPUTS:
           filename: Name of the file to read in. In case of a multiple files
                     output, any sub-file name can be given. The routine will
                     take care of reading data from each sub-file.
           ptype:    Defines either the particle type or the sub-set for
                     which the data has to be read in (see table below).
           sub_dir:  Define the group (/FOF or /SUBFIND) from which
                     particle and/or subhaloes properties are read-in
                     from (see table below).
    
     ptype | snapshot                     | Old FOF   
     ------+------------------------------+-------------
         0 | gas particles                | FOF/      
         1 | dark matter particles        | FOF/NSF   
         2 | extra collisionless parts    | FOF/SF    
         3 | extra collisionless parts    | FOF/Stars 
         4 | star particles               | -         
         5 | extra collisionless parts    | -         
         6 | -                            | -         
         7 | virtual particles (TRAPHIC)  | -         
         8 | -                            | -         
         9 | -                            | -         
        10 | -                            | -         
    
     ptype | SubFind / sub_dir='FOF'      | SubFind / sub_dir='SUBFIND'
     ------+------------------------------+-----------------------------
         0 | gas particles                | gas particles
         1 | dark matter particles        | dark matter particles
         2 | extra collisionless parts    | extra collisionless parts
         3 | extra collisionless parts    | extra collisionless parts
         4 | star particles               | star particles
         5 | extra collisionless parts    | extra collisionless parts
         6 | -                            | -
         7 | -                            | -
         8 | -                            | -
         9 | -                            | -
        10 | FOF                          | SUBFIND
        11 | FOF/NSF                      | SUBFIND/NSF
        12 | FOF/SF                       | SUBFIND/SF
        13 | FOF/Stars                    | SUBFIND/Stars
    
            var_name: Dataset to read in.
    
     OPTIONAL INPUTS:
    
    
           sub_dir:  Kind of data group to read:
                     'FOF' - friends-of-friends group file
                     'SUBFIND' - subfind group file
    
     KEYWORD PARAMETERS (set to True)
           floatunits:   returns quantities in single precision
           cgsunits:       Converts from Gadget code units to CGS
           physunits:      Converts from Gadget code units to Proper (with h-> h_73)
                           and a small change with 10^10 Msol to Msol, and ENTROPY!
           silent:       does not print any message on the screen
           smooth:       reads SPH smoothed quantities
           noloop:     limit the read in to only the file selected
           nopanda:    convert from PANDAS dataframe to Numpy array (you still need Pandas to run!)
           leaveh:    set to true to leave 'litte h' values as code 'h' dependency, will have no effect
                        unless cgsunits or physunits selected in which case it overrides them regarding h
    
     OUTPUTS:
            An array (either 1D or 2D depending on variable) in PANDAS dataframe format
            or Numpy style, if nopanda=True set... but you shouldn't as PANDAS is fast.
    
     RESTRICTIONS:
    
    
     PROCEDURE:
    
    
     EXAMPLE:
        
           Reading star positions from a snapshot file in CGS units:
    
           pos = pyread_gadget_hdf5('snap_035.0.hdf5', 4, 'Coordinates', cgsunits=True)
        
           and in phys units (proper h_73^-1 Mpc):
    
           pos = pyread_gadget_hdf5('snap_035.0.hdf5', 4, 'Coordinates', physunits=True)
        
           and in code units:
    
           pos = pyread_gadget_hdf5('snap_035.0.hdf5', 4, 'Coordinates')
        
           Reading of FOF data from a SubFind output:
    
           pos = pyread_gadget_hdf5('subhalo_035.0.hdf5', 10, 'CenterOfMass', sub_dir='fof')
        
           Reading of FOF gas particle data from a SubFind output:
    
           FeIa= pyread_gadget_hdf5('subhalo_035.0.hdf5', 0, 'IronFromSNIa', sub_dir='fof')
        
           Reading of SPH Smoothed FOF gas particle data from a SubFind output:
    
           SmoothFeIa = pyread_gadget_hdf5('subhalo_035.0.hdf5', 0, 'IronFromSNIa', sub_dir='fof', smooth=True)
        
           Reading of SUBFIND non-star-forming gas data from a SubFind output:
    
           pos = pyread_gadget_hdf5('subhalo_035.0.hdf5', 11, 'Temperature', sub_dir='subfind')
    
        
    
     MODIFICATION HISTORY (by Alan Duffy):
            
            1/09/13 Converted Claudio Dalla Vecchia's superlative IDL script nread_gadget_hdf5.pro
            5/09/13 Removed ' input ' as an option in the script
            5/09/13 Removed ' noconversion ', default is to leave in Code units, i.e. don't convert
            5/09/13 Added ' cgsunits ' as the option to convert variables entirely to CGS units 
            (the default switch in the IDL version)
            5/09/13 Added ' physunits ' as an option to convert variables to proper units with h-> h_73
            as well as masses to Msol instead of 10^10, and MaximumEntropy in particular changes
            6/09/13 Added ' noloop ' as an option to force a read in from one file alone
            6/09/13 Changed header information to match flake8 standard
            25/09/13 Added ' leaveh ' as an option to make units remain in code 'h' dependency
            08/01/14 Fixed bug on looping over subfiles
            08/01/14 Force C-Contiguous array

            Any issues please contact Alan Duffy on mail@alanrduffy.com or (preferred) twitter @astroduff
    """

#########################################################
##
## Check user choices...
##
#########################################################

    if cgsunits != None and physunits != None:
        print("[ERROR] Can't convert units to both CGS and PHYS, as CGS includes PHYS, assuming CGS ")
        physunits == None

    if silent == None:
        print("User gave "+filename)

    ## Determine the file type, i.e. SubFind, Snapshot or old FOF.
    if filename.rfind('/subhalo') >= 0:
        inputtype = 'SubFind' ## Define File type
        inputname = 'subhalo'
    elif filename.rfind('/group') >= 0:
        inputtype = 'FoF' ## Define File type
        inputname = 'group' 
    elif filename.rfind('/snap') >= 0:
        inputtype = 'Snapshot' ## Define File type
        inputname = 'snap'
    if silent == None:
        print("This is a "+inputtype+" output filetype")

    ## Check that the file given and the ptype used makes sense
    if ptype < 0 or \
       inputtype == 'Snapshot' and ptype > 7 or \
       inputtype == 'FoF' and ptype > 3 or \
       inputtype == 'SubFind' and ptype > 13:
        print('[ERROR] ptype = '+str(ptype)+' is not allowed with this input file type ('+input+')')
        return -1

    if ptype < 6:
        if inputtype == 'SubFind' and sub_dir == None:
            print('[ERROR] ptype = '+str(ptype)+' is not allowed without specifying sub_dir flag')
            return -2

#########################################################
##
## Determine if, and how many, subfiles we must loop over
##
#########################################################
    folder_index = filename.rfind(inputname)
    snapnum_index = filename[:folder_index-1].rfind('_')
    snapnum = int(filename[snapnum_index+1:folder_index-1])
    numfile = len(fnmatch.filter(os.listdir(filename[:folder_index]), '*.hdf5'))
    if noloop != None:
        numfile = 1 ## Force the code to only consider the input file

    if silent == None:
        print("Considering "+str(numfile)+" subfiles")
#########################################################
##
##  Read in common header parameters
##
#########################################################

    with h5py.File(filename, "r") as fin:
        hubble = fin['Header'].attrs['HubbleParam']
        aexp = fin['Header'].attrs['ExpansionFactor']
        if numfile == 0: ## Stupid bug with one file subfind outputs, the total array isn't created.
            numpart_total = fin['Header'].attrs['NumPart_Total'] 
        else: ## instead must read the case for the single file 'ThisFile'
            numpart_total = fin['Header'].attrs['NumPart_ThisFile']
        if ptype < 6:
            if numpart_total[ptype] == 0:
                print("There are no particles of type "+str(ptype)+" quitting")
                return -3
        masstable = fin['Header'].attrs['MassTable']
#########################################################
##
##  Read in unit conversion 
##
#########################################################
        if cgsunits != None or physunits != None:
            gamma = fin['Constants'].attrs['GAMMA']
            solar_mass = fin['Constants'].attrs['SOLAR_MASS'] ## [g]
            boltzmann = fin['Constants'].attrs['BOLTZMANN'] ## [erg/K]
            protonmass = fin['Constants'].attrs['PROTONMASS'] ## [g]
            sec_per_year = fin['Constants'].attrs['SEC_PER_YEAR']
        if silent == None:
            print("Note h="+("%.3f" % hubble)+" in this simulation"+\
            " and snapshot is at a="+("%.3f" % aexp)+\
            " (z="+("%.3f" % (1./aexp-1.))+")")

#########################################################
##
##  Read in Chemical Parameters, i.e. Element Names, if present
##
#########################################################
        e = 'Parameters/ChemicalElements' in fin
        if e == True:
            nelements = fin['Parameters/ChemicalElements'].attrs['BG_NELEMENTS']
            elementnames = fin['Parameters/ChemicalElements'].attrs['ElementNames']
            if silent == None:
                print("Number of Elements Tracked "+str(nelements)+", they are")
                for elementtype in elementnames:
                    print(elementtype)

#########################################################
##
##  Assuming the user has given us the correct information 
##  let's work out what array to access
##  If particle type is less than 6 (more than 10) we want 
##  the actual particle values, otherwise 10 is the group variables
##  and 
##
##  Also checks the case for smooth_element variables!
##
#########################################################        

    ## For the case of a Baryon run, check if the user is asking for an element
    ## AND if that's a smoothed quantity then change their request slightly...
    if numpart_total[0] > 0 or numpart_total[4] > 0 or numpart_total[5] > 0:
        if var_name in elementnames:
            var_name = 'ElementAbundance/'+var_name         
        if smooth != None:
            var_name = 'Smoothed'+var_name
    if ptype < 6:
        global_var_name = '/'+'PartType'+str(ptype)+'/'+var_name+'/'
    elif ptype == 10:
        global_var_name = '/'+var_name+'/'
    elif ptype == 11:
        global_var_name = '/NSF/'+var_name+'/'
    elif ptype == 12:
        global_var_name = '/SF/'+var_name+'/'
    elif ptype == 13:
        global_var_name = '/Stars/'+var_name+'/'

    if sub_dir != None:
        global_var_name = sub_dir.upper()+global_var_name

    if silent == None:
        print("Now select the requested variable "+global_var_name)

#########################################################
##
##  Now loop over the subfiles and concatenate arrays
##
#########################################################
    prefix_index = filename.rfind('.hdf5') ## First HDF5
    for ifile in range(0,numfile):
        if noloop != None:
            newinfile = filename
        else:
            newinfile = filename[0:folder_index] + inputname +'_'+str(snapnum).zfill(3)+'.'+str(ifile)+filename[prefix_index:]   
        if silent == None:
            print("Reading from "+newinfile)

        with h5py.File(newinfile, "r") as fin:
## If the dataset is present in the file then read it into tmpset, and if dataset exists concatenate them
            e = global_var_name in fin
            if e == True:
                tmpset = pd.DataFrame(fin[global_var_name].value) ## Let HDF5 metadata dictate array format

                try:
                    dataset
                except NameError:
                    dataset = tmpset
                    ## Grab the conversion factors while we find this dataset for the first time
                    if cgsunits != None or physunits != None:
                        cgsconvfactor = fin[global_var_name].attrs['CGSConversionFactor']
                        aexpscaleexponent = fin[global_var_name].attrs['aexp-scale-exponent']
                        hscaleexponent = fin[global_var_name].attrs['h-scale-exponent']
                else:
                    dataset = pd.concat([dataset,tmpset],ignore_index=True)

    try:
        dataset
    except NameError:
        if var_name == 'Mass' and inputtype == 'Snapshot':
            if ptype == 1 or ptype == 2 or ptype == 3:
            ## Have to hardwire in the case for collisionless particles, which all have the same mass, 
            ## as given by the masstable, and a number of particles given by Header numppart_total
            ## Assume that the standard Gadget units are used.
                if cgsunits != None or physunits != None:
                    cgsconvfactor=1.989e43
                    aexpscaleexponent=0.0
                    hscaleexponent=-1.0
                dataset = pd.DataFrame(numpy.ones(numpart_total[ptype])*masstable[ptype], dtype='f8')
            else:
                print("Are you reading in a strange particle type mass? ")
                return -4
        else:
            print("There is no dataset... something went wrong reading this! ")
            return -5


#########################################################
##
##  Horribe Bug in the Code which has the FOF CoM wrong scaling
##
#########################################################
        if cgsunits != None or physunits != None:
            if global_var_name == 'FOF/CenterOfMassVelocity/' and abs(aexpscaleexponent-0.5) < 0.01:
                print("OWLS SubFind had a bug which stated FoF CenterOfMassVelocity had aexp of 0.5, in reality it was -1.0 ") 
                aexpscaleexponent = -1.0

#########################################################
##
##  Convert the array to cgs units, including the factors 
##  for 'a' and 'h' dependencies, else output in code units
##
#########################################################
    if cgsunits != None:
        if leaveh != None:
            convfactor = aexp**aexpscaleexponent * cgsconvfactor
        else:
            convfactor = aexp**aexpscaleexponent * hubble**hscaleexponent * cgsconvfactor

    if physunits != None:
        if leaveh != None:
            convfactor = aexp**aexpscaleexponent
        else:
            convfactor = aexp**aexpscaleexponent * hubble**hscaleexponent 

        if  var_name == 'Mass' or var_name == 'MassType' or var_name == 'InitialMass' or var_name == 'Halo_M_TopHat200' or \
            var_name == 'Halo_M_Crit200' or var_name == 'Halo_M_Crit500' or var_name == 'Halo_M_Crit2500' or \
            var_name == 'Halo_M_Mean200' or var_name == 'Halo_M_Mean500' or var_name == 'Halo_M_Mean2500':
            convfactor *= cgsconvfactor/solar_mass
        if  var_name == 'StarFormationRate':
            convfactor *= cgsconvfactor/(solar_mass/sec_per_year)
        if  var_name == 'StellarAge':
            convfactor *= cgsconvfactor/sec_per_year
        if  var_name == 'MaximumEntropy':
            convfactor *= cgsconvfactor*protonmass**gamma / boltzmann

    if cgsunits != None or physunits != None:
        dataset *= convfactor
        if silent == None:
            if cgsunits != None:
                print("Converting units to CGS")
            if physunits != None:
                print("Converting units to Proper Normalised Units ")
            print("Scale Factor dependency "+str(aexpscaleexponent))
            if leaveh != None:
                print("User asked to leave little h unchanged! ")
            else:
                print("little h dependency "+str(hscaleexponent))

            print("Overall conversion factor "+str(convfactor))
    else:
        if silent == None:
            print("No conversion - still in Code units")

#########################################################
##
##  Convert the 1D array (nadded*3) to 2D matrix (nadded,3)
##  Except for particle type arrays which are (nadded,6)
##
#########################################################
    ncols = 1
    if  var_name == 'Coordinates' or var_name == 'Velocity' or var_name == 'CenterOfMass' \
        or var_name == 'CenterOfMassVelocity' or var_name == 'Position' or var_name == 'SubSpin' \
        or var_name == 'GasSpin' or var_name == 'StarSpin' or var_name == 'SFSpin' \
        or var_name == 'NSFSpin':
        if inputtype != 'Snapshot':
            ncols = 3
    if  var_name == 'LengthType' or var_name == 'MassType' or var_name == 'OffsetType' \
        or var_name == 'SubHalfMassProj' or var_name == 'SubHalfMass':
        if inputtype != 'Snapshot':
            ncols = 6

    if ncols > 1:
        nadded = len(dataset.index) / ncols
        dataset = pd.DataFrame(dataset.values.reshape((nadded,ncols)))

    if floatunits != None:
        dataset = dataset.astype('f4')## Make the units float32

    if silent == None:
        print("Done ")
    
    ## User wants to change from Dataframe to numpy format
    if nopanda != None:
        dataset = numpy.ascontiguousarray(dataset)

    return dataset
