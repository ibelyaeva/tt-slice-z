import cottoncandy as cc
cci = cc.get_interface(backend='gdrive')

#cci.backend_interface.ls(directory='/UMBC/research/mri_paper_completion/results')
cci.backend_interface.cd(directory='/UMBC/research')

#cci.backend_interface.drive.pwd()

#cci.backend_interface.ls(directory='1DXjM4vRdzUnNtEl-6Hdt80EPIU2svO-j')
#cci.backend_interface.pwd(directory, make_if_not_exist, isID)
exists = cci.backend_interface.check_ID_exists('1DXjM4vRdzUnNtEl-6Hdt80EPIU2svO-j')

print ("Folder exists ? " + str(exists))

#cci.

#cci.glob('/UMBC/research/mri_paper_completion/results')

#1vzk4KT-ep86XS6zYZmxL_l1AXkMbzL4V
#https://drive.google.com/open?id=1qQw9VDuOWyBYJL9RJyIfqiwMSj5gvZ57