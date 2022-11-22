# Alias the disks for usage
# !subst N: E:\Network_Data_Sync
# !subst S: E:\Network_Data_Sync
# !subst O: "E:\OneDrive - Washington University in St. Louis"
import os
# os.system(r'subst N: E:\Network_Data_Sync') # do this if at home.
os.system(r'subst S: E:\Network_Data_Sync')
os.system(r'subst O: "E:\OneDrive - Washington University in St. Louis"')
os.system(r"subst N: \\storage1.ris.wustl.edu\crponce\Active")