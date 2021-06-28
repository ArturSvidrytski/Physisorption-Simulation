# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:38:29 2019

@author: Artur Svidrytski
"""


import os
os.chdir( 'G:/Physisorption-Simulation'  )
import sample.helpers as hp
from os import path
import numpy as np
import subprocess
import pandas as pd

#%%

# Hyperparameters
data_dir = path.join( os.getcwd(), 'data')
object_dirname = 'SamplePore' 
bulk_width = 10
list_tstar = [0.87, 0.87]
list_e_wf = [ 1.3, 2.0 ]

list_start_act = [0.05, 0.05]
list_stop_act = [1.00, 1.00]
list_step_act = [0.05, 0.05]
list_subpoints = ['no', 'ads'] # 'no'/'ads'/'des'/'adsdes'
list_substart_act = [0.8, 0.8]   
list_substop_act = [0.84,  0.84]       
list_substep_act = [0.002, 0.002]
list_scan_state = ['des', 'no']

##

folder_path = r"d:\1_Work\PROJECTS\2017\MFT\MFT_RUN\MFT_RUN"
mft_exe_dirpath_src = r"d:\1_Work\PROJECTS\2017\MFT\MFT_RUN\x64\Debug"
mft_exe_fname = "MFT_RUN.exe"
mft_exe_fpath_src = path.join( mft_exe_dirpath_src, mft_exe_fname )
mft_exe_fpath_dst = path.join( data_dir, mft_exe_fname )
mft_exe_fpath = mft_exe_fpath_dst
##

object_dirpath = path.join( data_dir,
                            object_dirname )    

imgs_dirpath = path.join(object_dirpath, "Images")

map_fname = "map.mftin"
map_fpath = path.join( data_dir, 
                      object_dirname, 
                      map_fname )

map_size_fname = "map_size.mftin"
map_size_fpath = path.join( data_dir, 
                      object_dirname, 
                      map_size_fname )

lampres_fname = "lamrel_presrel_values.siminfo"
lamrel_fname = "lambda_values.mftin"
sim_params_fname = "sim_params.mftin"

sim_list_fname = "sim_list.siminfo"
sim_list_fpath = path.join( object_dirpath,
                           sim_list_fname )

out_mft_dirname = "mft_outfiles"
out_mft_dirpath = path.join( data_dir, 
                            object_dirname,
                            out_mft_dirname )

out_mft_fname = "activity_lambda_ads_0850.mftout"
out_mft_fpath = path.join( out_mft_dirpath, out_mft_fname )

res_imgs_dirname = "Results"
res_imgs_dirpath = path.join( data_dir, 
                             object_dirname, 
                             res_imgs_dirname )

hp.write_new_sim_list(sim_list_fpath)

# Prepare structure and write
obj3d = hp.read_tiff_stack( imgs_dirpath )




obj3d_bulk = hp.add_bulk(obj3d, bulk_width)

hp.write_map( obj3d_bulk, map_fpath )
hp.write_map_size(obj3d_bulk.shape, map_size_fpath)

for (cur_tstar, 
     cur_e_wf,     
     cur_start_act,
     cur_stop_act,
     cur_step_act,
     cur_subpoints,
     cur_substart_act,
     cur_substop_act,
     cur_substep_act,
     cur_scan_state) in zip(list_tstar, 
                        list_e_wf,
                        list_start_act,
						list_stop_act,
						list_step_act,
						list_subpoints,
						list_substart_act,
						list_substop_act,
						list_substep_act,
						list_scan_state):
    print( cur_tstar, cur_e_wf)
    hp.add_sim_jobs(object_dirpath,                
            tstar        = cur_tstar,
            e_wf         =   cur_e_wf,
			start_act    =  cur_start_act,
			stop_act     =  cur_stop_act,
			step_act     =  cur_step_act,
			subpoints    =  cur_subpoints,
			substart_act =  cur_substart_act, 
			substop_act  = cur_substop_act,
			substep_act  = cur_substep_act,
			scan_state   =  cur_scan_state) 


hp.build_file_structure(sim_list_fpath)
hp.run_sim_jobs(sim_list_fpath, mft_exe_fpath_src)



#%% Process output and save the result ready to plot
 
df_sim_list = pd.read_csv(sim_list_fpath, sep='\t')

dpoints_dirname = 'density_datapoints'
dpoints_dirpath = path.join( data_dir,
                              object_dirname,
                              dpoints_dirname )
                              
if not os.path.exists( dpoints_dirpath ):
            os.makedirs( dpoints_dirpath)

densdpoints_progress_fname = 'density_dpoints_progress.info'
densdpoints_progress_fpath = os.path.join( data_dir,
                              object_dirname,
                              densdpoints_progress_fname )

if os.path.isfile(densdpoints_progress_fpath) != True:
    df_densdpoints_progress = pd.DataFrame(columns=[
                      'sim_num',
                      'sim_object',
                      'sim_name',
                      'tstar',
                      'e_wf',
                      'csv_path',
                      'sim_path'])        
    df_densdpoints_progress.to_csv(densdpoints_progress_fpath,
                                    sep='\t',
                                    index=False)

dic_mft_sim_res = { 'sim_num':[], 
                   'sim_object':[],
                   'sim_name':[], 
                   'tstar':[], 
                   'e_wf':[], 
                   'data':[],
                   'sim_path':[] }

df_densdpoints_progress = pd.read_csv(densdpoints_progress_fpath, sep='\t')
for i in df_sim_list.index:
#for i in range(135, 145):
    cur_sim_num = df_sim_list.at[i, 'sim_number']
    if cur_sim_num not in df_densdpoints_progress['sim_num'].values:
        print('_________ Simulation', i, '_________')        
        cur_sim_path = df_sim_list.at[i, 'sim_path']
        cur_sim_object = os.path.basename( os.path.dirname( cur_sim_path ) )
        cur_sim_name = os.path.basename( cur_sim_path )
        mft_size_fpath = path.join(cur_sim_path, '..', 'map_size.mftin')
        sim_outfiles_dirpath = path.join(cur_sim_path, 'sim_outfiles_bin')
        cur_tstar = df_sim_list.at[i, 'Tstar']
        cur_e_wf = df_sim_list.at[i, 'e_wf']    
        df_results = hp.gather_dens_vals_extended(sim_outfiles_dirpath, 
                                               mft_size_fpath, bulk_width, cur_tstar)    
        dic_mft_sim_res['sim_num'].append(cur_sim_num)
        dic_mft_sim_res['sim_object'].append(cur_sim_object)
        dic_mft_sim_res['sim_name'].append(cur_sim_name)
        dic_mft_sim_res['tstar'].append(cur_tstar)
        dic_mft_sim_res['e_wf'].append(cur_e_wf)
        dic_mft_sim_res['data'].append(df_results)
        dic_mft_sim_res['sim_path'].append(cur_sim_path)


# sort points
for i in range(len(dic_mft_sim_res['data'])):
    tmp = dic_mft_sim_res['data'][i]
    tmp = pd.concat([
        tmp[tmp.state=='ads'],
        tmp[tmp.state=='des'].sort_values('lambda', ascending=False)        
        ], ignore_index = True)
    dic_mft_sim_res['data'][i] = tmp


for idx in range( len(dic_mft_sim_res['data']) ):
    #cur_pres_col_name = \
    #    f"pres (T = {round(dic_mft_sim_res['tstar'][idx], 2)}; e_wf = {round(dic_mft_sim_res['e_wf'][idx], 2)})"
    #cur_col_name = \
    #    f"T = {round(dic_mft_sim_res['tstar'][idx], 2)}; e_wf = {round(dic_mft_sim_res['e_wf'][idx], 2)}"
    #df_origin_plot[cur_pres_col_name] = dic_mft_sim_res['data'][idx]['relpres'].values
    #df_origin_plot[cur_col_name] = \
    #    dic_mft_sim_res['data'][idx]['density'].values
    df_densdpoints_progress = pd.read_csv(densdpoints_progress_fpath, sep='\t')
    
    df_origin_plot = pd.DataFrame(
        {'state':dic_mft_sim_res['data'][idx]['state'].values,
         'pressure':dic_mft_sim_res['data'][idx]['relpres'].values,
         'density':dic_mft_sim_res['data'][idx]['density'].values})
    
    cur_data_table_fname = f"{dic_mft_sim_res['sim_name'][idx]}.csv"
    dpoints_fpath = os.path.join( dpoints_dirpath,
                                      cur_data_table_fname )    
    df_origin_plot.to_csv(dpoints_fpath, sep='\t', index_label='row_number')
    
    df_densdpoints_progress = df_densdpoints_progress.append({
                        'sim_num': dic_mft_sim_res['sim_num'][idx],
                        'sim_object': dic_mft_sim_res['sim_object'][idx],
                        'sim_name': dic_mft_sim_res['sim_name'][idx],
                        'tstar': dic_mft_sim_res['tstar'][idx],
                        'e_wf': dic_mft_sim_res['e_wf'][idx],
                        'csv_path': dpoints_fpath,
                        'sim_path': dic_mft_sim_res['sim_path'][idx]}, ignore_index=True)
    
    df_densdpoints_progress.to_csv(densdpoints_progress_fpath, sep='\t', index=False)
    df_densdpoints_progress = pd.read_csv(densdpoints_progress_fpath, sep='\t')


