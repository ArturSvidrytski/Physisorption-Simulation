from os import listdir, path, unlink
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import cm as colmap
from matplotlib import colors
import gc
import subprocess
import  time
import pandas as pd
import string
import csv
import glob
import skimage.measure
import skimage.morphology
import scipy.ndimage
import struct


def read_input_mft( file ):
    with open(file, "rb") as f:
        read_byte_data = f.read()
    read_tuple_data = \
        struct.unpack(str(len(read_byte_data)) +'B', read_byte_data )
    input_data = np.asarray(read_tuple_data)
    return input_data

def read_output_mft( file ):
    with open(file, "rb") as f:
        read_byte_data = f.read()
    read_tuple_data = \
        struct.unpack(str(int(len(read_byte_data)/4)) +'f', read_byte_data )
    output_data = np.asarray(read_tuple_data)
    return output_data

def read_tiff_stack( imgs_path ): 
    imgs = [f for f in listdir(imgs_path) 
    if path.isfile(path.join(imgs_path, f))]
    cur_image = io.imread( path.join(imgs_path, imgs[0]) )
    obj3d = np.zeros( (cur_image.shape[0],
                       cur_image.shape[1], 
                       len(imgs)), dtype = np.uint8 )
    for imgidx in range(len(imgs)):
        status = 100*(imgidx+1)/len(imgs)
        if status % np.trunc(status) == 0:
            print( int(status), '%')    
        cur_image = io.imread( path.join(imgs_path, imgs[imgidx]) )
        cur_image[cur_image== np.uint8(255)] = np.uint8(1)
        obj3d[:,:,imgidx] = cur_image    
    return obj3d

def add_bulk( obj3d, bulk_width ):
    obj3d_bulk = np.full((obj3d.shape[0]+2*bulk_width, 
                        obj3d.shape[1]+2*bulk_width, 
                        obj3d.shape[2]+2*bulk_width),
        2,
        dtype = np.uint8 )
    obj3d_bulk[bulk_width:(bulk_width+obj3d.shape[0]),
               bulk_width:(bulk_width+obj3d.shape[1]),
               bulk_width:(bulk_width+obj3d.shape[2])] = np.copy(obj3d)
    return obj3d_bulk

def write_map( obj3d_bulk, input_file_path ):
    with open(input_file_path, "wb") as fid:
        obj3d_bulk.tofile(fid)


def save_obj3D_rgb(obj3d, imgs_dir, cmap='RdBu'):
    norm_colours = colors.Normalize(vmin=-1.0, vmax=1.0)
    
    if cmap=='RdBu':
        cur_cmap = colmap.get_cmap(cmap)
    elif cmap=='custom':
        cur_cmap = get_cmap_custom()
    
    obj3d_normcol = norm_colours(obj3d)
    obj3d_rgb = np.uint8( np.round( cur_cmap(obj3d_normcol)[:,:,:,0:3] * 255 ) )
    del obj3d_normcol; gc.collect()
    
    for k in range(obj3d_rgb.shape[2]):
        # print(k)
        rgb_img = obj3d_rgb[:,:,k,0:3]            
        dest_path = path.join(imgs_dir, 'Img_'+f"{k:04d}"+'.tif' )
        io.imsave(dest_path, rgb_img)
    
    del obj3d_rgb; gc.collect()


def get_cmap_tab10cust(solid_color='black'):
    cmap_tab10 = colmap.get_cmap('tab10')
    tab10_col = cmap_tab10(range(10))
    black = np.array([0/256, 0/256, 0/256, 1])
    white = np.array([256/256, 256/256, 256/256, 1])
    if solid_color=="black":
        solcol = black
    elif solid_color=="white":
        solcol = white    
    colors_custom = np.tile(solcol, (20,1))    
    colors_custom[10:, :] = tab10_col
    cmap_custom = colors.ListedColormap(colors_custom)
    return(cmap_custom)


def get_cmap_custom():    
    solcol = np.array([103/256, 0/256, 31/256, 1])
    white = np.array([256/256, 256/256, 256/256, 1])
    green = np.array([0/256, 165/256, 80/256, 1])
    blue = np.array([67/256, 147/256, 195/256, 1]) 
    colors_custom = np.tile([0.0,0.0,0.0,1.0], (8,1))    
    colors_custom[0:4, :] = solcol
    colors_custom[4:5, :] = white
    colors_custom[5:7, :] = green
    colors_custom[7:8, :] = blue    
    cmap_custom = colors.ListedColormap(colors_custom)
    return(cmap_custom)



def save_bw_stack(obj3d, imgs_dir):    
    obj3d_bin = np.uint8( np.round( obj3d * 255 ) )    
    
    for k in range(obj3d_bin.shape[2]):
        # print(k)
        rgb_img = obj3d_bin[:,:,k]            
        dest_path = path.join(imgs_dir, 'Img_'+f"{k:04d}"+'.tif' )
        io.imsave(dest_path, rgb_img)
    
    del obj3d_bin; gc.collect()




def remove_files_in_dir(dir_path):
    for cur_file in listdir(dir_path):
        cur_file_path = path.join(dir_path, cur_file)
        try:
            if path.isfile(cur_file_path):
                unlink(cur_file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def apply_thresh_fluid(obj3d, threshold_val, blue_intens = 0.6):
    obj3d_out_thresh = np.copy(obj3d)
    # blue colour intensity
    obj3d_out_thresh[obj3d > threshold_val ] = blue_intens
    obj3d_out_thresh[ np.logical_and(obj3d > 0.0,
                                     obj3d <= threshold_val)] = 0.0
    return obj3d_out_thresh

def cut_subdomain(obj3d, vox_to_cut):
    x_len = obj3d.shape[0]
    y_len = obj3d.shape[1]
    z_len = obj3d.shape[2]    
    obj3d_cut = obj3d[vox_to_cut:(x_len-vox_to_cut),
                      vox_to_cut:(y_len-vox_to_cut),
                      vox_to_cut:(z_len-vox_to_cut)]
    return obj3d_cut

def calc_density(obj3d):    
    liq_total = np.sum( obj3d[ obj3d>-1.0] )
    void_total = np.sum( obj3d > -1.0 )
    density_total = float(liq_total) / float(void_total)
    return density_total


def calc_lmda_sat(Treduced):
    return np.exp(-3.0/Treduced)

def calc_pressure_array(lmda, Treduced):
    rho = lmda
    for i in range(100000):
        rho = lmda/(lmda+np.exp(-6.0*rho/Treduced))
    rho[rho>0.5] = 1.0-rho[rho>0.5]
    p = -Treduced*np.log(1.0-rho)-3.0*rho**2
    return p
    
def calc_pressure_singval(lmda, Treduced):
    rho = lmda
    for i in range(100000):
        rho = lmda/(lmda+np.exp(-6.0*rho/Treduced))        
    if rho>0.5:
        rho = 1.0-rho
    p = -Treduced*np.log(1.0-rho)-3.0*rho**2
    return p

def transform_act_to_presrel(lmda_cur, Treduced):
    lmda_sat = calc_lmda_sat(Treduced)    
    p_sat = calc_pressure_singval(lmda_sat,  Treduced)    
    
    if type(lmda_cur) == np.ndarray:        
        p_cur = calc_pressure_array(lmda_cur*lmda_sat,  Treduced)
    else:        
        p_cur = calc_pressure_singval(lmda_cur*lmda_sat,  Treduced)    
    return p_cur/p_sat

def write_map_size(map_shape, fpath):
    df_map_size = pd.DataFrame({'vertical':map_shape[0],
                                'horizontal':map_shape[1],
                                'longitudinal':map_shape[2]},        
        index=[0])
    df_map_size.to_csv(fpath, sep='\t', index = False)    

def write_lampres_vals(df_lampres, fpath):
    with open(fpath, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerows(df_lampres)
   
def write_lambda_vals(df_lampres, fpath):
    with open(fpath, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')
        writer.writerows(df_lampres)


def create_df_lampres(tstar, start, stop, step, subpoints = 'no', substart = 0, substop = 0, substep = 0 ):
    lamrel_vals_ads = np.round( np.arange(start,
                                      stop+step, 
                                      step, 
                                      dtype=np.float32), decimals=3 )

    lamrel_vals_des = np.round( np.arange(stop-step, 
                                          start-step, 
                                          -step, 
                                          dtype=np.float32), decimals=3 )
    
    
    if subpoints == 'ads' or subpoints == 'adsdes':
        sublamrel_vals_ads = np.round( np.arange(substart,
                                          substop+substep, 
                                          substep,
                                          dtype=np.float32), decimals=3 )
        lamrel_vals_ads = np.concatenate( [ lamrel_vals_ads[ lamrel_vals_ads<substart ], 
                    sublamrel_vals_ads,
                    lamrel_vals_ads[ lamrel_vals_ads>substop ] ] )
    
    if subpoints == 'des' or subpoints == 'adsdes':
        sublamrel_vals_des = np.round( np.arange(substop, 
                                          substart-substep, 
                                          -substep, 
                                          dtype=np.float32), decimals=3 )
        lamrel_vals_des = np.concatenate( [ lamrel_vals_des[ lamrel_vals_des>substop ], 
                    sublamrel_vals_des,
                    lamrel_vals_des[ lamrel_vals_des<substart ] ] )
    
    df_lambda_ads = pd.DataFrame({'state':'ads', 'lambda':lamrel_vals_ads})
    df_lambda_des = pd.DataFrame({'state':'des', 'lambda':lamrel_vals_des})
    
    df_lampres_vals = pd.concat( [df_lambda_ads, df_lambda_des], ignore_index = True )
    
    presrel_vals = transform_act_to_presrel(df_lampres_vals['lambda'].values, tstar)    
    df_lampres_vals['pressure'] = presrel_vals
    return(df_lampres_vals)


def create_df_lampres_scan(scan_state, tstar, start, stop, step, subpoints = 'no', substart = 0, substop = 0, substep = 0 ):
    
    if scan_state == 'ads':
        lamrel_vals_ads = np.round( np.arange(start,
                                      stop+step, 
                                      step, 
                                      dtype=np.float32), decimals=3 )
        lamrel_vals_ads = np.concatenate(([1.0], lamrel_vals_ads), axis=0)     
        df_lambda_ads = pd.DataFrame({'state':'ads', 'lambda':lamrel_vals_ads})
        df_lampres_vals = df_lambda_ads
        
        
        
    elif scan_state == 'des':
        lamrel_vals_des = np.round( np.arange(stop, 
                                              start-step, 
                                              -step, 
                                              dtype=np.float32), decimals=3 )     
        df_lambda_des = pd.DataFrame({'state':'des', 'lambda':lamrel_vals_des})
        df_lampres_vals = df_lambda_des
        
        
        
    if subpoints == 'ads' or subpoints == 'adsdes':
        sublamrel_vals_ads = np.round( np.arange(substart,
                                          substop+substep, 
                                          substep,
                                          dtype=np.float32), decimals=3 )
        lamrel_vals_ads = np.concatenate( [ lamrel_vals_ads[ lamrel_vals_ads<substart ], 
                    sublamrel_vals_ads,
                    lamrel_vals_ads[ lamrel_vals_ads>substop ] ] )
    
    if subpoints == 'des' or subpoints == 'adsdes':
        sublamrel_vals_des = np.round( np.arange(substop, 
                                          substart-substep, 
                                          -substep, 
                                          dtype=np.float32), decimals=3 )
        lamrel_vals_des = np.concatenate( [ lamrel_vals_des[ lamrel_vals_des>substop ], 
                    sublamrel_vals_des,
                    lamrel_vals_des[ lamrel_vals_des<substart ] ] )
    
    
    presrel_vals = transform_act_to_presrel(df_lampres_vals['lambda'].values, tstar)    
    df_lampres_vals['pressure'] = presrel_vals
    return(df_lampres_vals)


def extract_state_lmbda(cur_fname):
    cur_fname_parts = cur_fname.split('.')[0]
    cur_fname_parts = cur_fname_parts.split('_')
    lmda = np.int16(cur_fname_parts[len(cur_fname_parts)-1])/1000
    ads_des_str = cur_fname_parts[len(cur_fname_parts)-2]    
    return( dict(state=ads_des_str, lam_val=lmda) )



def gather_density_vals(dirpath, obj3d_size, bulk_width):
    df_results = pd.DataFrame({"state":[], "lambda":[],  "density":[]})
    counter = 0
    fnames = listdir(dirpath)
    for cur_fname in fnames:
        counter += 1
        print( counter, 'of', len(fnames) )
        cur_fpath = path.join(dirpath, cur_fname)
        dic_state_lmbda = extract_state_lmbda(cur_fname)    
        cur_outdata = read_output_mft(cur_fpath)
        cur_obj3d_out = np.array(cur_outdata, dtype=np.float64 ).reshape( obj3d_size )        
        cur_dens = calc_density( cut_subdomain(cur_obj3d_out, bulk_width ) )    
        df_results = df_results.append({"state": dic_state_lmbda['state'],
                                        "lambda": dic_state_lmbda['lam_val'],
                                        "density": cur_dens}, ignore_index=True)
    return(df_results)





def gather_dens_vals_extended(dest_dirpath, mft_size_fpath, bulk_width, tstar):    
    tup_mft_size = read_mft_size(mft_size_fpath)
    df_results = gather_density_vals( dest_dirpath,
                                     tup_mft_size,
                                     bulk_width ) # BUG!!!!! NO BULK WIDTH IN THIS FUNCTION!!!
                                                        # this +10 must be implemented another way
    lmda_cur = df_results['lambda'].values
    p_rel = transform_act_to_presrel(lmda_cur, tstar)
    df_results.loc[:, 'relpres'] = p_rel
    
    return(df_results)



def read_mft_size(fpath):
    df_mft_size = pd.read_csv(fpath, sep='\t')
    tup_mft_size = tuple( df_mft_size.iloc[0] )
    return(tup_mft_size)

def write_new_sim_list(sim_list_fpath):
    if path.isfile(sim_list_fpath) != True:
        df_sim_list = pd.DataFrame(columns=['sim_number',
                          'is_completed',
                          'Tstar',
                          'e_wf',
                          'e_ff',
                          'start_act',
           					'stop_act',
        					'step_act',
        					'subpoints',
        					'substart_act',
        					'substop_act',
        					'substep_act',
        					'scan_state',
                            'precision',  
                            'max_iter',
                            'eval_iter',  
                            'gpu_device',    
                            'gpu_blocks',    
                            'gpu_threads',
                          'sim_path'])
        df_sim_list.to_csv(sim_list_fpath, sep='\t', index=False)


    
def add_sim_jobs(object_dirpath,                
                tstar = 0.87,
                e_wf = 1.3,
                e_ff = 1.0,
                start_act = 0.05,
				stop_act = 1.0,
				step_act = 0.05,
				subpoints = 'no',
				substart_act =  0.5,
				substop_act = 0.7,
				substep_act = 0.005,
				scan_state = 'no',
				precision = 1e-6,
				max_iter = np.int32(1e4),
				eval_iter = 30,
				gpu_device = 0,
				gpu_blocks = 128,
				gpu_threads = 1024):
    sim_list_fname = "sim_list.siminfo"
    sim_list_fpath = path.join( object_dirpath,
                               sim_list_fname )
    
    is_completed = 'no'        
    df_sim_list = pd.read_csv(sim_list_fpath, sep='\t')
    sim_number = df_sim_list.shape[0]        
    sim_dirname = 'simulation_' +  f"{sim_number:06d}"       
    
    simulation_path = path.join( object_dirpath,
                  sim_dirname )
    
    df_sim_list = df_sim_list.append({
                    'sim_number': sim_number,
                    'is_completed': is_completed,
                    'Tstar': tstar,
                    'e_wf': e_wf,
                    'e_ff': e_ff,
                    'start_act': start_act,     
					'stop_act': stop_act,     
					'step_act': step_act,     
					'subpoints': subpoints,    
					'substart_act': substart_act, 
					'substop_act': substop_act,
					'substep_act': substep_act,
					'scan_state': scan_state,   
                    'precision': precision,  
                    'max_iter': max_iter,
                    'eval_iter': eval_iter,  
                    'gpu_device': gpu_device,    
                    'gpu_blocks': gpu_blocks,    
                    'gpu_threads': gpu_threads,
                    'sim_path': simulation_path}, ignore_index=True)
    df_sim_list.to_csv(sim_list_fpath, sep='\t', index=False)        
        
    
    
def create_sim_params(sim_dirpath,
                      dic_cur_sim_params):
    sim_params_fname = "sim_params.mftin"
    sim_params_fpath = path.join( sim_dirpath,
                      sim_params_fname )
    
    df_sim_params = pd.DataFrame({
         'Tstar':float(dic_cur_sim_params.get('Tstar')),
         'e_wf': float(dic_cur_sim_params.get('e_wf')),
         'e_ff': float(dic_cur_sim_params.get('e_ff')),
         'precision': dic_cur_sim_params.get('precision'),
         'max_iter': np.int32(dic_cur_sim_params.get('max_iter')),
         'eval_iter': dic_cur_sim_params.get('eval_iter'),
         'gpu_device': dic_cur_sim_params.get('gpu_device'),
         'gpu_blocks': dic_cur_sim_params.get('gpu_blocks'),
         'gpu_threads': dic_cur_sim_params.get('gpu_threads')}, 
                                 index = [0])
    
    df_sim_params.to_csv(sim_params_fpath,
                         sep='\t',
                         index = False)   
    



def create_lampres(sim_dirpath, df_lampres_vals):    
    lampres_fname = "lamrel_presrel_values.siminfo"
    lampres_fpath = path.join( sim_dirpath,
                      lampres_fname )
    df_lampres_vals.to_csv(lampres_fpath, sep='\t', index_label = 'row_number')



def create_lamrel(sim_dirpath, df_lampres_vals):
    lamrel_fname = "lambda_values.mftin"
    lamrel_fpath = path.join( sim_dirpath,
                      lamrel_fname )
    df_lampres_vals[['state', 'lambda']].to_csv(lamrel_fpath,
                   sep='\t',
                   header = False,
                   index = False)    





