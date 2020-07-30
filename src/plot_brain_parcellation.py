from nilearn import plotting
import metric_util as mt
import mri_draw_utils as mrd
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def main():
    ward_parcellation_img_path = "/apps/git/python/tt-str/src/results/parcel_ward_158.nii"
    geometric_parcellation_img_path = "/apps/git/python/tt-str/src/results/parcel_geometric_158.nii"
    kmeans_parcellation_img_path = "/apps/git/python/tt-str/src/results/parcel_kmeans_158.nii"
    spectral_parcellation_img_path = "/apps/git/python/tt-str/src/results/parcel_spectral_158.nii"
    
    ward_parcel_img = mt.read_image_abs_path(ward_parcellation_img_path)
    geometric_parcel_img = mt.read_image_abs_path(geometric_parcellation_img_path)
    kmeans_parcel_img = mt.read_image_abs_path(kmeans_parcellation_img_path)
    spectral_parcel_img = mt.read_image_abs_path(spectral_parcellation_img_path)
    
    ward_parcel_out = "/apps/img/parcellation/ward_parcellation.pdf"
    geometric_parcel_out = "/apps/img/parcellation/geometric_parcel_out.pdf"
    kmeans_parcel_out = "/apps/img/parcellation/kmeans_parcellation.pdf"
    spectral_parcel_out = "/apps/img/parcellation/spectral_parcellation.pdf"
    
    plotting.plot_roi(geometric_parcel_img,title='', annotate=False, display_mode='z', cut_coords=1, bg_img=None,
                       output_file=geometric_parcel_out)
    
    plotting.plot_roi(kmeans_parcel_img,title='', annotate=False, display_mode='z', cut_coords=1, bg_img=None,
                       output_file=kmeans_parcel_out)
    
    plotting.plot_roi(spectral_parcel_img,title='', annotate=False, display_mode='z', cut_coords=1, bg_img=None,
                       output_file=spectral_parcel_out)
    
    print "Completed"
    pass

def save_data():
    ll_cv_ward = pickle.load( open("/apps/git/python/tt-str/src/results/ll_cv_ward.pck", "rb" ) )
    ll_cv_k_means = pickle.load( open("/apps/git/python/tt-str/src/results/ll_cv_kmeans.pck", "rb" ) )
    ll_cv_geometric = pickle.load( open("/apps/git/python/tt-str/src/results/ll_cv_geometric.pck", "rb" ) )
    
    #ward
    ll_cv_ward_path = "/apps/img/parcellation/results/ll_cv_ward.csv"
    dataset_id = ll_cv_ward
    df = pd.DataFrame()
    for key, value in ll_cv_ward.items():
        print key, value
        df = df.append({'k': key, 'll_cv':value}, ignore_index=True)
    print df
    
    df.to_csv(ll_cv_ward_path)
    
    #kmeans
    ll_cv_kmeans_path = "/apps/img/parcellation/results/ll_cv_kmeans.csv"
    dataset_id = ll_cv_k_means
    df = pd.DataFrame()
    for key, value in ll_cv_k_means.items():
        print key, value
        df = df.append({'k': key, 'll_cv':value}, ignore_index=True)
    print df
    
    df.to_csv(ll_cv_kmeans_path)
    
    #geometric
    ll_cv_geometric_path = "/apps/img/parcellation/results/ll_cv_geometric.csv"
    dataset_id = ll_cv_geometric
    df = pd.DataFrame()
    for key, value in ll_cv_geometric.items():
        print key, value
        df = df.append({'k': key, 'll_cv':value}, ignore_index=True)
    print df
    
    df.to_csv(ll_cv_geometric_path)
    
    # log-likelihood
    ll_ward = pickle.load( open("/apps/git/python/tt-str/src/results/all_ll_ward.pck", "rb" ))
    ll_kmeans = pickle.load( open("/apps/git/python/tt-str/src/results/all_ll_kmeans.pck", "rb" ))
    ll_geometric = pickle.load( open("/apps/git/python/tt-str/src/results/all_ll_geometric.pck", "rb"))
    
    #ward
    ll_ward_path = "/apps/img/parcellation/results/ll_ward.csv"
    dataset_id = ll_cv_ward
    df = pd.DataFrame()
    for key, value in ll_ward.items():
        print key, value
        df = df.append({'k': key, 'all_ll':value}, ignore_index=True)
    print df
    
    df.to_csv(ll_ward_path)
    
    #kmeans
    ll_kmeans_path = "/apps/img/parcellation/results/ll_kmeans.csv"
    dataset_id = ll_cv_ward
    df = pd.DataFrame()
    for key, value in ll_kmeans.items():
        print key, value
        df = df.append({'k': key, 'all_ll':value}, ignore_index=True)
    print df
    
    df.to_csv(ll_kmeans_path)
    
    #geometric
    ll_geometic_path = "/apps/img/parcellation/results/ll_geometric.csv"
    dataset_id = ll_cv_ward
    df = pd.DataFrame()
    for key, value in ll_geometric.items():
        print key, value
        df = df.append({'k': key, 'all_ll':value}, ignore_index=True)
    print df
    
    df.to_csv(ll_geometic_path)
    
    ari_score_ward = pickle.load( open("/apps/git/python/tt-str/src/results/ari_score_ward.pck", "rb" ))
    ami_score_ward = pickle.load( open("/apps/git/python/tt-str/src/results/ami_score_ward.pck", "rb" ))
    vm_score_ward = pickle.load( open("/apps/git/python/tt-str/src/results/vm_score_ward.pck", "rb" ))
    bic_score_ward = pickle.load( open("/apps/git/python/tt-str/src/results/all_bic_ward.pck", "rb" ))
    
    #ward
    #ari
    ari_ward_path = "/apps/img/parcellation/results/ari_ward.csv"
    df = pd.DataFrame()
    for key, value in ari_score_ward.items():
        print key, value
        df = df.append({'k': key, 'ari':value}, ignore_index=True)
    print df
    
    df.to_csv(ari_ward_path)
    
    #ami
    ami_ward_path = "/apps/img/parcellation/results/ami_ward.csv"
    df = pd.DataFrame()
    for key, value in ami_score_ward.items():
        print key, value
        df = df.append({'k': key, 'ami':value}, ignore_index=True)
    print df
    
    df.to_csv(ami_ward_path)
    
    #vm_score
    vm_score_ward_path = "/apps/img/parcellation/results/vm_score_ward.csv"
    df = pd.DataFrame()
    for key, value in vm_score_ward.items():
        print key, value
        df = df.append({'k': key, 'vm':value}, ignore_index=True)
    print df
    
    df.to_csv(vm_score_ward_path)
    
    #bic_score
    bic_score_ward_path = "/apps/img/parcellation/results/bic_ward.csv"
    df = pd.DataFrame()
    for key, value in bic_score_ward.items():
        print key, value
        df = df.append({'k': key, 'vm':value}, ignore_index=True)
    print df
    
    df.to_csv(bic_score_ward_path)
    
    #kmeans
    ari_score_kmeans = pickle.load( open("/apps/git/python/tt-str/src/results/ari_score_kmeans.pck", "rb" ))
    ami_score_kmeans = pickle.load( open("/apps/git/python/tt-str/src/results/ami_score_kmeans.pck", "rb" ))
    vm_score_kmeans = pickle.load( open("/apps/git/python/tt-str/src/results/vm_score_kmeans.pck", "rb" ))
    bic_score_kmeans = pickle.load( open("/apps/git/python/tt-str/src/results/all_bic_kmeans.pck", "rb" ))
    
    #ari
    ari_kmeans_path = "/apps/img/parcellation/results/ari_kmeans.csv"
    df = pd.DataFrame()
    for key, value in ari_score_kmeans.items():
        print key, value
        df = df.append({'k': key, 'ari':value}, ignore_index=True)
    print df
    
    df.to_csv(ari_kmeans_path)
    
    #ami
    ami_kmeans_path = "/apps/img/parcellation/results/ami_kmeans.csv"
    df = pd.DataFrame()
    for key, value in ami_score_kmeans.items():
        print key, value
        df = df.append({'k': key, 'ami':value}, ignore_index=True)
    print df
    
    df.to_csv(ami_kmeans_path)
    
    #vm_score
    vm_score_kmeans_path = "/apps/img/parcellation/results/vm_score_kmeans.csv"
    df = pd.DataFrame()
    for key, value in vm_score_kmeans.items():
        print key, value
        df = df.append({'k': key, 'vm':value}, ignore_index=True)
    print df
    
    df.to_csv(vm_score_kmeans_path)
    
    #bic_score
    bic_score_kmeans_path = "/apps/img/parcellation/results/bic_kmeans.csv"
    df = pd.DataFrame()
    for key, value in bic_score_kmeans.items():
        print key, value
        df = df.append({'k': key, 'vm':value}, ignore_index=True)
    print df
    
    df.to_csv(bic_score_kmeans_path)
    
    #geometric
    ari_score_geometric = pickle.load( open("/apps/git/python/tt-str/src/results/ari_score_geometric.pck", "rb" ))
    ami_score_geometric = pickle.load( open("/apps/git/python/tt-str/src/results/ami_score_geometric.pck", "rb" ))
    vm_score_geometric = pickle.load( open("/apps/git/python/tt-str/src/results/vm_score_geometric.pck", "rb" ))
    bic_score_geometric = pickle.load( open("/apps/git/python/tt-str/src/results/all_bic_geometric.pck", "rb" ))
    
    #ari
    ari_geometric_path = "/apps/img/parcellation/results/ari_geometric.csv"
    df = pd.DataFrame()
    for key, value in ari_score_geometric.items():
        print key, value
        df = df.append({'k': key, 'ari':value}, ignore_index=True)
    print df
    
    df.to_csv(ari_geometric_path)
    
    #ami
    ami_geometric_path = "/apps/img/parcellation/results/ami_geometric.csv"
    df = pd.DataFrame()
    for key, value in ami_score_geometric.items():
        print key, value
        df = df.append({'k': key, 'ami':value}, ignore_index=True)
    print df
    
    df.to_csv(ami_geometric_path)
    
    #vm_score
    vm_score_geometric_path = "/apps/img/parcellation/results/vm_score_geometric.csv"
    df = pd.DataFrame()
    for key, value in vm_score_geometric.items():
        print key, value
        df = df.append({'k': key, 'vm':value}, ignore_index=True)
    print df
    
    df.to_csv(vm_score_geometric_path)
    
    #bic_score
    bic_score_geometric_path = "/apps/img/parcellation/results/bic_geometric.csv"
    df = pd.DataFrame()
    for key, value in bic_score_geometric.items():
        print key, value
        df = df.append({'k': key, 'vm':value}, ignore_index=True)
    print df
    
    df.to_csv(bic_score_geometric_path)
   
   
if __name__ == "__main__":
    pass
    #main()
    save_data()