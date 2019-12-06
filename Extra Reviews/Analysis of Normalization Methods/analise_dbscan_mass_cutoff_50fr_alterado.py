# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import mdtraj as md
from sklearn import cluster, metrics
from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler, MaxAbsScaler

mass_atom = { 'H':1.00800,    # H ! polar H
	 'HC':1.00800,    # H ! N-ter H
	 'HA':1.00800,    # H ! nonpolar H
	 'HT':1.00800,    # H ! TIPS3P WATER HYDROGEN
	 'HP':1.00800,    # H ! aromatic H
	 'HB':1.00800,    # H ! backbone H
	 'HR1':1.00800,   # H ! his he1, (+) his HG,HD2
	 'HR2':1.00800,   # H ! (+) his HE1
	 'HR3':1.00800,   # H ! neutral his HG, HD2
	 'HS':1.00800,    # H ! thiol hydrogen
	 'HE1':1.00800,   # H ! for alkene; RHC=CR
	 'HE2':1.00800,   # H ! for alkene; H2C=CR
	 'C':12.01100,    # C ! carbonyl C, peptide backbone
	 'CA':12.01100,   # C ! aromatic C
	 'CT1':12.01100,  # C ! aliphatic sp3 C for CH
	 'CT2':12.01100,  # C ! aliphatic sp3 C for CH2
	 'CT3':12.01100,  # C ! aliphatic sp3 C for CH3
	 'CPH1':12.01100, # C ! his CG and CD2 carbons
	 'CPH2':12.01100, # C ! his CE1 carbon
	 'CPT':12.01100,  # C ! trp C between rings
	 'CY':12.01100,   # C ! TRP C in pyrrole ring
	 'CP1':12.01100,  # C ! tetrahedral C (proline CA)
	 'CP2':12.01100,  # C ! tetrahedral C (proline CB/CG)
	 'CP3':12.01100,  # C ! tetrahedral C (proline CD)
	 'CC':12.01100,   # C ! carbonyl C, asn,asp,gln,glu,cter,ct2
	 'CD':12.01100,   # C ! carbonyl C, pres aspp,glup,ct1
	 'CPA':12.01100,  # C ! heme alpha-C
	 'CPB':12.01100,  # C ! heme beta-C
	 'CPM':12.01100,  # C ! heme meso-C
	 'CM':12.01100,   # C ! heme CO carbon
	 'CS':12.01100,   # C ! thiolate carbon
	 'CE1':12.01100,  # C ! for alkene; RHC=CR
	 'CE2':12.01100,  # C ! for alkene; H2C=CR
	 'N':14.00700,    # N ! proline N
	 'NR1':14.00700,  # N ! neutral his protonated ring nitrogen
	 'NR2':14.00700,  # N ! neutral his unprotonated ring nitrogen
	 'NR3':14.00700,  # N ! charged his ring nitrogen
	 'NH1':14.00700,  # N ! peptide nitrogen
	 'NH2':14.00700,  # N ! amide nitrogen
	 'NH3':14.00700,  # N ! ammonium nitrogen
	 'NC2':14.00700,  # N ! guanidinium nitroogen
	 'NY':14.00700,   # N ! TRP N in pyrrole ring
	 'NP':14.00700,   # N ! Proline ring NH2+ (N-terminal)
	 'NPH':14.00700,  # N ! heme pyrrole N
	 'O':15.99900,    # O ! carbonyl oxygen
	 'OB':15.99900,   # O ! carbonyl oxygen in acetic acid
	 'OC':15.99900,   # O ! carboxylate oxygen
	 'OH1':15.99900,  # O ! hydroxyl oxygen
	 'OS':15.99940,   # O ! ester oxygen
	 'OT':15.99940,   # O ! TIPS3P WATER OXYGEN
	 'OM':15.99900,   # O ! heme CO/O2 oxygen
	 'S':32.06000,    # S ! sulphur
	 'SM':32.06000,   # S ! sulfur C-S-S-C type
	 'SS':32.06000,   # S ! thiolate sulfur
	 'HE':4.00260,    # HE ! helium
	 'NE':20.17970,   # NE ! neon
	 'CAL':40.08000,  # CA ! calcium 2+
	 'ZN':65.37000,   # ZN ! zinc (II) cation
	 'FE':55.84700,   # Fe ! heme iron 56
	 'P':30.974,      # P ! ruan que colocou
}


CLUSTER_PER_FRAME = []
FRAME = 0

#Data_analysis = dict()
#Data_analysis['frame'] = []
#Data_analysis['k_cluster'] = []
#Data_analysis['outliers'] = []
#Data_analysis['Silh_coef'] = []
#Data_analysis['clusters'] = []
#Data_analysis['quantidades'] = []

pdb_loc = "./../../../../imaged"
trj_loc = "./../../../../"
res_loc = "graf_disp_ct_distmin0.14_MaxAbsScaler"
val_eps = 0.14

dm_names = open(trj_loc+'traj_files_names.txt','r')

for dm_file_name in dm_names:
    
    print("\n ###########  Lendo arquivo " + dm_file_name.split('.')[0] + "  ###########")
    
    for chunk in md.iterload(trj_loc+dm_file_name.split('.')[0]+'.dcd', chunk=10, top=pdb_loc+'.pdb'):
        
        print('\n FRAME: ' + str(FRAME))
        print(chunk)
	
        if FRAME == 102:
            print("Nesse deu algo errado")

        selection = chunk[0].atom_slice(chunk[0].topology.select("resname MOL"))
        selection.save_pdb("tmp_frame_"+str(val_eps)+"_"+res_loc.split("_")[-1]+".pdb")
        
        D = []
        
        with open("tmp_frame_"+str(val_eps)+"_"+res_loc.split("_")[-1]+".pdb", 'r') as f:
            data = f.readlines()
             
            for line in data:
                #print(line.split())
                if (line.split()[0] == "ATOM"):
                    d = dict()
                    d['x'] = float(line.split()[6])
                    d['y'] = float(line.split()[7])
                    d['z'] = float(line.split()[8])
                    d['group'] = str(line.split()[5])
                    #try:
                    d['atom'] = str(line.split()[11])
                    #except:
                    #    d['group'] = str(line.split()[11])[:-1]
                    #    d['atom'] = str(line.split()[11])[-1]
                        
                    D.append(d)
        
        D = pd.DataFrame(D)
        grouped = D.groupby("group")
        
        
        center_masses = []
        
        for group in D.group.unique():
            aux = grouped.get_group(group)
            sum_x = sum_y = sum_z = sum_masses = 0.0
            for i in range(len(aux)):
                atom, _, x, y, z = aux.iloc[i]
                sum_x = sum_x + x
                sum_y = sum_y + y
                sum_z = sum_z + z
                sum_masses = sum_masses + mass_atom[atom]
            center_masses.append([sum_x/sum_masses, sum_y/sum_masses, sum_z/sum_masses])
        
        #center_masses = np.asarray(center_masses)
        
        #D.to_csv('molecules_micele.csv', index=False)
        
        #X = pd.read_csv('molecules_micele.csv')
        #aux = pd.read_csv('molecules_micele.csv')
        #X = aux[['x','y','z']]
        
        X = np.array(center_masses)
        
        # normalize the data in the space of 0 to 1
        #X = normalize(X, axis=0, norm='max')
        #scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        #scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
        scaler = MaxAbsScaler(copy=True)
        #X = 

        X = scaler.fit_transform(X)
        
        db = cluster.DBSCAN(eps=val_eps, min_samples=2, algorithm='kd_tree', metric='euclidean').fit(X)
        #db = cluster.DBSCAN(eps=4, min_samples=2, algorithm='kd_tree', metric='euclidean').fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        #SAVE DATA FROM FRAME
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        try:
            silh = metrics.silhouette_score(X, labels)
            print("Silhouette Coefficient: %0.3f" % silh)
        except:
            silh = -2
        #print("Labels:")
        #print(labels)
        unique_elements, counts_elements = np.unique(labels, return_counts=True)
        print("Clusters:")
        print(unique_elements)
        print("Quantidades:")
        print(counts_elements)
        
        size_of_clusters = [0, 0, 0, 0, 0]        
        for tc in counts_elements[1:]:
            if tc>=2 and tc<=4:     size_of_clusters[0] += 1
            if tc>=5 and tc<=8:     size_of_clusters[1] += 1
            if tc>=9 and tc<=15:    size_of_clusters[2] += 1
            if tc>=16 and tc<=31:   size_of_clusters[3] += 1
            if tc>=32:              size_of_clusters[4] += 1
    
        CLUSTER_PER_FRAME.append([FRAME, n_clusters_, n_noise_, silh, unique_elements, counts_elements,
                                 size_of_clusters[0], size_of_clusters[1], size_of_clusters[2], 
                                 size_of_clusters[3], size_of_clusters[4]])
        
        FRAME += 1
        
         #############################################################################
         #Plot result
        
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        # colors used after on the plot
        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(list(pd.DataFrame(X)[0]), list(pd.DataFrame(X)[1]), list(pd.DataFrame(X)[2]), 
                   c=colors[labels].tolist(), marker='o', s=100)
        
        ax.set_xlabel('X coor')
        ax.set_ylabel('Y coor')
        ax.set_zlabel('Z coor')
        
        plt.savefig(res_loc+"/plot_"+str(FRAME)+".png")
        plt.close()
        #plt.show()

Data_analysis = pd.DataFrame(CLUSTER_PER_FRAME)
Data_analysis.columns = ['frame', 'k_cluster', 'outliers', 'silh_coef', 'clusters_label', 'quantidades', 
                         'small_clst', 'intermediate_clst', 'large_clst', 'small_mic', 'micelles']
Data_analysis.to_csv(res_loc+"/data_analysis.csv")
