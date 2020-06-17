import blitzlib as blz
from mstrio import microstrategy
import warnings

# ============= GUI tkinter packages
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
# ============= machine learning packages
# sklearn
import sklearn as sk
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
# scipy
from scipy.cluster.hierarchy import dendrogram, linkage
# others
import numpy as np
import pandas as pd
import pickle
# ============= data plot packages
from mpl_toolkits import mplot3d
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
# baseURL = "https://aqueduct-tech.customer.cloud.microstrategy.com/MicroStrategyLibrary/api"
baseURL = "https://aqueduct.microstrategy.com/MicroStrategyLibrary/api"
projName = "Rally Analytics"

fields = ('LDAP Login Name:', 'Password:')
default_txt = ('rehuang', '**********')
# usrname = "rehuang"
# passwd = "38@kickboxing"

##############################################################
def model_kmeans(k, x_scaled):
    global labels

    print("\nK-means Clustering ... [" + str(len(x_scaled)) + ' training samples -> ' + str(k) + ' clusters]')
    # k-mean clustering
    model = KMeans(n_clusters=k, init='k-means++', max_iter=10000, n_init=1)
    labels = model.fit_predict(x_scaled)
    print('Done! Final cluster numbers = ', len(np.unique(labels)))
    do_dataframe()

##############################################################
def model_hc(k, x_scaled):
    global labels

    print("\nHieratical Clustering ... [" + str(len(x_scaled)) + ' training samples -> ' + str(k) + ' clusters]')
    model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    labels = model.fit_predict(x_scaled)
    print('Done! Final cluster numbers = ', len(np.unique(labels)))
    do_dataframe()

##############################################################
def model_sc(k, x_scaled):
    global labels

    print("\nSpectral Clustering ... [" + str(len(x_scaled)) + ' training samples -> ' + str(k) + ' clusters]')
    model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')
    labels = model.fit_predict(x_scaled)
    print('Done! Final cluster numbers = ', len(np.unique(labels)))
    do_dataframe()

##############################################################
def model_gmm(k, x_scaled):
    global labels

    print("\nGaussian Mixture Models ... [" + str(len(x_scaled)) + ' training samples -> ' + str(k) + ' clusters]')
    model = GaussianMixture(n_components=k)
    labels = model.fit_predict(x_scaled)
    print('Done! Final cluster numbers = ', len(np.unique(labels)))
    do_dataframe()

##############################################################
def model_dpgmm(k, x_scaled):
    global labels

    print("\nDirichlet Process Gaussian Mixture ... [" + str(len(x_scaled)) + ' training samples -> ' + str(k) + ' initial clusters]')
    model = BayesianGaussianMixture(n_components=k, covariance_type='full')
    labels = model.fit_predict(x_scaled)
    print('Done! Final cluster numbers = ', len(np.unique(labels)))
    do_dataframe()

##############################################################
def do_plotting():
    global outlier_x
    global x_scaled


    print('Plotting the results ...')
    nb_dim = len(x_scaled[0])
    outlier_x_ary = np.array(outlier_x)
    #
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')

    if nb_dim == 2:
        plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=labels,  s=4, cmap='viridis')
    else:
        ax.scatter3D(x_scaled[:, 0], x_scaled[:, 1], x_scaled[:, 2], c=labels,  s=2, cmap='viridis')
        ax.scatter3D(outlier_x_ary[:, 0], outlier_x_ary[:, 1], outlier_x_ary[:, 2], c='r', s=30)

    fig = plt.figure(2)
    ax = plt.axes(projection='3d')

    pca_x_ary = np.array(pca_x)
    outlier_pca_x_ary = np.array(outlier_pca_x)

    if nb_dim == 2:
        ax.scatter(pca_x_ary[:, 0], pca_x_ary[:, 1], c=labels, s=2, cmap='viridis')
        ax.scatter(outlier_pca_x_ary[:, 0], outlier_pca_x_ary[:, 1], c='r', s=30, marker='s')
    else:
        ax.scatter3D(pca_x_ary[:, 0], pca_x_ary[:, 1], pca_x_ary[:, 2], c=labels, s=2, cmap='viridis')
        ax.scatter3D(outlier_pca_x_ary[:, 0], outlier_pca_x_ary[:, 1], outlier_pca_x_ary[:, 2], c='r', s=30, marker='o')

    # rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    # rot_animation.save('path/rotation.gif', dpi=80, writer='imagemagick')
    plt.show()

##############################################################
def do_dataframe():
    global df_data2
    global df_data
    global this_prefix
    global outlier

    this_prefix = entry13.get()
    outfilename = this_prefix + '_clusters.csv'
    print('Generate dataframe ...' + outfilename)
    df_label = pd.DataFrame({'LABELS': labels})
    df_outlier = pd.DataFrame({'IS OUTLIER': outlier})

    df_data2 = pd.concat([df_data, df_label], axis=1)
    df_data2.to_csv(outfilename, index=False)

##############################################################
def do_outlier():
    global x_scaled
    global outlier

    threshold = w.get()

    # Whitening Transformation
    pca = PCA(n_components=3,svd_solver='full', whiten=True)

    # Find unique group labels
    uniq_label = np.unique(labels)

    nb_clusters = len(uniq_label)
    nb_samples = len(x_scaled)
    nb_dim = len(x_scaled[0])

    # initialization variable arrays
    outlier = [0]*nb_samples
    pca_x = [a[:] for a in [[0] * nb_dim] * nb_samples]
    sigma_pca_x = [a[:] for a in [[0] * nb_dim] * nb_clusters]
    sigma_x = [a[:] for a in [[0] * nb_dim] * nb_clusters]
    mu_pca_x = [a[:] for a in [[0] * nb_dim] * nb_clusters]
    mu_x = [a[:] for a in [[0] * nb_dim] * nb_clusters]

    for i in range(nb_clusters):
        idx1 = np.where(labels == uniq_label[i])[0]
        x_inCluster = x_scaled[idx1]
        pca_x_inCluster = pca.fit_transform(x_inCluster)
        m = len(pca_x_inCluster)
        for j in range(m):
            pca_x[idx1[j]] = pca_x_inCluster[j].tolist()

        mu_x[i] = x_inCluster.mean(axis=0)
        sigma_x[i] = x_inCluster.std(axis=0)

        mu_pca_x[i] = pca_x_inCluster.mean(axis=0)
        sigma_pca_x[i] = pca_x_inCluster.std(axis=0)

        idx_temp = np.where(abs(pca_x_inCluster) >= threshold)
        idx2 = np.unique(idx_temp[0])
        for u in idx1[idx2].tolist():
            outlier[u] = 1

    nb_outliers = sum(outlier)
    print('Found number of outliers: ' + str(nb_outliers))

    outlier_x = [x[:] for x in [[0] * nb_dim] * nb_outliers]
    outlier_pca_x = [x[:] for x in [[0] * nb_dim] * nb_outliers]
    c = 0

    for i in range(nb_samples):
        if outlier[i] == 1:
            outlier_x[c] = x_scaled[i].tolist()
            outlier_pca_x[c] = pca_x[i]
            c = c + 1

    do_plotting()

##############################################################
def get_data():
    global x_scaled
    global df_data

    model_name = []
    clust_no = 0

    infilename = askopenfilename(initialdir="/", title="Select Training Data", filetypes=((".csv", "*.csv"), (".sav", "*.sav"), ("all files", "*.*")))

    blz.tic()
    print('Selected data input: ', infilename)

    if infilename[-4:] == '.csv':
        df_data = pd.read_csv( infilename)
        x = df_data.values   # returns a numpy array
    elif infilename[-4:] == '.sav':
        x = pickle.load(open(infilename, 'rb'))

    min_max_scaler = preprocessing.MinMaxScaler()
    # normalization
    x_scaled = min_max_scaler.fit_transform(x)
    # f = pandas.DataFrame(x_scaled)

    blz.toc()
    print('\x1b[1;33m' + 'Done with [Data Loading].' + '\x1b[0m')

##############################################################
def do_clustering():
    global k
    global x_scaled

    sel_model()
    print(model_index)
    if model_index == 0:
        model_kmeans(k, x_scaled)
    elif model_index == 1:
        model_hc(k, x_scaled)
    elif model_index == 2:
        model_sc(k, x_scaled)
    elif model_index == 3:
        model_gmm(k, x_scaled)
    elif model_index == 4:
        model_dpgmm(k, x_scaled)
    else:
        print('Invalid Model')

###  GUI #######################################################
def close_window ():
    root.destroy()

###  GUI #######################################################
def sel_model():
    global model_name
    global model_index
    global var1
    global k

    k = 0
    model_name = var1.get()
    model_index = model_options.index(model_name)
    print(model_index)
    k = int(entry1.get())
    print("Clustering Algorithm:  " + model_name)
    if k_sel==0:
        print('Number of Clusters = '+ str(k))
    else:
        print('Self-adaptive to find optimal Clusters.' )

###  GUI ######################################################
def create1():
    global entry1
    entry1.config(state = 'normal')

def destroy1():
    global entry1
    entry1.config(state = 'disabled')

###  GUI #######################################################
def sel():
    global k_sel

    k_sel =  var2.get()
    print("Your selected the option " + str(k_sel))
    # label.config(text=selection)
    if  k_sel ==0:
        create1()
    else:
        destroy1()


###  GUI #######################################################
def rotate(angle):

    ax.view_init(azim=angle)

############################
def push2cube():

    global value_list
    global tst_out
    global out_filename1
    global this_prefix

    usrname = entry11.get()
    passwd = entry12.get()

    print('\nStarting ' + '\x1b[6;30;42m' + 'PUSH DATAFRAME TO MSTR CUBE: ' + '\x1b[0m')
    blz.tic()

    datasetName = this_prefix + '_cube'
    tableName = this_prefix + '_table'
    cubeinfo_name = this_prefix + '_cubeinfo'

    # Authentication request and connect to the Rally Analytics project
    conn = microstrategy.Connection(base_url=baseURL, login_mode=16,username=usrname, password=passwd, project_name=projName)
    conn.connect()

    print("Connect to " + baseURL)

    # if the cube does not exist, acquire Data Set Id & Table Id, and create a new cube
    newDatasetId, newTableId = conn.create_dataset(data_frame=df_data2, dataset_name=datasetName, table_name=tableName)
    # Store Data Set Id and Table Id locally
    cubeInfoFile = open(cubeinfo_name, 'w')
    cubeInfoFile.write(newDatasetId + '\n')
    cubeInfoFile.write(newTableId)
    cubeInfoFile.close()
    print("CREATE Cube on URL: " + baseURL[:-25])
    print('[ Dataset Name: ' + datasetName + ' \ Cube ID = ' + newDatasetId + ']   [Table Name: ' + tableName + ' \ Table ID = ' + newTableId + ' ]')
    blz.toc()
    print('\x1b[1;33m' + "Done with [Output to MSTR Cube for Dossier Reporting (without PA)]" + '\x1b[0m')

### MAIN ###########################################################
if __name__ == '__main__':

    global entry
    global k_sel
    global this_prefix

    model_options = ["K-means Clustering",
                     "Hirratical Clustering",
                     "Spectral Clustering",
                     "Gaussian Mixture Modeling (GMMMs) - EM",
                     "Bayesian Gaussian Mixture Modeling (DPGMM) - Dirichlet Process",
                     "(coming soon) Self Organizing Maps (SOMs) Neural Networks",
                     "(coming soon) Deep-Clustering (Convolutional Neural Network based model)"]

    k_sel = 0
    row0 = 0
    bottonwidth = 60
    # model_functions = [model_kmeans, model_hc, model_sc, model_gmm, model_dggmm, model_som, model_dlc]

    try:
        root.destroy()
    except:
        pass
    root = Tk()
    root.geometry('460x800')
    root.title('ML Outlier Detection Tool')

    b1 = tk.Button(root, text='Select Data Source  ...', command = get_data, bg="peach puff")
    b1.config(width = bottonwidth)
    dirname = tk.Button(b1)
    b1.grid(row= row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1
    #
    s= ttk.Separator(root)
    s.grid(row= row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1
    #
    l1 = Label(root, text="Choose Clustering Algorithm ...")
    l1.grid(row= row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1
    #
    var1 = StringVar(root)
    var1.set(model_options[0])  # default value
    p1 = OptionMenu(root, var1, *model_options)
    p1.grid(row= row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1
    #
    var2 = IntVar()
    var2.set(0)
    R1 = Radiobutton(root, text="No. of Clusters", variable=var2, value=0, command=sel)
    R1.grid(row= row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)

    entry1 = tk.Entry(state='normal')
    entry1.grid(row=row0, column=1, columnspan=2, sticky="w", padx=6, pady=12)
    row0 = row0 + 1
    #
    R2 = Radiobutton(root, text="Self-Adaptive", variable=var2, value=1, command=sel)
    R2.grid(row= row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)
    row0 = row0 + 1
    #
    b3 = tk.Button(root, text='Start Model Training...',  command = do_clustering, bg="papaya whip")
    b3.grid(row= row0, column=0, columnspan=2,sticky="ew",  padx=6, pady=12)
    row0 = row0 + 1
    #
    s = ttk.Separator(root)
    s.grid(row= row0, column=0, columnspan=2, sticky="ew", padx=6, pady=12)
    row0 = row0 + 1
    #
    l3 = Label(root, text="Outlier Threshold (Sigma):")
    l3.grid(row= row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)
    #
    w = Scale(root, from_=1, to=6, tickinterval=10, orient=HORIZONTAL)
    w.grid(row=row0, column=1, columnspan=2, sticky="ew", padx=6, pady=12)
    w.set(3)
    row0 = row0 + 1
    #
    b4 = tk.Button(root, text='Idedntifying Outliers... ',  command = do_outlier, bg="papaya whip")
    b4.grid(row= row0, column=0, columnspan=2,sticky="ew",  padx=6, pady=12)
    row0 = row0 + 1

    #
    l11 = Label(root, text="User Name:")
    l11.grid(row= row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)
    entry11 = tk.Entry(state='normal')
    entry11.grid(row=row0, column=1, columnspan=2, sticky="w", padx=6, pady=12)
    row0 = row0 + 1
    #
    l12 = Label(root, text="Password:")
    l12.grid(row= row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)
    entry12 = tk.Entry(state='normal')
    entry12.grid(row=row0, column=1, columnspan=2, sticky="w", padx=6, pady=12)
    row0 = row0 + 1
    #
    l13 = Label(root, text="Cube Prefix:")
    l13.grid(row=row0, column=0, columnspan=2, sticky="w", padx=6, pady=12)
    #
    entry13 = tk.Entry(state='normal')
    entry13.grid(row=row0, column=1, columnspan=2, sticky="w", padx=6, pady=12)
    row0 = row0 + 1
    #
    b5 = tk.Button(root, text='Output to MSTR Cube for Dossier... ', command=push2cube, bg = "light pink")
    b5.grid(row=row0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
    row0 = row0 + 1
    #
    b9 = tk.Button(root, text='Done ...', command= close_window, bg= "ivory2")
    b9.grid(row= row0, column=0, columnspan=2,sticky="ew",  padx=6, pady=6)
    root.mainloop()
