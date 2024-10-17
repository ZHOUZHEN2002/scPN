#preprocess
import scanpy as sc
import scvelo as scv
raw_data_path='/data1/zzhou/pseudotime/scPN/data/DentateGyrus.loom'
adata=sc.read_loom(raw_data_path,sparse=True)
import numpy as np
np.bool=np.bool_
np.int=np.int_
np.object = object
import scvelo as scv
import scanpy as sc
import pandas as pd
scv.logging.print_version()
scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.presenter_view = True  # set max width size for presenter view
scv.settings.set_figure_params('scvelo')  # for beautified visualization
scv.set_figure_params()

# adata_=adata[adata.obs['ClusterName'].isin(['CA','CA1_Sub','CA2-3-4','Granule','Nbl1','Nbl2','nIPC','RadialGlia','GlialProg','OPC','ImmAstro'])]
scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
sc.tl.umap(adata)
adata_3000=sc.pp.subsample(adata, fraction=None, n_obs=3000, random_state=0, copy=True)
sc.tl.leiden(adata_3000, resolution=0.1)
adata_3000.write('/data1/zzhou/pseudotime/scPN/data/adata3000_imputed_leiden11_ID.h5ad')
#add gene matrix
data_path='/data1/zzhou/pseudotime/scPN/data/adata3000_imputed_leiden11_ID.h5ad'
gene_path='/data1/zzhou/pseudotime/scPN/data/ChEA_2016.txt'
adata=sc.read(data_path)
gene_names = list(adata.var_names)
target_idx=np.zeros((len(gene_names),len(gene_names)))
gene_names= [name.upper() for name in gene_names]
with open(gene_path, "r") as f:
     for line in f.readlines():
         line = line.strip('\n')
         line = line.split('\t')
         TF_info = line[0]
         TF_name = TF_info.split(' ')[0]
         if not TF_name in gene_names:
            continue
         targets= line[2:]
         for target in targets:
             if target in gene_names:
                target_idx[gene_names.index(TF_name),gene_names.index(target)]=1
import torch
import torch.optim as optim
target_idx = torch.tensor(target_idx, device='cuda',dtype=torch.float32)
x=adata_3000.X.toarray()
def optimize_matrix_A(route,cluster='0', adata_file='/data1/zzhou/pseudotime/scPN/data/adata3000_imputed_leiden11_ID.h5ad',epochs=50000, lr=0.001):
    # 加载数据并处理
    adata = sc.read(adata_file)
    adata_0 = adata[adata.obs['leiden'] == cluster].X
    x = adata_0.toarray()
    n, m = x.shape
    X1=torch.tensor(x,device='cuda',dtype=torch.float32)


    # 初始化需要优化的矩阵A和矩阵W
    A = np.ones((m, m))  # 根据具体需求初始化
#     target_idx = np.random.randn(m, m)  # 这个变量需要定义或从数据中获取
    W = target_idx.cpu().detach().numpy()  # 假设W矩阵和target_idx相关

    # 根据W矩阵定义掩码矩阵
    mask = np.where(W != 0, 1, 0)
    mask_gpu = torch.tensor(mask, device='cuda', dtype=torch.float64)

    # 确保矩阵A中对应W为零的位置也为零
    A *= mask
    A_gpu = torch.tensor(A, requires_grad=True, device='cuda', dtype=torch.float64)

    # 初始化优化器
    optimizer = optim.Adam([A_gpu], lr=lr)

    # 定义目标函数
    def objective_function_A(A, route):
        scalar = torch.tensor(1, device='cuda', dtype=torch.float64)
        target_idx_gpu = torch.tensor(target_idx, device='cuda', dtype=torch.float64)
        dy_gpu = torch.tensor(dy(route, x), device='cuda', dtype=torch.float64)
        Ones = torch.ones(m, m, device='cuda', dtype=torch.float64)
        W_gpu = torch.tensor(W, device='cuda', dtype=torch.float64)
        X1=torch.tensor(x,device='cuda')

#         distance_matrix=distance_matrix2.cpu().detach().numpy()
#         total_distance = 0.0
#         for i in range(len(route) - 1):
#             total_distance += distance_matrix[route[i]-1, route[i + 1]-1]
        W_reciprocal = torch.reciprocal(100 * target_idx_gpu + scalar * Ones)

        x_t = np.zeros((n, m))
        for i in range(n):
            x_t[i] = x[route[i] - 1, :]
        x_t_gpu = torch.tensor(x_t[2:n-3, :], device='cuda', dtype=torch.float64)
#         distance_matrix2=torch.cdist(X1A,X1A,p=2)
        X1A_gpu = torch.mm(x_t_gpu, A)
        W_hadamard_A_gpu = torch.mul(W_reciprocal, A)
        matrix = dy_gpu - X1A_gpu
        matrix_norm = torch.norm(matrix, p='fro')
        W_hadamard_A_norm = torch.norm(W_hadamard_A_gpu, p='fro')
        x_t_minus=torch.diff(x_t_gpu, dim=0)
        total_distance=torch.norm(torch.mm(x_t_minus,A),p='fro')
        loss = matrix_norm+W_hadamard_A_norm+total_distance
        return loss

    def dy(route, x):
        m, n = x.shape
        dy1 = np.zeros((m-5, n))
        deltat = 1 / (m-1)
        for t in range(2, m - 3):
            dy1[t - 2, :] = (8 * x[route[t + 1]-1, :] - 8 * x[route[t - 1]-1, :] + x[route[t - 2]-1, :] - x[route[t + 2]-1, :]) / (12 * deltat)
        return dy1

    # 进行优化迭代
    for i in range(epochs):
        optimizer.zero_grad()  # 每次迭代前将梯度清零
        loss = objective_function_A(A_gpu, route)
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新矩阵A的数值
        
        # 应用掩码，使非优化元素保持为零
        with torch.no_grad():
            A_gpu *= mask_gpu
        
        if i % 1000 == 0:  # 每1000次迭代打印一次损失
            print(f"Iteration {i}, loss: {loss.item()}")

#     print("Optimized matrix A:")
    return A_gpu.cpu().detach().numpy()

def getdistance(A,adata):
    u, s, v = np.linalg.svd(A)
    n,m=adata.shape
    X1=torch.tensor(adata,device='cuda',dtype=torch.float32)

    A=torch.tensor(A,device='cuda',dtype=torch.float32)
    X1A=torch.mm(X1, A)
# print(XA.shape,XB.shape,type(XA),type(XB))
    distance_matrix1=torch.cdist(X1, X1, p=2)
    distance_matrix2=torch.dist(X1A,X1A,p=2)
    distance_matrix=distance_matrix1+distance_matrix2/(s[0]+1)
    distance=np.array(distance_matrix.cpu())
    return distance
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.heuristics import solve_tsp_local_search
A=np.zeros((x.shape[1],x.shape[1]))
adata_file = '/data1/zzhou/pseudotime/scPN/data/adata3000_imputed_leiden11_ID.h5ad'
n_filename = '/data1/zzhou/pseudotime/scPN/data/Dentateanswer/time_embedding_KNN.csv'
adata.obs['ID']=np.arange(3000)
if not os.path.exists(n_filename):
    for cluster in map(str, range(8)):  # Convert numbers to string
        A = np.zeros((x.shape[1], x.shape[1]))
        best_route = None
        best_distance = np.inf  # Initialize with infinity
        for iteration in range(10):
            distance_matrix = getdistance(A, x)
            route, distance = solve_tsp_simulated_annealing(distance_matrix)
            route, distance = solve_tsp_local_search(distance_matrix)
            
            # Only update if the new distance is smaller
            if distance < best_distance:
                best_distance = distance
                best_route = route
                A = optimize_matrix_A(route, cluster=cluster, adata_file=adata_file)

    
        # Save the best route for the current cluster
        route_df = pd.DataFrame(best_route)
        route_df.to_csv(f'/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_{cluster}_route.csv', index=False)
        A_df = pd.DataFrame(A)
        A_df.to_csv(f'/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_{cluster}_connection.csv', index=False)
    Answer5=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_5_route.csv',header=None).values
    adata_5=adata[adata.obs['leiden']=='5']
    adata_5list=[]
    for i in range(len(Answer5)):
        adata_5list.append(int(adata_5[Answer5[i]-1].obs['ID']))
    n5=len(Answer5)
    U_total=np.zeros(3000)
    U=np.zeros(n5)
    adata.obs['latent_time']=U_total

    for i in range(len(adata_5list)):
        U_total[adata_5list[-i]]=i/3000
        adata.obs['latent_time']=U_total
    adata_2=adata[adata.obs['leiden']=='2']
    adata_2list=[]
    Answer2=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_2_route.csv',header=None).values
    for i in range(len(Answer2)):
        adata_2list.append(int(adata_2[Answer2[i]-1].obs['ID']))
    n2=len(Answer2)

    U=np.zeros(n2)
    adata.obs['latent_time']=U_total

    for i in range(len(adata_2list)):
        U_total[adata_2list[i]]=i/n2
        adata.obs['latent_time']=U_total
    adata_0=adata[adata.obs['leiden']=='0']
    adata_0list=[]
    Answer0=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_0_route.csv',header=None).values
    for i in range(len(Answer0)):
        adata_0list.append(int(adata_0[Answer0[i]-1].obs['ID']))
    n0=len(Answer0)

    U=np.zeros(n0)
    adata.obs['latent_time']=U_total

    for i in range(len(adata_0list)):
        U_total[adata_0list[i]]=(i+n5)/(n5+n0)
        adata.obs['latent_time']=U_total
    adata_3=adata[adata.obs['leiden']=='3']
    adata_3list=[]
    Answer3=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_3_route.csv',header=None).values
    for i in range(len(Answer3)):
        adata_3list.append(int(adata_3[Answer3[i]-1].obs['ID']))
    n3=len(Answer3)

    U=np.zeros(n3)
    adata.obs['latent_time']=U_total

    for i in range(len(adata_3list)):
        U_total[adata_3list[-i]]=(i)/(n3)
        adata.obs['latent_time']=U_total
    adata_7=adata[adata.obs['leiden']=='7']
    Answer7=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_7_route.csv',header=None).values
    adata_7list=[]
    for i in range(len(Answer7)):
        adata_7list.append(int(adata_7[Answer7[i]-1].obs['ID']))
    n7=len(Answer7)


    U=np.zeros(n7)
    adata.obs['latent_time']=U_total

    for i in range(len(adata_7list)):
        U_total[adata_7list[-i]]=i/n7
        adata.obs['latent_time']=U_total
    adata_1=adata[adata.obs['leiden']=='1']
    adata_4=adata[adata.obs['leiden']=='4']
    Answer1=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_1_route.csv',header=None).values
    adata_1list=[]
    for i in range(len(Answer1)):
        adata_1list.append(int(adata_1[Answer1[i]-1].obs['ID']))
    n1=len(Answer1)
    n4=adata_4.shape[0]
    n=adata.shape[0]
    adata.obs['latent_time']=U_total
    for i in range(len(adata_1list)):
        U_total[adata_1list[-i]]=(i+n5)/(n1+n5+n4)
        adata.obs['latent_time']=U_total
    Answer4=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_4_route.csv',header=None).values
    adata_4=adata[adata.obs['leiden']=='4']
    adata_4list=[]
    for i in range(len(Answer4)):
        adata_4list.append(int(adata_4[Answer4[i]-1].obs['ID']))
    n4=len(Answer4)

    U=np.zeros(n4)
    adata.obs['latent_time']=U_total

    for i in range(len(adata_4list)):
        U_total[adata_4list[i]]=(i+n5+n1)/(n1+n5+n4)
        adata.obs['latent_time']=U_total
    adata_6=adata[adata.obs['leiden']=='6']
    Answer6=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/adata_6_route.csv',header=None).values
    adata_6list=[]
    for i in range(len(Answer6)):
        adata_6list.append(int(adata_6[Answer6[i]-1].obs['ID']))
    n6=len(Answer6)
    U=np.zeros(n6)
    adata.obs['latent_time']=U_total

    for i in range(len(adata_6list)):
        U_total[adata_6list[-i]]=(i+n1+n4)/(n6+n1+n4)
        adata.obs['latent_time']=U_total
    scv.pp.neighbors(adata, n_neighbors=100)
    neighbor_matrix=adata.uns['neighbors']['indices']
    k=100
    U_KNN=np.zeros(3000)
    for i in range(3000):
        for j in range(k):
            U_KNN[i]+=U_total[neighbor_matrix[i,j]]
        U_KNN[i]=U_KNN[i]/k
    adata.obs['latent_time']=U_KNN
    max_7=adata[adata.obs['leiden']=='7'].obs['latent_time'].max()
    max_2=adata[adata.obs['leiden']=='2'].obs['latent_time'].max()
    max_3=adata[adata.obs['leiden']=='3'].obs['latent_time'].max()
    max_0=adata[adata.obs['leiden']=='0'].obs['latent_time'].max()
    max_1=adata[adata.obs['leiden']=='1'].obs['latent_time'].max()
    max_4=adata[adata.obs['leiden']=='4'].obs['latent_time'].max()
    for i in range(3000):
        if list(adata[i].obs['leiden'])==['7']:
            U_KNN[i]=U_KNN[i]/max_7
        elif list(adata[i].obs['leiden'])==['2']:
            U_KNN[i]=U_KNN[i]/max_2
        elif list(adata[i].obs['leiden'])==['3']:
            U_KNN[i]=U_KNN[i]/max_3
        elif list(adata[i].obs['leiden'])==['0']:
            U_KNN[i]=U_KNN[i]/max_0
        elif list(adata[i].obs['leiden'])==['1']:
            U_KNN[i]=U_KNN[i]/max_1
        elif list(adata[i].obs['leiden'])==['4']:
            U_KNN[i]=U_KNN[i]/max_4
    adata.obs['latent_time']=U_KNN
    U_KNN_df = pd.DataFrame(U_KNN)
    U_KNN_df.to_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/time_embedding_KNN.csv', index=False)
    scv.pl.scatter(adata, color='latent_time', color_map='gnuplot',size=80, basis='X_umap')
adata=sc.read('/data1/zzhou/pseudotime/scPN/data/adata3000_imputed_leiden11_ID.h5ad')
Answer=pd.read_csv('/data1/zzhou/pseudotime/scPN/data/Dentateanswer/time_embedding_KNN.csv').values
n=len(Answer)
adata.obs['latent_time']=Answer
scv.pl.scatter(adata, color='latent_time', color_map='gnuplot',size=80, basis='X_umap')