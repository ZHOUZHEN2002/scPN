#preprocess
import scanpy as sc
import scvelo as scv
raw_data_path='/data1/zzhou/pseudotime/scPN/data/oligo_lite.h5ad'
adata=sc.read(raw_data_path,sparse=True)
import numpy as np
np.bool=np.bool_
np.int=np.int_
np.object = object
import scvelo as scv
import scanpy as sc
scv.logging.print_version()
scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.presenter_view = True  # set max width size for presenter view
scv.settings.set_figure_params('scvelo')  # for beautified visualization
scv.set_figure_params()

scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=1000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
adata_1000=sc.pp.subsample(adata, fraction=None, n_obs=1000, random_state=0, copy=True)
sc.tl.leiden(adata_1000, resolution=0.1)
sc.pl.umap(adata_1000)
adata_1000.write('/data1/zzhou/pseudotime/scPN/data/oligo_lite_1000.h5ad')
#add gene matrix
data_path='/data1/zzhou/pseudotime/scPN/data/oligo_lite_1000.h5ad'
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
                

neighbors=100
n_filename = '/data1/zzhou/pseudotime/scPN/data/oligo_answer.csv'
def optimize_matrix_A(route,cluster='0', adata_file='/data1/zzhou/pseudotime/scPN/data/oligo_lite_1000.h5ad',epochs=50000, lr=0.001):
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
        loss = matrix_norm+W_hadamard_A_norm
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
if not os.path.exists(n_filename):
    A = np.zeros((x.shape[1], x.shape[1]))
    best_route = None
    for iteration in range(10):
        distance_matrix = getdistance(A, x)
        route, _ = solve_tsp_simulated_annealing(distance_matrix)
        route, _ = solve_tsp_local_search(distance_matrix)
        A = optimize_matrix_A(route, cluster=cluster, adata_file=adata_file)
    
    # Save the best route for the current cluster
    route_df = pd.DataFrame(route)
    route_df.to_csv('/data1/zzhou/pseudotime/scPN/data/oligo_answer.csv', index=False)
adata=sc.read('/data1/zzhou/pseudotime/scPN/data/oligo_lite_1000.h5ad')
Answer=pd.read_csv(n_filename).values
n=len(Answer)
q=0
U=np.zeros(n)
for i in range(n):
    
    U[Answer[-i]-1]=1/n*q
    q+=1
adata.obs['latent_time']=U
scv.pp.neighbors(adata, n_neighbors=neighbors)
neighbor_matrix=adata.uns['neighbors']['indices']
neighbor_matrix=adata.uns['neighbors']['indices']
k=neighbors
U_KNN=np.zeros(n)
for i in range(n):
    for j in range(k):
        U_KNN[i]+=U[neighbor_matrix[i,j]]
    U_KNN[i]=U_KNN[i]/k
adata.obs['latent_time']=U_KNN/U_KNN.max()
scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=80,basis='X_umap')               